"""Compact W8A16 support for DeepSeek-style absorbed-MLA ``kv_b_proj``.

The immutable checkpoint stores one backend-neutral representation for both
the Triton and grouped-Marlin implementations:

* ``kc_qweight``: GPTQ-packed INT32 ``[H, 192 / 4, 512]``;
* ``kc_scales``: BF16 ``[H, 1, 512]``;
* ``vc_qweight``: GPTQ-packed INT32 ``[H, 512 / 4, 256]``;
* ``vc_scales``: BF16 ``[H, 1, 256]``.

The four bytes in each INT32 word are biased signed-INT8 values for adjacent K
lanes in little-endian order.  Triton bit-extracts those words directly.
Marlin performs a compact-to-compact layout/scale permutation once at load.
Neither backend materializes a persistent BF16 weight.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

import torch

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_BACKEND_ENVIRONMENT_VARIABLE = "SGLANG_MLA_KV_B_W8_BACKEND"
_SUPPORTED_BACKENDS = frozenset({"auto", "marlin", "triton"})
_MARLIN_WORKSPACE_SLOTS = 4


def _fake_grouped_marlin(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
    input_features: int,
    output_features: int,
    block_size_m: int,
) -> torch.Tensor:
    del qweight, scales, workspace, input_features, block_size_m
    return x.new_empty((x.shape[1], x.shape[0], output_features))


@register_custom_op(
    op_name="mla_kv_b_grouped_marlin_w8a16",
    mutates_args=["workspace"],
    fake_impl=_fake_grouped_marlin,
)
def _grouped_marlin_w8a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
    input_features: int,
    output_features: int,
    block_size_m: int,
) -> torch.Tensor:
    """Run one grouped Marlin launch, treating each head as an expert."""

    if x.ndim != 3:
        raise ValueError(
            "grouped MLA Marlin expects [tokens, heads, channels], "
            f"got shape={tuple(x.shape)}"
        )
    token_count, head_count, observed_input_features = x.shape
    expected_qweight_shape = (
        head_count,
        input_features // 16,
        output_features * 4,
    )
    if observed_input_features != input_features:
        raise ValueError(
            "grouped MLA Marlin input K mismatch: "
            f"shape={tuple(x.shape)}, expected_k={input_features}"
        )
    if tuple(qweight.shape) != expected_qweight_shape:
        raise ValueError(
            "grouped MLA Marlin qweight shape mismatch: "
            f"actual={tuple(qweight.shape)}, expected={expected_qweight_shape}"
        )
    if tuple(scales.shape) != (head_count, 1, output_features):
        raise ValueError(
            "grouped MLA Marlin scale shape mismatch: "
            f"actual={tuple(scales.shape)}, "
            f"expected={(head_count, 1, output_features)}"
        )
    if (
        x.dtype != torch.bfloat16
        or qweight.dtype != torch.int32
        or scales.dtype != torch.bfloat16
    ):
        raise TypeError(
            "grouped MLA Marlin requires BF16 activations/scales and INT32 "
            f"qweight, got x={x.dtype}, qweight={qweight.dtype}, "
            f"scales={scales.dtype}"
        )
    if not x.is_cuda or x.device != qweight.device or x.device != scales.device:
        raise ValueError("grouped MLA Marlin tensors must share one CUDA device")
    if workspace.device != x.device or workspace.dtype != torch.int32:
        raise ValueError("grouped MLA Marlin workspace has the wrong device or dtype")
    if token_count == 0:
        return x.new_empty((head_count, 0, output_features))

    from sgl_kernel.scalar_type import scalar_types

    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    rows = x.transpose(0, 1).contiguous().view(token_count * head_count, input_features)
    head_ids = (
        torch.arange(head_count, dtype=torch.int32, device=x.device)
        .repeat_interleave(token_count)
        .reshape(-1, 1)
        .contiguous()
    )
    topk_weights = torch.ones(
        (rows.shape[0], 1),
        dtype=torch.float32,
        device=x.device,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        head_ids,
        block_size_m,
        head_count,
    )
    output = moe_wna16_marlin_gemm(
        rows,
        None,
        qweight,
        None,
        scales,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=scalar_types.uint8b128,
        size_m=rows.shape[0],
        size_n=output_features,
        size_k=input_features,
        is_k_full=True,
        use_atomic_add=torch.cuda.get_device_capability(x.device)[0] >= 9,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    return output.view(head_count, token_count, output_features)


class GPTQMLAKVW8Method(LinearMethodBase):
    """Consume the backend-neutral per-head GPTQ W8 MLA checkpoint layout."""

    is_mla_kv_b_w8 = True
    requires_mla_absorb = True

    def __init__(self, quant_config: Any, format_config: Mapping[str, object]) -> None:
        self.quant_config = quant_config
        self.format_config = dict(format_config)
        expected = {
            "bits": 8,
            "block_size_m": 8,
            "format": "gptq_packed_rows_per_head_v1",
            "implicit_bias": 128,
            "kv_lora_rank": 512,
            "num_attention_heads": 64,
            "pack_axis": "K",
            "pack_order": "little_endian_k_lanes_0_1_2_3",
            "qk_nope_head_dim": 192,
            "scale_compute_dtype": "float32",
            "scale_dtype": "bfloat16",
            "v_head_dim": 256,
        }
        mismatches = {
            key: (self.format_config.get(key), value)
            for key, value in expected.items()
            if self.format_config.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"invalid Exo MLA kv_b W8 format metadata: mismatches={mismatches}"
            )
        if (
            quant_config.weight_bits != 8
            or quant_config.group_size != -1
            or quant_config.desc_act
            or not quant_config.is_sym
            or quant_config.pack_factor != 4
        ):
            raise ValueError(
                "Exo MLA kv_b W8 requires symmetric channelwise W8, "
                "four K lanes per INT32, and desc_act=false"
            )
        configured_backend = os.getenv(
            _BACKEND_ENVIRONMENT_VARIABLE,
            "auto",
        ).lower()
        if configured_backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"{_BACKEND_ENVIRONMENT_VARIABLE} must be one of "
                f"{sorted(_SUPPORTED_BACKENDS)}, got {configured_backend!r}"
            )
        self.backend = "triton" if configured_backend == "auto" else configured_backend
        self._stream_slots: dict[int, int] = {}

    @property
    def qk_nope_head_dim(self) -> int:
        return int(self.format_config["qk_nope_head_dim"])

    @property
    def v_head_dim(self) -> int:
        return int(self.format_config["v_head_dim"])

    @property
    def kv_lora_rank(self) -> int:
        return int(self.format_config["kv_lora_rank"])

    @property
    def num_heads(self) -> int:
        return int(self.format_config["num_attention_heads"])

    @property
    def block_size_m(self) -> int:
        return int(self.format_config["block_size_m"])

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "Exo MLA kv_b W8 currently requires BF16 activations/scales, "
                f"got {params_dtype}"
            )
        if input_size_per_partition != self.kv_lora_rank:
            raise ValueError(
                "kv_b input dimension differs from format metadata: "
                f"{input_size_per_partition} != {self.kv_lora_rank}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        head_width = self.qk_nope_head_dim + self.v_head_dim
        expected_output_size = self.num_heads * head_width
        if output_size != expected_output_size:
            raise ValueError(
                "kv_b global output dimension differs from format metadata: "
                f"{output_size} != {expected_output_size}"
            )
        if output_size_per_partition % head_width != 0:
            raise ValueError(
                "kv_b local output dimension is not an integral number of "
                f"MLA heads: local={output_size_per_partition}, "
                f"head_width={head_width}"
            )
        self.num_local_heads = output_size_per_partition // head_width
        if self.num_heads % self.num_local_heads != 0:
            raise ValueError(
                "MLA attention heads cannot be evenly tensor parallelized: "
                f"global={self.num_heads}, local={self.num_local_heads}"
            )
        weight_loader = extra_weight_attrs["weight_loader"]

        def packed_weight(
            input_features: int,
            output_features: int,
        ) -> PackedvLLMParameter:
            return PackedvLLMParameter(
                data=torch.empty(
                    self.num_local_heads,
                    input_features // 4,
                    output_features,
                    dtype=torch.int32,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=4,
                weight_loader=weight_loader,
            )

        def scales(output_features: int) -> ChannelQuantScaleParameter:
            return ChannelQuantScaleParameter(
                data=torch.empty(
                    self.num_local_heads,
                    1,
                    output_features,
                    dtype=params_dtype,
                ),
                output_dim=0,
                weight_loader=weight_loader,
            )

        layer.register_parameter(
            "kc_qweight",
            packed_weight(self.qk_nope_head_dim, self.kv_lora_rank),
        )
        layer.register_parameter("kc_scales", scales(self.kv_lora_rank))
        layer.register_parameter(
            "vc_qweight",
            packed_weight(self.kv_lora_rank, self.v_head_dim),
        )
        layer.register_parameter("vc_scales", scales(self.v_head_dim))

    @staticmethod
    def _replace_contiguous(layer: torch.nn.Module, name: str) -> None:
        value = getattr(layer, name)
        replace_parameter(
            layer,
            name,
            torch.nn.Parameter(value.data.contiguous(), requires_grad=False),
        )

    def _repack_for_marlin(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.gptq import (
            gptq_marlin_moe_repack,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_moe_permute_scales,
        )

        device = layer.kc_qweight.device
        empty_permutation = torch.empty(
            (self.num_local_heads, 0),
            dtype=torch.int32,
            device=device,
        )
        for stem, input_features, output_features in (
            ("kc", self.qk_nope_head_dim, self.kv_lora_rank),
            ("vc", self.kv_lora_rank, self.v_head_dim),
        ):
            repacked = gptq_marlin_moe_repack(
                getattr(layer, f"{stem}_qweight").contiguous(),
                empty_permutation,
                input_features,
                output_features,
                8,
            )
            permuted_scales = marlin_moe_permute_scales(
                getattr(layer, f"{stem}_scales").contiguous(),
                input_features,
                output_features,
                -1,
            )
            replace_parameter(layer, f"{stem}_qweight", repacked)
            replace_parameter(layer, f"{stem}_scales", permuted_scales)

        for stem in ("kc", "vc"):
            for slot in range(_MARLIN_WORKSPACE_SLOTS):
                layer.register_buffer(
                    f"{stem}_marlin_workspace_{slot}",
                    marlin_make_workspace(device, max_blocks_per_sm=4),
                    persistent=False,
                )
        default_stream_handle = int(torch.cuda.default_stream(device).cuda_stream)
        self._stream_slots[default_stream_handle] = 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.kc_qweight.device
        if device.type != "cuda":
            raise ValueError("Exo MLA kv_b W8 weights require a CUDA device")
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("Exo MLA kv_b W8 requires an Ampere-or-newer GPU")
        for name in ("kc_qweight", "kc_scales", "vc_qweight", "vc_scales"):
            self._replace_contiguous(layer, name)
        if self.backend == "marlin":
            self._repack_for_marlin(layer)
        else:
            from sglang.srt.layers.quantization.mla_kv_b_triton import (
                mla_kv_b_gptq_w8a16_bmm,
            )

            del mla_kv_b_gptq_w8a16_bmm
        logger.info(
            "Loaded compact MLA kv_b W8 backend=%s local_heads=%d",
            self.backend,
            self.num_local_heads,
        )

    def _workspace_for(
        self,
        layer: torch.nn.Module,
        stem: str,
    ) -> torch.Tensor:
        stream_handle = int(
            torch.cuda.current_stream(layer.kc_qweight.device).cuda_stream
        )
        slot = self._stream_slots.get(stream_handle)
        if slot is None:
            used_slots = set(self._stream_slots.values())
            available_slots = [
                candidate
                for candidate in range(_MARLIN_WORKSPACE_SLOTS)
                if candidate not in used_slots
            ]
            if not available_slots:
                raise RuntimeError(
                    "compact MLA kv_b Marlin observed more concurrent CUDA "
                    f"streams than its {_MARLIN_WORKSPACE_SLOTS}-slot workspace pool"
                )
            slot = available_slots[0]
            self._stream_slots[stream_handle] = slot
        return getattr(layer, f"{stem}_marlin_workspace_{slot}")

    def _apply(
        self,
        layer: torch.nn.Module,
        stem: str,
        x: torch.Tensor,
        *,
        input_features: int,
        output_features: int,
    ) -> torch.Tensor:
        qweight = getattr(layer, f"{stem}_qweight")
        scales = getattr(layer, f"{stem}_scales")
        if self.backend == "triton":
            from sglang.srt.layers.quantization.mla_kv_b_triton import (
                mla_kv_b_gptq_w8a16_bmm,
            )

            return mla_kv_b_gptq_w8a16_bmm(x, qweight, scales)
        return _grouped_marlin_w8a16(
            x,
            qweight,
            scales,
            self._workspace_for(layer, stem),
            input_features,
            output_features,
            self.block_size_m,
        )

    def apply_mla_k(self, layer: torch.nn.Module, q_nope: torch.Tensor) -> torch.Tensor:
        """Return the absorbed K projection in torch.bmm's [H, M, N] layout."""

        return self._apply(
            layer,
            "kc",
            q_nope,
            input_features=self.qk_nope_head_dim,
            output_features=self.kv_lora_rank,
        )

    def apply_mla_v(
        self,
        layer: torch.nn.Module,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return the value expansion flattened to [M, H * value_dim]."""

        return self.apply_mla_v_bmm(layer, attn_output).transpose(0, 1).flatten(1, 2)

    def apply_mla_v_bmm(
        self,
        layer: torch.nn.Module,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return the value expansion in torch.bmm's [H, M, N] layout."""

        return self._apply(
            layer,
            "vc",
            attn_output,
            input_features=self.kv_lora_rank,
            output_features=self.v_head_dim,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del layer, x, bias
        raise RuntimeError(
            "the compact Exo MLA kv_b W8 layout has no ordinary Linear "
            "orientation; attention dispatch must use absorbed MLA"
        )
