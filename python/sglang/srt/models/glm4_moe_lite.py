# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only GLM-Lite model compatible with HuggingFace weights"""

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.quant_method_registry import is_wrapped_method
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    get_device_sm,
    is_cuda,
    log_info_on_rank0,
    make_layers,
)

_is_cuda = is_cuda()
_device_sm = get_device_sm()

logger = logging.getLogger(__name__)

_GLM47_FLASH_NUM_LAYERS = 47
_GLM47_FLASH_FIRST_MOE_LAYER = 1
_GLM47_FLASH_NUM_ROUTED_EXPERTS = 64
_GLM47_FLASH_KT_WRAPPER_ID = "kt_ep"


def _require_glm47_flash_kt_ep(
    config: PretrainedConfig, pp_size: int, prefix: str
) -> bool:
    """Validate the exact GLM-4.7-Flash KT topology before allocating layers."""
    server_args = get_global_server_args()
    if server_args.kt_weight_path is None:
        return False

    from sglang.srt.layers.moe.kt_ep_wrapper import require_kt_ep_registration

    require_kt_ep_registration()

    expected_config = {
        "num_hidden_layers": _GLM47_FLASH_NUM_LAYERS,
        "first_k_dense_replace": _GLM47_FLASH_FIRST_MOE_LAYER,
        "n_routed_experts": _GLM47_FLASH_NUM_ROUTED_EXPERTS,
        "moe_layer_freq": 1,
    }
    mismatches = {
        name: {"expected": expected, "actual": getattr(config, name, None)}
        for name, expected in expected_config.items()
        if getattr(config, name, None) != expected
    }
    if mismatches:
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash integration received an unsupported "
            f"model topology: {mismatches}"
        )
    hash_layer_counts = {}
    for name in ("n_hash_layers", "num_hash_layers"):
        value = getattr(config, name, None)
        if value not in (None, 0):
            hash_layer_counts[name] = value
    if hash_layer_counts:
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash does not support hash-routed MoE "
            f"layers: {hash_layer_counts}"
        )
    if not 1 <= pp_size <= _GLM47_FLASH_NUM_LAYERS:
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash requires a pipeline parallel size "
            f"between 1 and {_GLM47_FLASH_NUM_LAYERS}, got {pp_size}"
        )
    if prefix != "model":
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash requires canonical model prefix "
            f"'model', got {prefix!r}"
        )
    return True


def _build_glm47_flash_kt_ep_coverage_receipt(
    layers: nn.ModuleList,
    config: PretrainedConfig,
    prefix: str,
    pp_rank: int,
    pp_size: int,
    start_layer: int,
    end_layer: int,
) -> Dict[str, Any]:
    """Verify every local routed layer and return a machine-readable receipt."""
    from sglang.srt.layers.moe.kt_ep_wrapper import get_kt_ep_gpu_experts_masks

    server_args = get_global_server_args()
    masks = get_kt_ep_gpu_experts_masks()
    expected_mask_shape = (
        _GLM47_FLASH_NUM_LAYERS,
        _GLM47_FLASH_NUM_ROUTED_EXPERTS,
    )
    if tuple(masks.shape) != expected_mask_shape:
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash expert placement mask shape mismatch: "
            f"expected {expected_mask_shape}, got {tuple(masks.shape)}"
        )

    if (
        pp_size < 1
        or not 0 <= pp_rank < pp_size
        or not 0 <= start_layer < end_layer <= _GLM47_FLASH_NUM_LAYERS
    ):
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash received an invalid local pipeline "
            f"range: rank={pp_rank}, size={pp_size}, "
            f"layers=[{start_layer}, {end_layer})"
        )

    expected_layer_ids = list(
        range(max(start_layer, _GLM47_FLASH_FIRST_MOE_LAYER), end_layer)
    )
    actual_layer_ids = [
        layer_id
        for layer_id, layer in enumerate(layers)
        if isinstance(getattr(layer, "mlp", None), Glm4MoeLiteSparseMoeBlock)
    ]
    if actual_layer_ids != expected_layer_ids:
        raise RuntimeError(
            "KTransformers GLM-4.7-Flash routed-layer coverage mismatch: "
            f"expected {expected_layer_ids}, got {actual_layer_ids}"
        )

    configured_resident_count = server_args.kt_num_gpu_experts
    layer_receipts = []
    for layer_id in expected_layer_ids:
        decoder_layer = layers[layer_id]
        sparse_block = decoder_layer.mlp
        experts = sparse_block.experts
        module_path = f"{prefix}.layers.{layer_id}.mlp.experts"

        if getattr(sparse_block, "is_hash", None) is not False:
            raise RuntimeError(
                f"{module_path} must disable DeepSeek hash routing, got "
                f"{getattr(sparse_block, 'is_hash', None)!r}"
            )
        if type(experts) is not FusedMoE:
            raise RuntimeError(
                f"{module_path} must be the local FusedMoE implementation, got "
                f"{type(experts).__module__}.{type(experts).__name__}"
            )
        if getattr(experts, "_registry_prefix", None) != module_path:
            raise RuntimeError(
                f"{module_path} registry path mismatch: "
                f"{getattr(experts, '_registry_prefix', None)!r}"
            )

        quant_method = experts.quant_method
        if not is_wrapped_method(quant_method, _GLM47_FLASH_KT_WRAPPER_ID):
            raise RuntimeError(
                f"{module_path} is not wrapped by {_GLM47_FLASH_KT_WRAPPER_ID!r}"
            )
        kt_config = getattr(quant_method, "kt_config", None)
        if kt_config is None:
            raise RuntimeError(f"{module_path} KT wrapper has no KTConfig")

        linkage = {
            "decoder_layer_id": getattr(decoder_layer, "layer_id", None),
            "sparse_block_layer_id": getattr(sparse_block, "layer_id", None),
            "fused_moe_layer_id": getattr(experts, "layer_id", None),
            "runner_layer_id": getattr(experts.moe_runner_config, "layer_id", None),
            "kt_config_layer_idx": getattr(kt_config, "layer_idx", None),
        }
        invalid_linkage = {
            name: value for name, value in linkage.items() if value != layer_id
        }
        if invalid_linkage:
            raise RuntimeError(
                f"{module_path} layer linkage mismatch: {invalid_linkage}"
            )
        if getattr(sparse_block, "config", None) is not config:
            raise RuntimeError(f"{module_path} is not linked to the model config")
        if getattr(kt_config, "num_layers", None) != _GLM47_FLASH_NUM_LAYERS:
            raise RuntimeError(
                f"{module_path} KTConfig num_layers mismatch: "
                f"{getattr(kt_config, 'num_layers', None)!r}"
            )

        layer_mask = getattr(kt_config, "gpu_experts_mask", None)
        if layer_mask is None or tuple(layer_mask.shape) != (
            _GLM47_FLASH_NUM_ROUTED_EXPERTS,
        ):
            raise RuntimeError(
                f"{module_path} KTConfig mask must have shape "
                f"({_GLM47_FLASH_NUM_ROUTED_EXPERTS},)"
            )
        if not torch.equal(layer_mask.cpu(), masks[layer_id].cpu()):
            raise RuntimeError(
                f"{module_path} KTConfig mask is not linked to placement row {layer_id}"
            )
        resident_count = int(layer_mask.sum().item())
        if resident_count != configured_resident_count:
            raise RuntimeError(
                f"{module_path} GPU-resident expert count mismatch: expected "
                f"{configured_resident_count}, got {resident_count}"
            )

        layer_receipts.append(
            {
                "layer_id": layer_id,
                "module_path": module_path,
                "wrapper_id": _GLM47_FLASH_KT_WRAPPER_ID,
                "gpu_resident_experts": resident_count,
            }
        )

    mask_bytes = bytes(int(value) for value in masks.to(dtype=torch.uint8).view(-1))
    return {
        "schema_version": 1,
        "model_architecture": "Glm4MoeLiteForCausalLM",
        "pipeline_parallel_rank": pp_rank,
        "pipeline_parallel_size": pp_size,
        "pipeline_layer_start": start_layer,
        "pipeline_layer_end": end_layer,
        "layer_count": _GLM47_FLASH_NUM_LAYERS,
        "routed_layer_ids": expected_layer_ids,
        "routed_expert_count": _GLM47_FLASH_NUM_ROUTED_EXPERTS,
        "gpu_expert_mask_shape": list(expected_mask_shape),
        "gpu_expert_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "configured_gpu_resident_experts_per_layer": configured_resident_count,
        "ktransformers_method": server_args.kt_method,
        "wrapper_id": _GLM47_FLASH_KT_WRAPPER_ID,
        "layers": layer_receipts,
    }


class Glm4MoeLiteMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ):
        # Keep parity with DeepseekV2MLP.forward signature since DeepseekV2DecoderLayer
        # invokes MLP modules with these extra arguments.
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        # Some quantization wrappers store the underlying parameter as `weight_packed`.
        if not hasattr(self.gate_up_proj, "weight"):
            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
        if not hasattr(self.down_proj, "weight"):
            self.down_proj.weight = getattr(self.down_proj, "weight_packed")

        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and self.gate_up_proj.weight.dtype == torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


class Glm4MoeLiteGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )

    def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None):
        # NOTE: For some unknown reason, router_gemm seems degrade accept length.
        if (
            _is_cuda
            and not self.is_nextn
            and hidden_states.shape[0] < 4
            and hidden_states.shape[1] == 7168
            and self.weight.shape[0] == 256
            and _device_sm >= 90
        ):
            from sgl_kernel import dsv3_router_gemm

            logits = dsv3_router_gemm(hidden_states, self.weight).to(
                hidden_states.dtype
            )
        else:
            logits = F.linear(hidden_states, self.weight, None)

        return logits


class Glm4MoeLiteSparseMoeBlock(DeepseekV2MoE):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn
        # This class inherits DeepseekV2MoE's forward methods but constructs
        # GLM's score-based TopK, which never accepts hash-routing inputs.
        self.is_hash = False

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = Glm4MoeLiteGate(
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn
        )

        if get_global_server_args().kt_weight_path is not None:
            from sglang.srt.layers.moe.kt_ep_wrapper import (
                require_kt_ep_registration,
            )

            require_kt_ep_registration()

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard. We use quant_config to determine the output format.
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe, or with fp4 allgather
            self.shared_experts = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            )
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )

        self.top_k = config.num_experts_per_tok

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()


class Glm4MoeLiteDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config

        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        rope_theta = 1000000
        rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 202752)
        self.layer_id = layer_id

        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            reduce_results=False,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Glm4MoeLiteSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(
                is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
            ),
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
        )


class Glm4MoeLiteModel(DeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()
        kt_ep_enabled = _require_glm47_flash_kt_ep(
            config=config,
            pp_size=self.pp_group.world_size,
            prefix=prefix,
        )

        # DeepseekV2Model.forward expects these attributes to exist.
        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.cp_size = get_attention_tp_size() if self.nsa_enable_prefill_cp else None
        self.gemm_output_zero_allocator_size = 0
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm4MoeLiteDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []
        self.kt_ep_coverage_receipt = None
        if kt_ep_enabled:
            self.kt_ep_coverage_receipt = _build_glm47_flash_kt_ep_coverage_receipt(
                layers=self.layers,
                config=config,
                prefix=prefix,
                pp_rank=self.pp_group.rank_in_group,
                pp_size=self.pp_group.world_size,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
            logger.info(
                "GLM47_FLASH_KT_EP_COVERAGE_RECEIPT %s",
                json.dumps(self.kt_ep_coverage_receipt, sort_keys=True),
            )


class Glm4MoeLiteForCausalLM(DeepseekV2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config.moe_layer_freq = 1
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.determine_num_fused_shared_experts("Glm4MoeLiteForCausalLM")
        self.model = Glm4MoeLiteModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.kt_ep_coverage_receipt = self.model.kt_ep_coverage_receipt
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)
            }
        )
        self.capture_aux_hidden_states = False

        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            from sglang.srt.layers.dp_attention import (
                get_attention_tp_rank,
                get_attention_tp_size,
            )

            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_rank = self.cp_size = None

    def determine_num_fused_shared_experts(
        self, architecture: str = "Glm4MoeLiteForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if (
            not _is_cuda
            or torch.cuda.get_device_capability("cuda") < (8, 0)
            or self.config.architectures[0] != architecture
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Only GLM-4.5 or GLM-4.6 on NV-platform with capability >= 80 can use shared experts fusion optimization."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = "GLM-4.5 or GLM-4.6 cannot use shared experts fusion optimization under expert parallelism."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn=False,
        params_dict=None,
        is_eagle=False,
    ):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1

            def iter_weights_with_fused_shared_experts(
                weights: Iterable[Tuple[str, torch.Tensor]],
            ) -> Iterable[Tuple[str, torch.Tensor]]:
                import re

                pattern = re.compile(
                    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(.+)$"
                )
                for name, weight in weights:
                    match = pattern.match(name)
                    if match:
                        layer_id = int(match.group(1))
                        suffix = match.group(2)
                        name = f"model.layers.{layer_id}.mlp.experts.{self.config.n_routed_experts}.{suffix}"
                    yield name, weight

            weights = iter_weights_with_fused_shared_experts(weights)

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]
        else:
            nextn_layer_prefix = None
            nextn_spec_weight_names = []

        eagle_ignore_weight_names = []
        if is_eagle:
            eagle_ignore_weight_names = [
                "eagle_draft_tokens_map",
                "eagle_lm_head.weight",
            ]

        if params_dict is None:
            params_dict = dict(self.named_parameters())

        weight_names = []
        for name, loaded_weight in weights:
            weight_names.append(name)

            if not is_nextn and not is_eagle:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if nextn_layer_prefix and not name.startswith(nextn_layer_prefix):
                    continue

                if nextn_layer_prefix is not None:  # mtp
                    # Use shared head and embed weights from target model
                    if "shared_head.head" in name or "embed_tokens" in name:
                        continue

                    is_decoder = True
                    # For nextn specific weights
                    for weight_name in nextn_spec_weight_names:
                        if weight_name in name:
                            name = name.replace(nextn_layer_prefix, "model")
                            is_decoder = False
                            break
                    # For decoder layer weights
                    if is_decoder:
                        name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Mark as expert weight regardless of whether we can process it
                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in eagle_ignore_weight_names:
                        continue

                    # GLM NOTE: for MLA
                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=0
                            )
                            param_name = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                                )
                            )
                            if param_name not in params_dict:
                                continue
                            param = params_dict[param_name]

                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        if (
                            "k_scale" in name or "v_scale" in name
                        ) and name not in params_dict:
                            # modelopt attn kv scale is named differently
                            if any(scale in name for scale in ["k_scale", "v_scale"]):
                                name = name.replace("_proj", "attn_mqa")
                            else:
                                logger.warning(
                                    f"Unknown scale found in checkpoint: {name}"
                                )

                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # DeepseekV2AttentionMLA.forward_* expects post_load_weights() to populate
        # per-layer packed weights like `w_kc`/`w_vc` (used during CUDA graph capture).
        # GLM-Lite configs may not set `config.mla`, but this model always uses
        # DeepseekV2AttentionMLA, so we must run the post-load processing.
        # Use weight_names=None to ensure we always process all layers. Some checkpoints /
        # naming schemes may not include "kv_b_proj" in `weight_names`, but `w_kc`/`w_vc`
        # are still required by DeepseekV2AttentionMLA at runtime.
        self.post_load_weights(is_nextn=is_nextn, weight_names=None)


EntryClass = [Glm4MoeLiteForCausalLM]
