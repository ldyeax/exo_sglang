from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union, cast

import torch

from sglang.kernels.jit.utils import is_hip_runtime
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


CompressMetadata: TypeAlias = Union[CompressorDecodePlan, CompressorPrefillPlan]
# NOTE: alias for backward compatibility
FusedCompressMetadata: TypeAlias = CompressMetadata

_is_hip = is_hip_runtime()


def _use_online_compress(compress_ratio: int) -> bool:
    """Online state-pool path is c128-only."""
    return compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def _extract_positions_from_plan(
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    compress_ratio: int,
) -> torch.Tensor:
    """Extract RoPE positions from plan tensors (decode or prefill).

    DecodePlan layout: [bs, 16] uint8, first 4 bytes = uint32 seq_len.
    CompressPlan layout: [num_c, 16] uint8, first 4 bytes = uint32 seq_len.
    Position for RoPE = seq_len - compress_ratio.
    """
    plan_tensor = plan[1]  # plan_d or plan_c
    seq_lens = plan_tensor[:, :4].contiguous().view(torch.int32).squeeze(-1)
    positions = seq_lens.to(torch.int32) - compress_ratio
    return positions


def _oscar_store_locations_and_mask(
    plan: CompressMetadata,
    out_loc: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map compressor rows to safe locations and mask every invalid row.

    Prefill and target-verify plans are padded with ``CompressPlan::invalid``
    rows whose signed sequence length is ``-1`` and whose ragged id bits are
    not a valid index.  The generic fused store rejects those rows internally;
    OSCAR's separate writer must carry the same validity predicate explicitly.
    """

    plan_raw = plan[1].view(torch.int32)
    sequence_lengths = plan_raw[:, 0]
    if plan.is_decode:
        valid = (sequence_lengths >= compress_ratio) & (
            sequence_lengths % compress_ratio == 0
        )
        return out_loc, valid.contiguous()

    valid = sequence_lengths >= compress_ratio
    ragged_ids = plan_raw[:, 1].to(torch.int32) & 0xFFFF
    safe_ragged_ids = torch.where(valid, ragged_ids, torch.zeros_like(ragged_ids))
    return out_loc[safe_ragged_ids.long()], valid.contiguous()


def _compress_forward_c128_fallback(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    head_dim: int,
) -> torch.Tensor:
    """PyTorch fallback for C128 compress_forward on HIP (wave64).

    Fully vectorized, compatible with CUDA graph capture.
    kv_score_buffer: [num_pages, 128, head_dim * 2]
    ape: [128, head_dim]

    IMPORTANT: This also performs the write to state buffer (like the JIT kernel).
    The JIT kernel does: (1) write kv_score_input to buffer, (2) compress from buffer.
    """
    num_total_slots = kv_score_buffer.shape[0] * kv_score_buffer.shape[1]
    num_pages = kv_score_buffer.shape[0]
    last_dim = kv_score_buffer.shape[-1]

    # Step 1: WRITE kv_score_input to state buffer
    if num_total_slots > 0:
        buf_flat = kv_score_buffer.view(-1, last_dim)
        if plan.is_decode:
            # Decode: plan_d has write_loc per batch item
            plan_raw = plan[1].view(torch.int32)  # [bs, 4]
            write_locs = plan_raw[:, 1].long()
            # Only write valid locations (>= 0 and < buffer size)
            valid_write = (write_locs >= 0) & (write_locs < num_total_slots)
            if valid_write.any():
                buf_flat[write_locs[valid_write]] = kv_score_input[valid_write]
        else:
            # Prefill: plan_w has {ragged_id, write_loc} per write entry
            plan_w = plan[2]  # [num_w, 8] uint8 = WritePlan
            if plan_w.shape[0] > 0:
                plan_w_raw = plan_w.view(torch.int32)  # [num_w, 2]
                ragged_ids = plan_w_raw[:, 0].long() & 0xFFFF
                write_locs = plan_w_raw[:, 1].long()
                valid_write = (write_locs >= 0) & (write_locs < num_total_slots)
                ragged_ids_safe = ragged_ids.clamp(
                    min=0, max=kv_score_input.shape[0] - 1
                )
                if valid_write.any():
                    buf_flat[write_locs[valid_write]] = kv_score_input[
                        ragged_ids_safe[valid_write]
                    ]

    # Step 2: COMPRESS (read from buffer page and do softmax-pool)
    plan_c = plan[1]  # plan_d for decode, plan_c for prefill
    num_tokens = plan_c.shape[0]
    if num_pages == 0 or num_tokens == 0:
        return kv_score_input.new_zeros(num_tokens, head_dim)

    plan_c_raw = plan_c.view(torch.int32)  # [N, 4]
    read_page_0 = plan_c_raw[:, 2].long()
    # Use torch.where instead of clamp to handle -1 (invalid) gracefully
    valid_read = (read_page_0 >= 0) & (read_page_0 < num_pages)
    read_page_0_safe = torch.where(
        valid_read, read_page_0, torch.zeros_like(read_page_0)
    )

    gathered = kv_score_buffer[read_page_0_safe]  # [N, 128, head_dim*2]
    kv = gathered[:, :, :head_dim].float()
    score = gathered[:, :, head_dim:].float() + ape.float().unsqueeze(0)
    weights = score.softmax(dim=1)
    out = (weights * kv).sum(dim=1)

    # For decode: zero out non-boundary tokens (seq_len % 128 != 0)
    # so they don't corrupt kvcache location 0 when stored.
    if plan.is_decode:
        seq_lens = plan_c_raw[:, 0].to(torch.int32)
        is_boundary = (seq_lens % 128 == 0).unsqueeze(-1)  # [N, 1]
        out = torch.where(is_boundary, out, torch.zeros_like(out))

    return out.to(kv_score_input.dtype)


class CompressorBackendMixin:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DSV4Metadata

    def _get_paged_compress_metadata(self, compress_ratio: int) -> CompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        return getattr(self.forward_metadata, attr_name)

    def _get_out_loc(self, compress_ratio: int) -> torch.Tensor:
        attr_name = f"c{compress_ratio}_out_loc"
        return getattr(self.forward_metadata.core_metadata, attr_name)

    def _forward_compress_all_in_one(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        is_indexer: bool,
        rotate: bool,
        compress_ratio: int,
        page_size: int,
        out_loc: torch.Tensor,
        capture_layer_id: int,
        capture_forward_batch: ForwardBatch,
        capture_target_model: bool,
        use_fp4_indexer: bool = False,
        bf16_store: bool = False,
        int4_store: bool = False,
    ) -> None:
        assert compress_ratio == 4 or compress_ratio == 128
        assert rotate == is_indexer == (head_dim == 128)
        if use_fp4_indexer:
            assert is_indexer
            assert compress_ratio == 4
            assert head_dim == 128
        if bf16_store and int4_store:
            raise ValueError("BF16 and INT4 DSV4 cache stores are mutually exclusive")
        if use_fp4_indexer and int4_store:
            raise ValueError(
                "FP4 and signed-INT4 C4 indexer stores are mutually exclusive"
            )

        plan = self._get_paged_compress_metadata(compress_ratio)
        is_online = _use_online_compress(compress_ratio)
        if is_online:
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
        else:
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            last_dim = 2 * head_dim * coff
            assert kv_score_buffer.shape[-1] == last_dim
            kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)

        # Step 1: compress_forward
        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape.view(-1, head_dim),
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            is_online=is_online,
        )

        # The production store fuses norm, RoPE, optional fixed Hadamard, and
        # quantization.  OSCAR needs observations immediately before its own
        # learned rotation (and before the indexer's legacy Hadamard), so make
        # one bounded capture-only clone while retaining the untouched tensor
        # for the baseline fused store.
        from sglang.srt.layers.attention.dsv4.oscar_int2_capture import (
            capture_configured,
            capture_should_materialize,
        )

        if capture_configured():
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
        else:
            parallel = None
        if (
            parallel is not None
            and parallel.attn_tp_rank == 0
            and (
                capture_should_materialize(
                    capture_forward_batch, target_model=capture_target_model
                )
                and kv_compressed.shape[0] > 0
            )
        ):
            capture_value = kv_compressed.clone()
            from sglang.kernels.ops.attention.deepseek_v4_rope import (
                fused_norm_rope_inplace_triton,
            )

            positions = _extract_positions_from_plan(plan, compress_ratio)
            fused_norm_rope_inplace_triton(
                capture_value,
                norm.weight,
                norm.variance_epsilon,
                freqs_cis_cache,
                positions=positions.clamp(min=0),
            )
            from sglang.srt.layers.attention.dsv4.oscar_int2_capture import (
                maybe_capture_compressed_domain,
            )

            maybe_capture_compressed_domain(
                layer_id=capture_layer_id,
                compressed=capture_value,
                is_indexer=is_indexer,
                forward_batch=capture_forward_batch,
                target_model=capture_target_model,
                tp_rank=parallel.attn_tp_rank,
                tp_size=parallel.attn_tp_size,
            )

        # Step 2: norm + rope + store
        compress_norm_rope_store(
            kv_compressed,
            plan,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
            freq_cis=freqs_cis_cache,
            out_loc=out_loc,
            kvcache=kv_cache,
            page_size=page_size,
            use_fp4=use_fp4_indexer,
            bf16_store=bf16_store,
            int4_store=int4_store,
        )

    def _forward_compress_oscar_int2(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        compressor: Compressor,
        layer_id: int,
    ) -> None:
        """Create calibrated history and scorer keys in OSCAR INT2 layouts.

        The ordinary CUDA path fuses norm/RoPE with FP8 or signed-INT4 store.
        OSCAR keeps the compressor output as the sole temporary, applies norm
        and RoPE in place, then dispatches either the shared-latent writer or
        the separately calibrated C4 scorer writer.  The legacy indexer
        Hadamard is deliberately absent: OSCAR's learned C4 rotation replaces
        it.  A device mask preserves decode/speculative non-boundary slots.
        """

        if compressor.is_in_indexer and (
            compressor.ratio != 4 or compressor.head_dim != 128
        ):
            raise ValueError("OSCAR C4 scorer compression requires ratio=4, dim=128")
        plan = self._get_paged_compress_metadata(compressor.ratio)
        is_online = _use_online_compress(compressor.ratio)
        if is_online:
            kv_score_buffer = kv_score_buffer.view(-1, 1, compressor.head_dim * 3)
        else:
            coefficient = 2 if is_overlap_compress(compressor.ratio) else 1
            kv_score_buffer = kv_score_buffer.view(
                -1,
                compressor.ratio,
                2 * compressor.head_dim * coefficient,
            )
        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=compressor.ape.view(-1, compressor.head_dim),
            plan=plan,
            compress_ratio=compressor.ratio,
            head_dim=compressor.head_dim,
            is_online=is_online,
        )
        if kv_compressed.shape[0] == 0:
            return

        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            fused_norm_rope_inplace_triton,
        )

        positions = _extract_positions_from_plan(plan, compressor.ratio)
        fused_norm_rope_inplace_triton(
            kv_compressed,
            compressor.norm.weight,
            compressor.norm.variance_epsilon,
            compressor.freqs_cis,
            positions=positions.clamp(min=0),
        )

        out_loc_to_store, write_mask = _oscar_store_locations_and_mask(
            plan,
            self._get_out_loc(compressor.ratio),
            compressor.ratio,
        )

        if compressor.is_in_indexer:
            token_to_kv_pool.set_index_k_fused(
                layer_id,
                out_loc_to_store,
                kv_compressed.to(torch.bfloat16),
                write_mask=write_mask,
            )
        else:
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id,
                out_loc_to_store,
                kv_compressed.to(torch.bfloat16),
                write_mask=write_mask,
            )

    def forward_unified(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
        kv_score_output: Optional[torch.Tensor] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return

        token_to_kv_pool = self.token_to_kv_pool
        token_to_kv_pool = cast("DeepSeekV4TokenToKVPool", token_to_kv_pool)
        kv_score_input = compressor.compute_kv_score(
            x,
            forward_batch,
            output=kv_score_output,
        )

        state_pool = compressor.get_state_pool(self)
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if _is_hip and not envs.SGLANG_OPT_USE_JIT_NORM.get():
            self._forward_unified_hip(
                token_to_kv_pool=token_to_kv_pool,
                kv_score_input=kv_score_input,
                state_pool=state_pool,
                compressor=compressor,
                layer_id=layer_id,
            )
        else:
            out_loc = self._get_out_loc(compressor.ratio)
            use_fp4_indexer = (
                compressor.is_in_indexer and self.enable_deepseek_v4_fp4_indexer
            )
            bf16_store = False
            int4_store = False
            if compressor.is_in_indexer:
                if token_to_kv_pool.c4_indexer_kv_pool.use_bf16_cache:
                    kv_cache = token_to_kv_pool.get_index_k_bf16_buffer(
                        layer_id
                    ).flatten(1)
                    bf16_store = True
                else:
                    kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
                page_size = token_to_kv_pool.get_index_k_page_size()
                bf16_store = token_to_kv_pool.c4_indexer_kv_pool.use_bf16_cache
                int4_store = token_to_kv_pool.c4_indexer_kv_pool.use_int4_cache
            elif is_unified_kv_triton():
                kv_cache = token_to_kv_pool.get_unified_kv(layer_id)
                page_size = 1
                out_loc = getattr(
                    self.forward_metadata.core_metadata.unified,
                    f"c{compressor.ratio}_out_loc",
                )
                bf16_store = True
                int4_store = False
            else:
                _, _, compress_kv_pool = token_to_kv_pool.layer_mapping[layer_id]
                assert compress_kv_pool is not None
                kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
                bf16_store = compress_kv_pool.use_bf16_cache
                int4_store = compress_kv_pool.use_int4_cache
                if hasattr(compress_kv_pool, "translate_loc_to_hisparse_device"):
                    out_loc = compress_kv_pool._translate_loc_to_hisparse_device(
                        out_loc
                    )
            if token_to_kv_pool.use_oscar_int2_storage:
                self._forward_compress_oscar_int2(
                    token_to_kv_pool=token_to_kv_pool,
                    kv_score_buffer=state_pool.kv_score_buffer.kv_score,
                    kv_score_input=kv_score_input,
                    compressor=compressor,
                    layer_id=layer_id,
                )
            else:
                self._forward_compress_all_in_one(
                    kv_score_buffer=state_pool.kv_score_buffer.kv_score,
                    kv_score_input=kv_score_input,
                    ape=compressor.ape,
                    head_dim=compressor.head_dim,
                    norm=compressor.norm,
                    freqs_cis_cache=compressor.freqs_cis,
                    kv_cache=kv_cache.view(dtype=torch.uint8).view(
                        kv_cache.shape[0], -1
                    ),
                    is_indexer=compressor.is_in_indexer,
                    rotate=compressor.rotate,
                    compress_ratio=compressor.ratio,
                    page_size=page_size,
                    out_loc=out_loc,
                    use_fp4_indexer=use_fp4_indexer,
                    bf16_store=bf16_store,
                    int4_store=int4_store,
                    capture_layer_id=layer_id,
                    capture_forward_batch=forward_batch,
                    capture_target_model=getattr(
                        compressor, "_dsv4_oscar_capture_target", False
                    ),
                )
        online_c128_mtp = getattr(self, "online_c128_mtp", None)
        if online_c128_mtp is not None:
            online_c128_mtp.write_prefix_states(
                layer_id=layer_id,
                compressor=compressor,
                kv_score_input=kv_score_input,
                logical_forward_mode=getattr(
                    forward_batch, "_original_forward_mode", None
                )
                or forward_batch.forward_mode,
            )

    def _forward_unified_hip(
        self,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        kv_score_input: torch.Tensor,
        state_pool,
        compressor: Compressor,
        layer_id: int,
    ) -> None:
        """HIP-specific forward path using PyTorch/Triton fallbacks."""
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            fused_norm_rope_inplace_triton,
        )
        from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
            quant_to_nope_fp8_rope_bf16_pack_triton,
        )
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        compress_ratio = compressor.ratio
        head_dim = compressor.head_dim
        is_indexer = compressor.is_in_indexer

        plan = self._get_paged_compress_metadata(compress_ratio)
        out_loc = self._get_out_loc(compress_ratio)

        # Step 1: compress_forward (always use JIT for both C4 and C128)
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        last_dim = 2 * head_dim * coff
        kv_score_buffer = state_pool.kv_score_buffer.kv_score
        kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=compressor.ape.view(-1, head_dim),
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            is_online=False,
        )

        if kv_compressed.shape[0] == 0:
            return

        # For decode: zero out non-boundary tokens to prevent corrupting kvcache loc 0.
        if plan.is_decode:
            plan_raw = plan[1].view(torch.int32)
            seq_lens_plan = plan_raw[:, 0].to(torch.int32)
            is_boundary = (seq_lens_plan % compress_ratio == 0).unsqueeze(-1)
            kv_compressed = torch.where(
                is_boundary, kv_compressed, torch.zeros_like(kv_compressed)
            )

        # Step 2: norm + rope (Triton fallback for precision parity with V1)
        positions = _extract_positions_from_plan(plan, compress_ratio)
        positions_safe = positions.clamp(min=0)

        fused_norm_rope_inplace_triton(
            kv_compressed,
            compressor.norm.weight,
            compressor.norm.variance_epsilon,
            compressor.freqs_cis,
            positions=positions_safe,
        )

        # Step 3: optional Hadamard rotation for indexer
        if compressor.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # Step 4: store to kvcache
        # For decode: store ALL tokens. Non-boundary tokens have out_loc=0 (safe).
        # For prefill: plan_c already only contains valid entries.
        if plan.is_decode:
            kv_to_store = kv_compressed
            out_loc_to_store = out_loc
        else:
            kv_to_store = kv_compressed
            plan_raw = plan[1].view(torch.int32)
            ragged_ids = plan_raw[:, 1].to(torch.int32) & 0xFFFF
            out_loc_to_store = out_loc[ragged_ids.long()]

        if kv_to_store.shape[0] == 0:
            return

        if (
            token_to_kv_pool.use_ampere_fp8_storage
            or envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get()
        ):
            # fused kernel: BF16 in -> FP8 quant + paged scatter in one launch
            if is_indexer:
                token_to_kv_pool.set_index_k_fused(
                    layer_id=layer_id,
                    loc=out_loc_to_store,
                    cache_k=kv_to_store,
                )
            else:
                token_to_kv_pool.set_extra_key_buffer_fused(
                    layer_id=layer_id,
                    loc=out_loc_to_store,
                    cache_k=kv_to_store,
                )
        else:
            if is_indexer:
                kv_fp8, kv_scale = act_quant(kv_to_store)
                token_to_kv_pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=out_loc_to_store,
                    index_k=kv_fp8,
                    index_k_scale=kv_scale,
                )
            else:
                pack = quant_to_nope_fp8_rope_bf16_pack_triton(kv_to_store.bfloat16())
                token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc_to_store, pack)

    # NOTE: alias for backward compatibility
    forward_indexer_compressor = forward_unified
    forward_core_compressor = forward_unified


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
    online_state_slot_offset: int = 0,
) -> CompressMetadata:
    """Build the paged compress metadata (= the plan).

    State-pool slot translation is done inside the C++ planner; the
    Python side just hands the relevant tensors over.
    """
    if _use_online_compress(compress_ratio):
        return _create_online_paged_compressor_data(
            is_prefill=is_prefill,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens_cpu=extend_lens_cpu,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            num_q_tokens=num_q_tokens,
            online_state_slot_offset=online_state_slot_offset,
        )

    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # NOTE: This is actually a proxy, which encounter some bug with tvm-ffi.
    # As a workaround, we use `.detach()` to get the real tensor.
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
    req_pool_indices_i64 = req_pool_indices.to(torch.int64)

    if is_prefill:
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)

        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_to_token=req_to_token,
            full_to_state=full_to_swa,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            num_q_tokens=num_q_tokens,
            use_cuda_graph=use_prefill_cuda_graph,
        )
    else:
        return CompressorDecodePlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            req_to_token=req_to_token,
            full_to_state=full_to_swa,
            seq_lens=seq_lens.to(torch.int64),
            swa_page_size=swa_page_size,
            ring_size=ring_size,
        )


def _create_online_paged_compressor_data(
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    seq_lens_cpu: Optional[List[int]],
    extend_lens_cpu: Optional[List[int]],
    use_prefill_cuda_graph: bool,
    num_q_tokens: Optional[int],
    online_state_slot_offset: int = 0,
) -> CompressMetadata:
    req_pool_indices = req_pool_indices.to(torch.int64)

    if is_prefill:
        # Sync-on-entry: catch IMA from a prior layer / kernel BEFORE we touch
        # anything in this builder, so blame doesn't land on us spuriously.
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens_planner = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)
            num_q_tokens_planner = num_q_tokens

        return CompressorPrefillPlan.generate_online(
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            num_q_tokens=int(num_q_tokens_planner),
            use_cuda_graph=use_prefill_cuda_graph,
            state_slot_offset=online_state_slot_offset,
        )
    else:
        return CompressorDecodePlan.generate_online(
            seq_lens=seq_lens.to(torch.int64),
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            state_slot_offset=online_state_slot_offset,
        )
