"""SM86 sparse MLA decode with byte-FP8 SWA and BF16 C128 pages.

This is intentionally a narrow DeepSeek-V4 storage specialization.  The SWA
ring retains the 584-byte E4M3FN-as-storage layout, while C128 compressed keys
use 512 contiguous BF16 elements.  Keeping the two loops separate avoids
executing the software FP8 decoder for the overwhelmingly larger C128 region
and preserves one online-softmax state across C128, SWA, and the attention
sink.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    get_e4m3fn_decode_lut,
)
from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
    DEEPSEEK_V4_MLA_HEAD_DIM,
    FP8_DS_MLA_FP8_DIM,
    FP8_DS_MLA_SCALE_BYTES,
    FP8_DS_MLA_SCALE_GROUP,
    FP8_DS_MLA_TOKEN_BYTES,
    _decode_launch_config,
    _prepare_packed_page_buffer,
    _resolve_packed_page_size,
)

LOG2E = math.log2(math.e)
BF16_TOKEN_ELEMS = DEEPSEEK_V4_MLA_HEAD_DIM
BF16_TOKEN_BYTES = BF16_TOKEN_ELEMS * 2


@triton.jit
def _update_online_softmax(
    q,
    k,
    valid,
    head_mask,
    running_max,
    running_sum,
    accumulator,
    sm_scale_log2: tl.constexpr,
):
    logits = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
    logits = tl.where(
        head_mask[:, None] & valid[None, :],
        logits,
        -3.4028234663852886e38,
    )
    next_max = tl.maximum(tl.max(logits, 1), running_max)
    rescale = tl.exp2(running_max - next_max)
    probability = tl.exp2(logits - next_max[:, None])
    probability = tl.where(
        head_mask[:, None] & valid[None, :], probability, 0.0
    )
    accumulator = accumulator * rescale[:, None] + tl.dot(
        probability.to(tl.bfloat16), k
    )
    running_sum = running_sum * rescale + tl.sum(probability, 1)
    return next_max, running_sum, accumulator


@triton.jit
def _decode_sparse_attention_fp8_swa_bf16_extra_kernel(
    q_ptr,
    swa_cache_u8_ptr,
    swa_cache_bf16_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    lut_ptr,
    sink_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    swa_index_topk: tl.constexpr,
    extra_index_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_stride_block_bytes: tl.constexpr,
    extra_stride_block_elems: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_swa_idx_t: tl.constexpr,
    stride_swa_idx_k: tl.constexpr,
    stride_extra_idx_t: tl.constexpr,
    stride_extra_idx_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_group: tl.constexpr,
    scale_bytes: tl.constexpr,
    fp8_token_bytes: tl.constexpr,
    bf16_token_elems: tl.constexpr,
    has_sink: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * block_h + tl.arange(0, block_h)
    dims = tl.arange(0, block_d)
    head_mask = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + dims[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )
    if has_sink:
        sink = tl.load(sink_ptr + heads, mask=head_mask, other=-float("inf"))
        running_max = sink * 1.4426950408889634
        running_sum = tl.where(head_mask, 1.0, 0.0)
    else:
        running_max = tl.full((block_h,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_h,), tl.float32)
    accumulator = tl.zeros((block_h, block_d), tl.float32)

    # C128 comes first in the production ordering.  It is normally 32x larger
    # than the SWA tail, so keep this hot loop free of FP8 LUT/scale work.
    extra_len = tl.load(extra_lens_ptr + token_id)
    extra_loop_end = tl.cdiv(extra_len, block_n) * block_n
    for start in range(0, extra_loop_end, block_n):
        columns = start + tl.arange(0, block_n)
        selected = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_idx_t
            + columns * stride_extra_idx_k,
            mask=columns < extra_index_topk,
            other=-1,
        )
        valid = (columns < extra_len) & (selected >= 0)
        safe_selected = tl.where(valid, selected, 0)
        block = safe_selected // extra_block_size
        position = safe_selected - block * extra_block_size
        valid &= block < extra_num_blocks
        token_base = (
            block * extra_stride_block_elems + position * bf16_token_elems
        )
        k = tl.load(
            extra_cache_bf16_ptr + token_base[:, None] + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        running_max, running_sum, accumulator = _update_online_softmax(
            q,
            k,
            valid,
            head_mask,
            running_max,
            running_sum,
            accumulator,
            sm_scale_log2,
        )

    # The SWA ring remains byte-packed E4M3FN.  Decode only this short tail,
    # hoisting its seven UE8M0 scales once per key.
    swa_len = tl.load(swa_lens_ptr + token_id)
    swa_loop_end = tl.cdiv(swa_len, block_n) * block_n
    is_nope = dims < fp8_dim
    scale_group_id = dims // scale_group
    for start in range(0, swa_loop_end, block_n):
        columns = start + tl.arange(0, block_n)
        selected = tl.load(
            swa_indices_ptr
            + token_id * stride_swa_idx_t
            + columns * stride_swa_idx_k,
            mask=columns < swa_index_topk,
            other=-1,
        )
        valid = (columns < swa_len) & (selected >= 0)
        safe_selected = tl.where(valid, selected, 0)
        block = safe_selected // swa_block_size
        position = safe_selected - block * swa_block_size
        valid &= block < swa_num_blocks

        token_base = block * swa_stride_block_bytes + position * fp8_token_bytes
        scale_base = (
            block * swa_stride_block_bytes
            + swa_block_size * fp8_token_bytes
            + position * scale_bytes
        )
        scale_0 = tl.load(swa_cache_u8_ptr + scale_base, mask=valid, other=127).to(
            tl.float32
        )
        scale_1 = tl.load(
            swa_cache_u8_ptr + scale_base + 1, mask=valid, other=127
        ).to(tl.float32)
        scale_2 = tl.load(
            swa_cache_u8_ptr + scale_base + 2, mask=valid, other=127
        ).to(tl.float32)
        scale_3 = tl.load(
            swa_cache_u8_ptr + scale_base + 3, mask=valid, other=127
        ).to(tl.float32)
        scale_4 = tl.load(
            swa_cache_u8_ptr + scale_base + 4, mask=valid, other=127
        ).to(tl.float32)
        scale_5 = tl.load(
            swa_cache_u8_ptr + scale_base + 5, mask=valid, other=127
        ).to(tl.float32)
        scale_6 = tl.load(
            swa_cache_u8_ptr + scale_base + 6, mask=valid, other=127
        ).to(tl.float32)
        encoded_scale = tl.where(scale_group_id[None, :] == 0, scale_0[:, None], 127.0)
        encoded_scale = tl.where(
            scale_group_id[None, :] == 1, scale_1[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group_id[None, :] == 2, scale_2[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group_id[None, :] == 3, scale_3[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group_id[None, :] == 4, scale_4[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group_id[None, :] == 5, scale_5[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group_id[None, :] == 6, scale_6[:, None], encoded_scale
        )
        decoded_scale = tl.exp2(encoded_scale - 127.0)

        raw = tl.load(
            swa_cache_u8_ptr + token_base[:, None] + dims[None, :],
            mask=valid[:, None] & is_nope[None, :],
            other=0,
        )
        lut_index = raw.to(tl.uint32).to(tl.int32)
        nope = tl.load(lut_ptr + lut_index).to(tl.float32) * decoded_scale

        rope_offset = (token_base[:, None] + fp8_dim) // 2
        rope_offset += dims[None, :] - fp8_dim
        rope = tl.load(
            swa_cache_bf16_ptr + rope_offset,
            mask=valid[:, None] & (~is_nope[None, :]),
            other=0.0,
        ).to(tl.float32)
        k = tl.where(is_nope[None, :], nope, rope).to(tl.bfloat16)
        running_max, running_sum, accumulator = _update_online_softmax(
            q,
            k,
            valid,
            head_mask,
            running_max,
            running_sum,
            accumulator,
            sm_scale_log2,
        )

    accumulator /= tl.maximum(running_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + dims[None, :] * stride_out_d,
        accumulator.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


def _prepare_bf16_page_buffer(
    cache: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, int]:
    if cache.ndim < 2:
        raise ValueError(f"extra_cache must have a page dimension, got {cache.shape}")
    cache_u8 = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    if cache_u8.stride(-1) != 1:
        raise ValueError("extra_cache's innermost byte dimension must be contiguous")
    page_stride_bytes = cache_u8.stride(0)
    required_page_bytes = page_size * BF16_TOKEN_BYTES
    if page_stride_bytes < required_page_bytes:
        raise ValueError(
            f"extra_cache page stride is {page_stride_bytes} bytes, but page size "
            f"{page_size} requires {required_page_bytes} bytes"
        )
    if page_stride_bytes % 2 or cache_u8.storage_offset() % 2:
        raise ValueError("extra_cache must be BF16-aligned")
    return cache_u8.view(torch.bfloat16), page_stride_bytes // 2


def decode_sparse_attention_fp8_swa_bf16_extra(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lens: torch.Tensor,
    *,
    swa_block_size: int | None = None,
    extra_block_size: int | None = None,
) -> None:
    """Decode into caller-owned ``out`` using the selective C128 layout.

    The architecture check is deliberately fail-closed.  Native-FP8 devices
    and other Ampere variants must continue through their established paths.
    """

    if q.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("selective C128 BF16 decode requires NVIDIA CUDA")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (8, 6):
        raise RuntimeError(
            "selective C128 BF16 decode is implemented only for exact SM86, "
            f"got SM{capability[0]}{capability[1]}"
        )
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)
    if q.ndim != 3 or q.shape[-1] != DEEPSEEK_V4_MLA_HEAD_DIM:
        raise ValueError(
            "q must have shape [tokens, heads, "
            f"{DEEPSEEK_V4_MLA_HEAD_DIM}]"
        )
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise TypeError("q must be contiguous BF16")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 with the same shape as q")
    if swa_indices.dtype != torch.int32 or extra_indices.dtype != torch.int32:
        raise TypeError("sparse indices must use int32")
    if swa_lens.dtype != torch.int32 or extra_lens.dtype != torch.int32:
        raise TypeError("sparse lengths must use int32")
    if q.shape[0] == 0:
        return

    resolved_swa_block_size = _resolve_packed_page_size(
        swa_cache, swa_block_size, name="swa_cache"
    )
    resolved_extra_block_size = _resolve_packed_page_size(
        extra_cache, extra_block_size, name="extra_cache"
    )
    swa_u8, swa_bf16, swa_stride_bytes = _prepare_packed_page_buffer(
        swa_cache, resolved_swa_block_size, name="swa_cache"
    )
    extra_bf16, extra_stride_elems = _prepare_bf16_page_buffer(
        extra_cache, resolved_extra_block_size
    )

    tensors = (
        swa_u8,
        swa_indices,
        swa_lens,
        extra_bf16,
        extra_indices,
        extra_lens,
        out,
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all selective C128 tensors must share one CUDA device")
    if attn_sink is not None and attn_sink.device != q.device:
        raise ValueError("attn_sink must share q's CUDA device")

    num_tokens, num_heads, _ = q.shape
    block_h, block_n, num_warps = _decode_launch_config(q.device)
    lut = get_e4m3fn_decode_lut(q.device)
    _decode_sparse_attention_fp8_swa_bf16_extra_kernel[
        (num_tokens, triton.cdiv(num_heads, block_h))
    ](
        q,
        swa_u8,
        swa_bf16,
        swa_indices,
        swa_lens,
        extra_bf16,
        extra_indices,
        extra_lens,
        lut,
        attn_sink if attn_sink is not None else q,
        out,
        num_heads=num_heads,
        swa_index_topk=swa_indices.shape[-1],
        extra_index_topk=extra_indices.shape[-1],
        swa_num_blocks=swa_u8.shape[0],
        extra_num_blocks=extra_bf16.shape[0],
        swa_block_size=resolved_swa_block_size,
        extra_block_size=resolved_extra_block_size,
        swa_stride_block_bytes=swa_stride_bytes,
        extra_stride_block_elems=extra_stride_elems,
        sm_scale_log2=scale * LOG2E,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_swa_idx_t=swa_indices.stride(0),
        stride_swa_idx_k=swa_indices.stride(1),
        stride_extra_idx_t=extra_indices.stride(0),
        stride_extra_idx_k=extra_indices.stride(1),
        stride_out_t=out.stride(0),
        stride_out_h=out.stride(1),
        stride_out_d=out.stride(2),
        block_h=block_h,
        block_n=block_n,
        block_d=DEEPSEEK_V4_MLA_HEAD_DIM,
        fp8_dim=FP8_DS_MLA_FP8_DIM,
        scale_group=FP8_DS_MLA_SCALE_GROUP,
        scale_bytes=FP8_DS_MLA_SCALE_BYTES,
        fp8_token_bytes=FP8_DS_MLA_TOKEN_BYTES,
        bf16_token_elems=BF16_TOKEN_ELEMS,
        has_sink=attn_sink is not None,
        num_stages=1,
        num_warps=num_warps,
    )


__all__ = ["decode_sparse_attention_fp8_swa_bf16_extra"]
