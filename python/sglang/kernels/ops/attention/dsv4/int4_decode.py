"""Fused signed-INT4 sparse MLA decode for DeepSeek V4 on SM86.

The cache is architecture-neutral uint8 storage.  Every token occupies 368
bytes: 224 packed signed-nibble bytes, 64 BF16 RoPE values, seven BF16 scales,
and two padding bytes.  This kernel unpacks and scales keys in registers, feeds
BF16 operands to Ampere HMMA, and never materializes a BF16 cache workspace.
"""

from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.int4_storage import (
    GROUP_SIZE,
    HEAD_DIM,
    NOPE_DIM,
    NUM_GROUPS,
    ROPE_OFFSET_BYTES,
    SCALE_OFFSET_BYTES,
    STORAGE_BYTES_PER_TOKEN,
)

LOG2E = math.log2(math.e)


@functools.cache
def _decode_launch_config(device: torch.device) -> tuple[int, int, int]:
    """Return the SM86-tuned (head tile, key tile, warps) geometry.

    Keep the device query out of CUDA-graph replay.  The cached helper also
    gives isolated tuning tools one narrow function to override without
    changing the production call contract.
    """

    if device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 6):
        return 8, 16, 4
    return 8, 16, 8


@triton.jit
def _decode_sparse_attention_int4_kernel(
    q_ptr,
    swa_cache_u8_ptr,
    swa_cache_bf16_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_u8_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
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
    extra_stride_block_bytes: tl.constexpr,
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
    nope_d: tl.constexpr,
    group_d: tl.constexpr,
    num_scale_groups: tl.constexpr,
    token_bytes: tl.constexpr,
    rope_offset: tl.constexpr,
    scale_offset: tl.constexpr,
    has_extra: tl.constexpr,
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

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id) if has_extra else 0
    total_len = swa_len + extra_len
    loop_end = tl.cdiv(total_len, block_n) * block_n
    for start in range(0, loop_end, block_n):
        offsets = start + tl.arange(0, block_n)
        use_extra = has_extra & (offsets < extra_len)
        use_swa = (offsets >= extra_len) & (offsets < total_len)
        extra_cols = offsets
        swa_cols = offsets - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_idx_t
            + extra_cols * stride_extra_idx_k,
            mask=has_extra & (extra_cols < extra_index_topk),
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr + token_id * stride_swa_idx_t + swa_cols * stride_swa_idx_k,
            mask=(swa_cols >= 0) & (swa_cols < swa_index_topk),
            other=-1,
        )
        selected_idx = tl.where(use_extra, extra_idx, swa_idx)
        selected_block_size = tl.where(use_extra, extra_block_size, swa_block_size)
        selected_num_blocks = tl.where(use_extra, extra_num_blocks, swa_num_blocks)
        valid = (use_extra | use_swa) & (selected_idx >= 0)
        safe_idx = tl.where(valid, selected_idx, 0)
        block = safe_idx // selected_block_size
        position = safe_idx - block * selected_block_size
        valid &= block < selected_num_blocks
        selected_stride = tl.where(
            use_extra, extra_stride_block_bytes, swa_stride_block_bytes
        )
        token_base = block * selected_stride + position * token_bytes

        cache_u8_ptr = tl.where(
            use_extra[:, None], extra_cache_u8_ptr, swa_cache_u8_ptr
        )
        cache_bf16_ptr = tl.where(
            use_extra[:, None], extra_cache_bf16_ptr, swa_cache_bf16_ptr
        )
        is_nope = dims < nope_d
        # Each byte owns two adjacent signed nibbles.  Keep the explicit
        # dimension mapping here: unlike the C4 scorer tile, Triton's
        # ``interleave`` lowering for this 16x256 value does not preserve the
        # [key, dimension] order required by the subsequent 512-D HMMA.
        packed_offsets = token_base[:, None] + dims[None, :] // 2
        packed = tl.load(
            cache_u8_ptr + packed_offsets,
            mask=valid[:, None] & is_nope[None, :],
            other=0,
        ).to(tl.int32)
        nibble = tl.where(
            (dims[None, :] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        )
        signed_code = tl.where(nibble < 8, nibble, nibble - 16).to(tl.float32)

        # Hoist the seven BF16 scales out of the per-dimension load.  The old
        # address matrix repeated the same load for every one of 64 dimensions
        # in a group (448 scale-load lanes/key); this issues seven loads/key and
        # broadcasts them in registers, matching the optimized SM86 FP8 path.
        scale_base = (token_base + scale_offset) // 2
        scale_ptr = tl.where(use_extra, extra_cache_bf16_ptr, swa_cache_bf16_ptr)
        scale_0 = tl.load(scale_ptr + scale_base, mask=valid, other=0.0).to(tl.float32)
        scale_1 = tl.load(scale_ptr + scale_base + 1, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_2 = tl.load(scale_ptr + scale_base + 2, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_3 = tl.load(scale_ptr + scale_base + 3, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_4 = tl.load(scale_ptr + scale_base + 4, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_5 = tl.load(scale_ptr + scale_base + 5, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_6 = tl.load(scale_ptr + scale_base + 6, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_group = dims // group_d
        scales = tl.where(scale_group[None, :] == 0, scale_0[:, None], 0.0)
        scales = tl.where(scale_group[None, :] == 1, scale_1[:, None], scales)
        scales = tl.where(scale_group[None, :] == 2, scale_2[:, None], scales)
        scales = tl.where(scale_group[None, :] == 3, scale_3[:, None], scales)
        scales = tl.where(scale_group[None, :] == 4, scale_4[:, None], scales)
        scales = tl.where(scale_group[None, :] == 5, scale_5[:, None], scales)
        scales = tl.where(scale_group[None, :] == 6, scale_6[:, None], scales)
        scales = tl.where(
            valid[:, None] & is_nope[None, :],
            scales,
            0.0,
        )
        no_pe = signed_code * scales

        rope_offsets = (token_base[:, None] + rope_offset) // 2
        rope_offsets += dims[None, :] - nope_d
        rope = tl.load(
            cache_bf16_ptr + rope_offsets,
            mask=valid[:, None] & (~is_nope[None, :]),
            other=0.0,
        ).to(tl.float32)
        k = tl.where(is_nope[None, :], no_pe, rope).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
        qk = tl.where(
            head_mask[:, None] & valid[None, :],
            qk,
            -3.4028234663852886e38,
        )
        next_max = tl.maximum(tl.max(qk, 1), running_max)
        rescale = tl.exp2(running_max - next_max)
        probability = tl.exp2(qk - next_max[:, None])
        probability = tl.where(head_mask[:, None] & valid[None, :], probability, 0.0)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(tl.bfloat16), k
        )
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        running_max = next_max

    accumulator /= tl.maximum(running_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + dims[None, :] * stride_out_d,
        accumulator.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


def _canonical_cache(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    cache_u8 = cache.view(torch.uint8)
    if cache_u8.ndim < 2:
        raise ValueError("INT4 cache must contain a page and byte dimension")
    cache_u8 = cache_u8.reshape(cache_u8.shape[0], -1)
    expected = page_size * STORAGE_BYTES_PER_TOKEN
    if cache_u8.shape[1] != expected or cache_u8.stride(1) != 1:
        raise ValueError(
            f"INT4 cache page must contain exactly {expected} bytes, got "
            f"{tuple(cache_u8.shape)}"
        )
    if cache_u8.stride(0) % 2 or cache_u8.storage_offset() % 2:
        raise ValueError("INT4 cache pages must be BF16 aligned")
    return cache_u8


def decode_sparse_attention_int4(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    swa_block_size: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_block_size: int | None = None,
) -> None:
    """Run allocation-free INT4 sparse decode into caller-owned ``out``."""

    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if q.dtype != torch.bfloat16 or head_dim != HEAD_DIM or not q.is_contiguous():
        raise ValueError(f"q must be contiguous BF16 [T, H, {HEAD_DIM}]")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 with the same shape as q")
    if swa_lens.dtype != torch.int32 or swa_indices.dtype != torch.int32:
        raise ValueError("INT4 sparse metadata must use int32")

    swa_u8 = _canonical_cache(swa_cache, swa_block_size)
    has_extra = bool(
        extra_cache is not None
        and extra_indices is not None
        and extra_lens is not None
        and extra_block_size is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_lens is not None
        assert extra_block_size is not None
        extra_u8 = _canonical_cache(extra_cache, extra_block_size)
    else:
        extra_u8 = swa_u8
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens
        extra_block_size = swa_block_size
    assert extra_indices is not None
    assert extra_lens is not None
    assert extra_block_size is not None

    tensors = (swa_u8, swa_indices, swa_lens, extra_u8, extra_indices, extra_lens, out)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all INT4 sparse-attention tensors must share one device")
    swa_bf16 = swa_u8.view(torch.bfloat16)
    extra_bf16 = extra_u8.view(torch.bfloat16)
    block_h, block_n, num_warps = _decode_launch_config(q.device)
    block_d = HEAD_DIM
    _decode_sparse_attention_int4_kernel[(num_tokens, triton.cdiv(num_heads, block_h))](
        q,
        swa_u8,
        swa_bf16,
        swa_indices,
        swa_lens,
        extra_u8,
        extra_bf16,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_heads=num_heads,
        swa_index_topk=swa_indices.shape[-1],
        extra_index_topk=extra_indices.shape[-1] if has_extra else 0,
        swa_num_blocks=swa_u8.shape[0],
        extra_num_blocks=extra_u8.shape[0],
        swa_block_size=swa_block_size,
        extra_block_size=extra_block_size,
        swa_stride_block_bytes=swa_u8.stride(0),
        extra_stride_block_bytes=extra_u8.stride(0),
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
        block_d=block_d,
        nope_d=NOPE_DIM,
        group_d=GROUP_SIZE,
        num_scale_groups=NUM_GROUPS,
        token_bytes=STORAGE_BYTES_PER_TOKEN,
        rope_offset=ROPE_OFFSET_BYTES,
        scale_offset=SCALE_OFFSET_BYTES,
        has_extra=has_extra,
        has_sink=attn_sink is not None,
        num_stages=1,
        num_warps=num_warps,
    )


__all__ = ["decode_sparse_attention_int4"]
