import math

import torch
import triton
import triton.language as tl


LOG2E = math.log2(math.e)
BF16_TOKEN_ELEMS = 512


@triton.jit
def _decode_sparse_attention_bf16_kernel(
    q_ptr,
    swa_cache_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_ptr,
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
    swa_stride_block: tl.constexpr,
    extra_stride_block: tl.constexpr,
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
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_mask = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + dims[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )
    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=head_mask, other=-float("inf"))
        running_max = sink * 1.4426950408889634
        running_sum = tl.where(head_mask, 1.0, 0.0)
    else:
        running_max = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id) if HAS_EXTRA else 0
    total_len = swa_len + extra_len
    loop_end = tl.cdiv(total_len, BLOCK_N) * BLOCK_N
    for start in range(0, loop_end, BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        use_extra = HAS_EXTRA & (offsets < extra_len)
        use_swa = (offsets >= extra_len) & (offsets < total_len)
        extra_cols = offsets
        swa_cols = offsets - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_idx_t
            + extra_cols * stride_extra_idx_k,
            mask=HAS_EXTRA & (extra_cols < extra_index_topk),
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr
            + token_id * stride_swa_idx_t
            + swa_cols * stride_swa_idx_k,
            mask=(swa_cols >= 0) & (swa_cols < swa_index_topk),
            other=-1,
        )
        idx = tl.where(use_extra, extra_idx, swa_idx)
        extra_block = idx // extra_block_size
        extra_pos = idx - extra_block * extra_block_size
        swa_block = idx // swa_block_size
        swa_pos = idx - swa_block * swa_block_size
        valid_extra = use_extra & (idx >= 0) & (extra_block < extra_num_blocks)
        valid_swa = use_swa & (idx >= 0) & (swa_block < swa_num_blocks)
        valid = valid_extra | valid_swa

        extra_base = (
            extra_block * extra_stride_block + extra_pos * 512
        )
        swa_base = swa_block * swa_stride_block + swa_pos * 512
        token_base = tl.where(use_extra, extra_base, swa_base)
        k = tl.load(
            tl.where(use_extra[:, None], extra_cache_ptr, swa_cache_ptr)
            + token_base[:, None]
            + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale_log2
        qk = tl.where(
            head_mask[:, None] & valid[None, :], qk, -3.4028234663852886e38
        )
        next_max = tl.maximum(tl.max(qk, 1), running_max)
        rescale = tl.exp2(running_max - next_max)
        probability = tl.exp2(qk - next_max[:, None])
        probability = tl.where(
            head_mask[:, None] & valid[None, :], probability, 0.0
        )
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(k.dtype), k
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


def decode_sparse_attention_bf16(
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
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)
    num_tokens, num_heads, _ = q.shape
    if num_tokens == 0:
        return
    has_extra = bool(
        extra_cache is not None
        and extra_indices is not None
        and extra_lens is not None
    )
    if not has_extra:
        extra_cache = swa_cache
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens
        extra_block_size = swa_block_size
    assert extra_cache is not None
    assert extra_indices is not None
    assert extra_lens is not None
    assert extra_block_size is not None

    swa_bf16 = swa_cache.view(torch.bfloat16)
    extra_bf16 = extra_cache.view(torch.bfloat16)
    block_h, block_n, block_d = 8, 16, 512
    grid = (num_tokens, triton.cdiv(num_heads, block_h))
    _decode_sparse_attention_bf16_kernel[grid](
        q,
        swa_bf16,
        swa_indices,
        swa_lens,
        extra_bf16,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_heads,
        swa_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        swa_bf16.shape[0],
        extra_bf16.shape[0],
        swa_block_size,
        extra_block_size,
        swa_bf16.stride(0),
        extra_bf16.stride(0),
        scale * LOG2E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_indices.stride(0),
        swa_indices.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        HAS_EXTRA=has_extra,
        HAS_SINK=attn_sink is not None,
        num_stages=1,
        num_warps=8,
    )
