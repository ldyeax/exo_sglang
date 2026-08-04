# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from vllm-project/vllm PR #40929 (head: bbbearxyz/vllm@10934adf,
# vllm/model_executor/layers/deepseek_v4_triton_kernels.py).
# Provides a portable Triton implementation of DeepSeek V4 sparse FP8
# MLA decode for hardware where the standalone flash_mla CUDA kernel is
# unavailable (notably SM_120 / RTX 5090).
"""Triton fallback for DeepSeek V4 sparse FP8 MLA decode."""

import functools

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    get_e4m3fn_decode_lut,
)

LOG2E = 1.4426950408889634

DEEPSEEK_V4_MLA_HEAD_DIM = 512
FP8_DS_MLA_FP8_DIM = 448
FP8_DS_MLA_SCALE_GROUP = 64
FP8_DS_MLA_SCALE_BYTES = 8
FP8_DS_MLA_TOKEN_BYTES = 576


@functools.lru_cache(maxsize=None)
def _decode_launch_config(device: torch.device) -> tuple[int, int, int]:
    """Return (heads, keys, warps) tuned without querying during graph replay."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 6):
        # A 16-head tile fills Ampere's native MMA M tile, while four warps
        # avoid the scheduling overhead measured with eight warps.  BLOCK_N=32
        # exceeds GA102's opt-in shared-memory limit for this 512-D fused QK/PV
        # kernel, so 16 is intentionally retained.
        return 16, 16, 4
    return 8, 16, 8


@triton.jit
def _decode_sparse_attention_fp8_kernel(
    q_ptr,
    swa_cache_u8_ptr,
    swa_cache_bf16_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_u8_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    lut_ptr,
    sink_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
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
    stride_swa_indices_t: tl.constexpr,
    stride_swa_indices_k: tl.constexpr,
    stride_extra_indices_t: tl.constexpr,
    stride_extra_indices_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_DIM: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
    LOG2E_CONST: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=mask_h, other=-float("inf"))
        e_max = sink * LOG2E_CONST
        e_sum = tl.where(mask_h, 1.0, 0.0)
    else:
        e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id) if HAS_EXTRA else 0
    total_len = extra_len + swa_len

    # Loop only over the actual KV positions, not the allocated tensor shape:
    # extra_index_topk + swa_index_topk reflects the page-table preallocation
    # for the full max_position_embeddings (e.g. ~8192 BLOCK_N tiles for V4
    # odd layers at compress_ratio=128 with 1M context), while total_len is
    # the actual per-token KV length (often a few tiles). In-loop masks
    # already zero out invalid lanes, so output is unaffected, but every
    # extra iteration still loads HBM and dispatches QK / softmax / V matmuls.
    loop_end = tl.cdiv(total_len, BLOCK_N) * BLOCK_N
    for start in range(0, loop_end, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        use_extra = HAS_EXTRA & (offs_n < extra_len)
        use_swa = (offs_n >= extra_len) & (offs_n < total_len)

        extra_cols = offs_n
        swa_cols = offs_n - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_indices_t
            + extra_cols * stride_extra_indices_k,
            mask=HAS_EXTRA & (extra_cols < extra_index_topk),
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr
            + token_id * stride_swa_indices_t
            + swa_cols * stride_swa_indices_k,
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

        extra_token_base = extra_block * extra_stride_block_bytes
        extra_token_base += extra_pos * TOKEN_BYTES
        swa_token_base = swa_block * swa_stride_block_bytes
        swa_token_base += swa_pos * TOKEN_BYTES
        token_base = tl.where(use_extra, extra_token_base, swa_token_base)
        block_size = tl.where(use_extra, extra_block_size, swa_block_size)
        stride_block_bytes = tl.where(
            use_extra, extra_stride_block_bytes, swa_stride_block_bytes
        )
        pos = tl.where(use_extra, extra_pos, swa_pos)

        is_fp8 = offs_d < FP8_DIM
        # Each token has only seven UE8M0 scales. Load each once per key and
        # broadcast it across its 64-D group instead of issuing one scale load
        # for every FP8 element. This is 7*BLOCK_N byte loads rather than
        # 448*BLOCK_N address/load lanes on the Ampere software-decode path.
        scale_base = (
            tl.where(use_extra, extra_block, swa_block) * stride_block_bytes
            + block_size * TOKEN_BYTES
            + pos * SCALE_BYTES
        )
        scale_ptr = tl.where(use_extra, extra_cache_u8_ptr, swa_cache_u8_ptr)
        scale_0 = tl.load(scale_ptr + scale_base, mask=valid, other=127).to(tl.float32)
        scale_1 = tl.load(scale_ptr + scale_base + 1, mask=valid, other=127).to(
            tl.float32
        )
        scale_2 = tl.load(scale_ptr + scale_base + 2, mask=valid, other=127).to(
            tl.float32
        )
        scale_3 = tl.load(scale_ptr + scale_base + 3, mask=valid, other=127).to(
            tl.float32
        )
        scale_4 = tl.load(scale_ptr + scale_base + 4, mask=valid, other=127).to(
            tl.float32
        )
        scale_5 = tl.load(scale_ptr + scale_base + 5, mask=valid, other=127).to(
            tl.float32
        )
        scale_6 = tl.load(scale_ptr + scale_base + 6, mask=valid, other=127).to(
            tl.float32
        )
        scale_group = offs_d // SCALE_GROUP
        encoded_scale = tl.where(scale_group[None, :] == 0, scale_0[:, None], 127.0)
        encoded_scale = tl.where(
            scale_group[None, :] == 1, scale_1[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group[None, :] == 2, scale_2[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group[None, :] == 3, scale_3[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group[None, :] == 4, scale_4[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group[None, :] == 5, scale_5[:, None], encoded_scale
        )
        encoded_scale = tl.where(
            scale_group[None, :] == 6, scale_6[:, None], encoded_scale
        )
        fp8_scale = tl.exp2(encoded_scale - 127.0)

        fp8_offsets = token_base[:, None] + offs_d[None, :]
        # Load nope as raw byte index (uint8) and decode FP8 E4M3 via
        # a pre-computed BF16 lookup table. Avoids Triton's fp8e4nv
        # type (unsupported on SM < 90).
        raw = tl.load(
            tl.where(use_extra[:, None], extra_cache_u8_ptr, swa_cache_u8_ptr)
            + fp8_offsets,
            mask=valid[:, None] & is_fp8[None, :],
            other=0,
        )
        # Force unsigned zero-extension: uint8→uint32→int32.
        # .to(tl.int32) on a uint8 value may sign-extend bytes ≥128
        # into negative int32, causing an out-of-bounds LUT access.
        idx = raw.to(tl.uint32).to(tl.int32)
        fp8_val = tl.load(lut_ptr + idx)
        fp8_vals = fp8_val * fp8_scale

        bf16_offsets = (token_base[:, None] + FP8_DIM) // 2
        bf16_offsets += offs_d[None, :] - FP8_DIM
        bf16_vals = tl.load(
            tl.where(use_extra[:, None], extra_cache_bf16_ptr, swa_cache_bf16_ptr)
            + bf16_offsets,
            mask=valid[:, None] & (~is_fp8[None, :]),
            other=0.0,
        ).to(tl.float32)
        k = tl.where(is_fp8[None, :], fp8_vals, bf16_vals)

        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale_log2
        qk = tl.where(
            mask_h[:, None] & valid[None, :],
            qk,
            -3.4028234663852886e38,
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & valid[None, :], p, 0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(k.dtype), k)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


# ---------------------------------------------------------------------------
# BF16-only decode kernel (SM_86, all-bf16 cache).
# The KV cache stores nope+rope as bf16 → no FP8 dequant, no LUT, no scale.
# ---------------------------------------------------------------------------

FP8_DS_BF16_NOPE_DIM = 448
FP8_DS_BF16_ROPE_DIM = 64
FP8_DS_BF16_TOKEN_ELEMS = FP8_DS_BF16_NOPE_DIM + FP8_DS_BF16_ROPE_DIM  # 512


@triton.jit
def _decode_sparse_attention_bf16_kernel(
    q_ptr,
    swa_cache_bf16_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    sink_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    swa_index_topk: tl.constexpr,
    extra_index_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_stride_block_elems: tl.constexpr,
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
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TOKEN_ELEMS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
    LOG2E_CONST: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=mask_h, other=-float("inf"))
        e_max = sink * LOG2E_CONST
        e_sum = tl.where(mask_h, 1.0, 0.0)
    else:
        e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id) if HAS_EXTRA else 0
    total_len = extra_len + swa_len

    loop_end = tl.cdiv(total_len, BLOCK_N) * BLOCK_N
    for start in range(0, loop_end, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        use_extra = HAS_EXTRA & (offs_n < extra_len)
        use_swa = (offs_n >= extra_len) & (offs_n < total_len)

        extra_cols = offs_n
        swa_cols = offs_n - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_idx_t
            + extra_cols * stride_extra_idx_k,
            mask=HAS_EXTRA & (extra_cols < extra_index_topk),
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr + token_id * stride_swa_idx_t + swa_cols * stride_swa_idx_k,
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

        # All-bf16 cache: each token is TOKEN_ELEMS (512) bf16 values,
        # contiguous nope[0:448] + rope[448:512]. One load gives both.
        extra_token_base = extra_block * extra_stride_block_elems
        extra_token_base += extra_pos * TOKEN_ELEMS
        swa_token_base = swa_block * swa_stride_block_elems
        swa_token_base += swa_pos * TOKEN_ELEMS
        token_base = tl.where(use_extra, extra_token_base, swa_token_base)

        k_offsets = token_base[:, None] + offs_d[None, :]
        k = tl.load(
            tl.where(use_extra[:, None], extra_cache_bf16_ptr, swa_cache_bf16_ptr)
            + k_offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale_log2
        qk = tl.where(mask_h[:, None] & valid[None, :], qk, -3.4028234663852886e38)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & valid[None, :], p, 0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(k.dtype), k)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def decode_sparse_attention_bf16(
    q,
    swa_cache,
    swa_indices,
    swa_lens,
    scale,
    attn_sink,
    out,
    extra_cache=None,
    extra_indices=None,
    extra_lens=None,
) -> None:
    """SM_86 BF16 decode: cache is all-bf16, no FP8 at all."""
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)

    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return

    has_extra = bool(
        extra_cache is not None and extra_indices is not None and extra_lens is not None
    )
    if not has_extra:
        extra_cache = swa_cache
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens

    BLOCK_H, BLOCK_N, BLOCK_D = 8, 16, 512
    grid = (num_tokens, triton.cdiv(num_heads, BLOCK_H))

    # Cache is viewed as bf16: [num_blocks, block_size, TOKEN_ELEMS]
    swa_cache_bf16 = swa_cache.view(torch.bfloat16)
    extra_cache_bf16 = extra_cache.view(torch.bfloat16)

    _decode_sparse_attention_bf16_kernel[grid](
        q,
        swa_cache_bf16,
        swa_indices,
        swa_lens,
        extra_cache_bf16,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_tokens,
        num_heads,
        swa_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        swa_cache_bf16.shape[0],
        extra_cache_bf16.shape[0],
        swa_cache_bf16.shape[1],
        extra_cache_bf16.shape[1],
        swa_cache_bf16.stride(0),
        extra_cache_bf16.stride(0),
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
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NOPE_DIM=FP8_DS_BF16_NOPE_DIM,
        ROPE_DIM=FP8_DS_BF16_ROPE_DIM,
        TOKEN_ELEMS=FP8_DS_BF16_TOKEN_ELEMS,
        HAS_EXTRA=has_extra,
        HAS_SINK=attn_sink is not None,
        LOG2E_CONST=LOG2E,
        num_stages=1,
        num_warps=8,
    )


def _resolve_packed_page_size(
    cache: torch.Tensor,
    explicit_page_size: int | None,
    *,
    name: str,
) -> int:
    if explicit_page_size is not None:
        if explicit_page_size <= 0:
            raise ValueError(f"{name}_block_size must be positive")
        return explicit_page_size

    # FlashMLA presents the raw page storage as
    # [num_pages, page_size, 1, 584].  Preserve that compatibility, while
    # requiring callers with the allocator's native 2-D padded page buffer to
    # provide the logical page size explicitly.
    if cache.ndim >= 3:
        page_size = int(cache.shape[1])
        if page_size > 0:
            return page_size
    raise ValueError(
        f"{name}_block_size is required for a 2-D packed cache; got "
        f"{tuple(cache.shape)}"
    )


def _prepare_packed_page_buffer(
    cache: torch.Tensor,
    page_size: int,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if cache.ndim < 2:
        raise ValueError(f"{name} must have a page dimension, got {cache.shape}")
    if cache.element_size() != 1:
        raise TypeError(
            f"{name} must use one-byte E4M3FN/uint8 storage, got {cache.dtype}"
        )

    # Never expose a float8 pointer to Triton on Ampere.  A dtype view is
    # allocation-free and retains the allocator's padded outer-page stride.
    cache_u8 = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    if cache_u8.stride(-1) != 1:
        raise ValueError(f"{name}'s innermost byte dimension must be contiguous")
    page_stride_bytes = cache_u8.stride(0)
    required_page_bytes = page_size * (FP8_DS_MLA_TOKEN_BYTES + FP8_DS_MLA_SCALE_BYTES)
    if page_stride_bytes < required_page_bytes:
        raise ValueError(
            f"{name} page stride is {page_stride_bytes} bytes, but page size "
            f"{page_size} requires at least {required_page_bytes} bytes"
        )
    if page_stride_bytes % 2 or cache_u8.storage_offset() % 2:
        raise ValueError(f"{name} must be BF16-aligned")

    # RoPE occupies bytes [448, 576) of each value record.  Use a second typed
    # view solely for those loads; all addresses passed to the kernel remain
    # explicit offsets from the same stable storage.
    return cache_u8, cache_u8.view(torch.bfloat16), page_stride_bytes


def decode_sparse_attention_triton(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    *,
    swa_block_size: int | None = None,
    extra_block_size: int | None = None,
) -> None:
    """Run V4 sparse FP8 MLA decode through a portable Triton kernel.

    The kernel expects:
      q          : (N_tokens, num_heads, 512) bf16
      swa_cache  : raw uint8/E4M3FN page storage. Both the native allocator
                   shape ``(num_pages, padded_page_bytes)`` and FlashMLA's
                   logical shape ``(num_pages, page_size, 1, 584)`` work.
                   Physical layout is ``page_size * 576`` value bytes followed
                   by ``page_size * 8`` UE8M0 scale bytes; the outer page stride
                   may include allocator padding.
      swa_indices: (N_tokens, topk_swa) int32 (gets squeezed if 3D)
      swa_lens   : (N_tokens,) int32
      out        : (N_tokens, num_heads, 512) bf16, written in place
    """
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)

    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if head_dim != DEEPSEEK_V4_MLA_HEAD_DIM:
        raise ValueError(
            "DeepSeek V4 decode Triton fallback expects "
            f"D={DEEPSEEK_V4_MLA_HEAD_DIM}, got {head_dim}"
        )
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("DeepSeek V4 Ampere attention requires BF16 q and out")

    has_extra = (
        extra_cache is not None and extra_indices is not None and extra_lens is not None
    )
    resolved_swa_block_size = _resolve_packed_page_size(
        swa_cache, swa_block_size, name="swa_cache"
    )
    if not has_extra:
        extra_cache = swa_cache
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens
        resolved_extra_block_size = resolved_swa_block_size
    else:
        assert extra_cache is not None
        resolved_extra_block_size = _resolve_packed_page_size(
            extra_cache, extra_block_size, name="extra_cache"
        )

    assert extra_cache is not None
    assert extra_indices is not None
    assert extra_lens is not None
    swa_cache_u8, swa_cache_bf16, swa_page_stride_bytes = _prepare_packed_page_buffer(
        swa_cache,
        resolved_swa_block_size,
        name="swa_cache",
    )
    extra_cache_u8, extra_cache_bf16, extra_page_stride_bytes = (
        _prepare_packed_page_buffer(
            extra_cache,
            resolved_extra_block_size,
            name="extra_cache",
        )
    )
    block_h, block_n, num_warps = _decode_launch_config(q.device)
    grid = (num_tokens, triton.cdiv(num_heads, block_h))
    # SM < 89 cannot consume a native float8 pointer. Decode the exact
    # E4M3FN bit pattern through a persistent per-device BF16 LUT instead.
    lut = get_e4m3fn_decode_lut(q.device)
    _decode_sparse_attention_fp8_kernel[grid](
        q,
        swa_cache_u8,
        swa_cache_bf16,
        swa_indices,
        swa_lens,
        extra_cache_u8,
        extra_cache_bf16,
        extra_indices,
        extra_lens,
        lut,  # Raw E4M3FN byte -> BF16 LUT
        attn_sink if attn_sink is not None else q,
        out,
        num_tokens,
        num_heads,
        swa_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        swa_cache_u8.shape[0],
        extra_cache_u8.shape[0],
        resolved_swa_block_size,
        resolved_extra_block_size,
        swa_page_stride_bytes,
        extra_page_stride_bytes,
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
        BLOCK_D=DEEPSEEK_V4_MLA_HEAD_DIM,
        FP8_DIM=FP8_DS_MLA_FP8_DIM,
        SCALE_GROUP=FP8_DS_MLA_SCALE_GROUP,
        SCALE_BYTES=FP8_DS_MLA_SCALE_BYTES,
        TOKEN_BYTES=FP8_DS_MLA_TOKEN_BYTES,
        HAS_EXTRA=has_extra,
        HAS_SINK=attn_sink is not None,
        LOG2E_CONST=LOG2E,
        num_stages=1,  # SM_86 tuned: single-stage pipeline fits 99KB LDS
        num_warps=num_warps,
    )
