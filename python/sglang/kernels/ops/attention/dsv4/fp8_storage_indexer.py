"""Ampere C4 indexer over architecture-neutral E4M3FN byte storage.

DeepSeek V4 stores each 64-token indexer page as 8,192 E4M3FN value
bytes followed by 64 FP32 per-token scales.  Ampere cannot consume an FP8
tensor in Triton, so this kernel loads the value bytes as unsigned integers,
decodes them through the graph-stable E4M3FN LUT, and feeds BF16 operands to
Ampere tensor cores.

The public wrapper intentionally mirrors the existing paged-MQA scorer
signature.  It accepts a logical ``torch.float8_e4m3fn`` query but passes only
``uint8`` pointers to Triton.  No cache-sized dequantization workspace is
materialized.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    get_e4m3fn_decode_lut,
)

DSV4_INDEXER_PAGE_SIZE = 64
DSV4_INDEXER_NUM_HEADS = 64
DSV4_INDEXER_HEAD_DIM = 128
DSV4_INDEXER_VALUE_BYTES = DSV4_INDEXER_PAGE_SIZE * DSV4_INDEXER_HEAD_DIM
DSV4_INDEXER_SCALE_BYTES = DSV4_INDEXER_PAGE_SIZE * 4
DSV4_INDEXER_PAGE_BYTES = DSV4_INDEXER_VALUE_BYTES + DSV4_INDEXER_SCALE_BYTES
# Enough independent programs to saturate Ampere for prefill-sized query
# batches.  Decode-sized batches use a smaller per-query persistent grid so a
# graph captured for 128K does not launch thousands of empty CTAs when only a
# handful of pages are live.
DSV4_INDEXER_TARGET_PROGRAMS = 2048
DSV4_INDEXER_SINGLE_QUERY_PROGRAMS = 256
DSV4_INDEXER_SMALL_BATCH_PROGRAMS_PER_QUERY = 64
DSV4_INDEXER_SMALL_BATCH_LIMIT = 8
DSV4_INDEXER_DIRECT_MAX_PAGES = 64


@triton.jit
def _fp8_storage_paged_mqa_logits_direct_kernel(
    q_u8_ptr,
    cache_u8_ptr,
    cache_f32_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    lut_ptr,
    output_ptr,
    cache_page_stride_bytes: tl.constexpr,
    cache_page_stride_f32: tl.constexpr,
    page_table_stride: tl.constexpr,
    max_seq_len: tl.constexpr,
    page_table_width: tl.constexpr,
    num_cache_pages: tl.constexpr,
    programs_per_query: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_bytes: tl.constexpr,
):
    _ = programs_per_query
    batch_idx = tl.program_id(0)
    page_slot = tl.program_id(1)
    page_start = page_slot * page_size
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), max_seq_len)

    # This loop-free specialization avoids dynamic-loop overhead for genuinely
    # short captured widths, where one program per possible page is cheap.
    if (page_start < bounded_seq_len) & (page_slot < page_table_width):
        token_offsets = tl.arange(0, page_size)
        head_offsets = tl.arange(0, num_heads)
        dim_offsets = tl.arange(0, head_dim)
        page_id = tl.load(page_table_ptr + batch_idx * page_table_stride + page_slot)
        page_is_valid = (page_id >= 0) & (page_id < num_cache_pages)
        safe_page_id = tl.where(page_is_valid, page_id, 0)

        q_byte_offsets = (
            batch_idx * num_heads * head_dim
            + head_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        q_raw = tl.load(q_u8_ptr + q_byte_offsets)
        q_indices = q_raw.to(tl.uint32).to(tl.int32)
        q_bf16 = tl.load(lut_ptr + q_indices)

        cache_byte_offsets = (
            safe_page_id * cache_page_stride_bytes
            + token_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        k_raw = tl.load(
            cache_u8_ptr + cache_byte_offsets,
            mask=page_is_valid,
            other=0,
        )
        k_indices = k_raw.to(tl.uint32).to(tl.int32)
        k_bf16 = tl.load(lut_ptr + k_indices)

        logits = tl.dot(k_bf16, tl.trans(q_bf16), out_dtype=tl.float32)
        logits = tl.maximum(logits, 0.0)
        weights = tl.load(weight_ptr + batch_idx * num_heads + head_offsets).to(
            tl.float32
        )
        logits_sum = tl.sum(logits * weights[None, :], axis=1)

        scale_offsets = (
            safe_page_id * cache_page_stride_f32 + value_bytes // 4 + token_offsets
        )
        k_scales = tl.load(
            cache_f32_ptr + scale_offsets,
            mask=page_is_valid,
            other=0.0,
        )
        logits_sum *= k_scales
        logits_sum = tl.where(page_is_valid, logits_sum, 0.0)
        output_positions = page_start + token_offsets
        tl.store(
            output_ptr + batch_idx * max_seq_len + output_positions,
            logits_sum,
            mask=output_positions < bounded_seq_len,
        )


@triton.jit
def _fp8_storage_paged_mqa_logits_persistent_kernel(
    q_u8_ptr,
    cache_u8_ptr,
    cache_f32_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    lut_ptr,
    output_ptr,
    cache_page_stride_bytes: tl.constexpr,
    cache_page_stride_f32: tl.constexpr,
    page_table_stride: tl.constexpr,
    max_seq_len: tl.constexpr,
    page_table_width: tl.constexpr,
    num_cache_pages: tl.constexpr,
    programs_per_query: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_bytes: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    program_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), max_seq_len)
    active_pages = tl.minimum(tl.cdiv(bounded_seq_len, page_size), page_table_width)

    # The launch grid depends only on captured shapes.  Runtime sequence lengths
    # bound a persistent, strided page loop on replay.  Program p is the sole
    # writer of pages p, p + P, p + 2P, ... for this query, so every active page
    # is covered exactly once without changing the fixed output layout.
    if program_idx < active_pages:
        head_offsets = tl.arange(0, num_heads)
        dim_offsets = tl.arange(0, head_dim)
        token_offsets = tl.arange(0, page_size)

        q_byte_offsets = (
            batch_idx * num_heads * head_dim
            + head_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        q_raw = tl.load(q_u8_ptr + q_byte_offsets)
        # Preserve bytes >= 128 as positive lookup-table indices.  A direct
        # uint8 -> int32 conversion has sign-extended in older Triton builds.
        q_indices = q_raw.to(tl.uint32).to(tl.int32)
        q_bf16 = tl.load(lut_ptr + q_indices)

        weights = tl.load(weight_ptr + batch_idx * num_heads + head_offsets).to(
            tl.float32
        )

        for page_slot in tl.range(program_idx, active_pages, programs_per_query):
            page_start = page_slot * page_size
            output_positions = page_start + token_offsets
            page_id = tl.load(
                page_table_ptr + batch_idx * page_table_stride + page_slot,
                mask=page_slot < page_table_width,
                other=-1,
            )
            page_is_valid = (page_id >= 0) & (page_id < num_cache_pages)
            safe_page_id = tl.where(page_is_valid, page_id, 0)

            cache_byte_offsets = (
                safe_page_id * cache_page_stride_bytes
                + token_offsets[:, None] * head_dim
                + dim_offsets[None, :]
            )
            k_raw = tl.load(
                cache_u8_ptr + cache_byte_offsets,
                mask=page_is_valid,
                other=0,
            )
            k_indices = k_raw.to(tl.uint32).to(tl.int32)
            k_bf16 = tl.load(lut_ptr + k_indices)

            # [64 tokens, 128 dims] @ [128 dims, 64 heads]. Both operands are
            # BF16 after table decoding, so SM86 lowers this to BF16 HMMA.
            logits = tl.dot(k_bf16, tl.trans(q_bf16), out_dtype=tl.float32)
            logits = tl.maximum(logits, 0.0)
            logits_sum = tl.sum(logits * weights[None, :], axis=1)

            scale_offsets = (
                safe_page_id * cache_page_stride_f32 + value_bytes // 4 + token_offsets
            )
            k_scales = tl.load(
                cache_f32_ptr + scale_offsets,
                mask=page_is_valid,
                other=0.0,
            )
            logits_sum *= k_scales
            logits_sum = tl.where(page_is_valid, logits_sum, 0.0)
            tl.store(
                output_ptr + batch_idx * max_seq_len + output_positions,
                logits_sum,
                mask=output_positions < bounded_seq_len,
            )


def _programs_per_query(batch_size: int, max_pages: int) -> int:
    """Return a shape-static persistent-grid width.

    One query needs at least one wave across Ampere's SMs for a long context,
    while a small decode batch already supplies inter-query parallelism.  Large
    prefill batches retain the previous 2,048-program aggregate target without
    paying that target independently for every query.
    """
    if batch_size == 1:
        target = DSV4_INDEXER_SINGLE_QUERY_PROGRAMS
    elif batch_size <= DSV4_INDEXER_SMALL_BATCH_LIMIT:
        target = DSV4_INDEXER_SMALL_BATCH_PROGRAMS_PER_QUERY
    else:
        target = triton.cdiv(DSV4_INDEXER_TARGET_PROGRAMS, batch_size)
    return max(1, min(max_pages, target))


def _as_e4m3fn_bytes(query: torch.Tensor) -> torch.Tensor:
    if query.dtype == torch.uint8:
        return query
    if query.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "query must contain E4M3FN bytes as torch.float8_e4m3fn or "
            f"torch.uint8, got {query.dtype}"
        )
    return query.view(torch.uint8)


def fp8_storage_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    kvcache_u8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
    *,
    out: torch.Tensor | None = None,
    _programs_per_query_override: int | None = None,
    _force_direct_kernel: bool = False,
) -> torch.Tensor:
    """Score packed C4 indexer pages using BF16 tensor cores on SM86.

    Args:
        q_fp8: Contiguous E4M3FN query with shape ``[B, 1, 64, 128]``.
        kvcache_u8: Raw packed cache with shape ``[num_pages, 8448]``.
        weight: FP32 head weights, including the query's quantization scale,
            with shape ``[B, 64]``.
        seq_lens: INT32 sequence lengths with shape ``[B]`` or ``[B, 1]``.
        page_table: INT32 physical-page IDs with shape ``[B, max_pages]``.
        deep_gemm_metadata: Accepted for scorer API compatibility; unused.
        max_seq_len: Output width. Must fit in ``page_table``.
        clean_logits: Clear positions outside each sequence before scoring.
            Production passes ``False`` because downstream top-k is length
            masked; this avoids writing the potentially very large padded tail.
        out: Optional contiguous FP32 output with shape ``[B, max_seq_len]``.
        _programs_per_query_override: Private benchmark/testing override for the
            persistent grid width.  Production callers must leave this unset.
        _force_direct_kernel: Private benchmark override for the old loop-free
            one-program-per-page kernel.

    The LUT and JIT kernel must be primed by an eager call before CUDA graph
    capture.  Once primed, the wrapper performs no device allocation when
    ``out`` is supplied.
    """
    del deep_gemm_metadata

    if q_fp8.ndim != 4:
        raise ValueError(f"query must be rank 4, got shape {tuple(q_fp8.shape)}")
    batch_size = q_fp8.shape[0]
    if batch_size == 0:
        raise ValueError("query batch must not be empty")
    expected_query_shape = (
        batch_size,
        1,
        DSV4_INDEXER_NUM_HEADS,
        DSV4_INDEXER_HEAD_DIM,
    )
    if tuple(q_fp8.shape) != expected_query_shape:
        raise ValueError(
            f"query must have shape {expected_query_shape}, got {tuple(q_fp8.shape)}"
        )
    query_u8 = _as_e4m3fn_bytes(q_fp8)
    if not query_u8.is_contiguous():
        raise ValueError("query must be contiguous")

    if kvcache_u8.dtype != torch.uint8:
        raise TypeError(f"cache must be torch.uint8, got {kvcache_u8.dtype}")
    if kvcache_u8.ndim != 2 or kvcache_u8.shape[1] != DSV4_INDEXER_PAGE_BYTES:
        raise ValueError(
            f"cache must have shape [num_pages, 8448], got {tuple(kvcache_u8.shape)}"
        )
    if kvcache_u8.stride(1) != 1:
        raise ValueError("cache rows must have contiguous byte storage")
    cache_page_stride_bytes = kvcache_u8.stride(0)
    if cache_page_stride_bytes < DSV4_INDEXER_PAGE_BYTES:
        raise ValueError("cache page stride is smaller than its packed layout")
    if cache_page_stride_bytes % 4 or kvcache_u8.storage_offset() % 4:
        raise ValueError("cache pages and storage offset must be FP32-aligned")
    cache_f32 = kvcache_u8.view(torch.float32)
    cache_page_stride_f32 = cache_f32.stride(0)

    if weight.dtype != torch.float32 or tuple(weight.shape) != (
        batch_size,
        DSV4_INDEXER_NUM_HEADS,
    ):
        raise ValueError(
            f"weight must be FP32 [{batch_size}, 64], got "
            f"{weight.dtype} {tuple(weight.shape)}"
        )
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    valid_seq_lens_shape = tuple(seq_lens.shape) in {
        (batch_size,),
        (batch_size, 1),
    }
    if seq_lens.dtype != torch.int32 or not valid_seq_lens_shape:
        raise ValueError(
            f"seq_lens must be INT32 [{batch_size}] or [{batch_size}, 1], got "
            f"{seq_lens.dtype} {tuple(seq_lens.shape)}"
        )
    if not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be contiguous")
    seq_lens_flat = seq_lens.view(batch_size)
    if page_table.dtype != torch.int32 or page_table.ndim != 2:
        raise ValueError(
            "page_table must be a rank-2 INT32 tensor, got "
            f"{page_table.dtype} {tuple(page_table.shape)}"
        )
    if page_table.shape[0] != batch_size or page_table.stride(1) != 1:
        raise ValueError("page_table must have contiguous rows for the query batch")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len}")
    if max_seq_len > page_table.shape[1] * DSV4_INDEXER_PAGE_SIZE:
        raise ValueError("max_seq_len exceeds the supplied page-table capacity")

    device = q_fp8.device
    tensors = (kvcache_u8, weight, seq_lens, page_table)
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError("all scorer tensors must reside on the same CUDA device")

    if out is None:
        out = torch.empty((batch_size, max_seq_len), dtype=torch.float32, device=device)
    elif (
        out.dtype != torch.float32
        or tuple(out.shape) != (batch_size, max_seq_len)
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous FP32 [{batch_size}, {max_seq_len}] on {device}"
        )
    if clean_logits:
        out.zero_()

    lut = get_e4m3fn_decode_lut(device)
    max_pages = triton.cdiv(max_seq_len, DSV4_INDEXER_PAGE_SIZE)
    use_direct_kernel = _force_direct_kernel or (
        _programs_per_query_override is None
        and max_pages <= DSV4_INDEXER_DIRECT_MAX_PAGES
    )
    if _programs_per_query_override is not None:
        if (
            not isinstance(_programs_per_query_override, int)
            or isinstance(_programs_per_query_override, bool)
            or _programs_per_query_override <= 0
        ):
            raise ValueError("_programs_per_query_override must be a positive integer")
        programs_per_query = min(max_pages, _programs_per_query_override)
    else:
        programs_per_query = _programs_per_query(batch_size, max_pages)
    if use_direct_kernel:
        programs_per_query = max_pages
    grid = (batch_size, programs_per_query)
    num_warps = 8 if max_pages <= 64 else 4
    kernel = (
        _fp8_storage_paged_mqa_logits_direct_kernel
        if use_direct_kernel
        else _fp8_storage_paged_mqa_logits_persistent_kernel
    )
    kernel[grid](
        query_u8,
        kvcache_u8,
        cache_f32,
        weight,
        seq_lens_flat,
        page_table,
        lut,
        out,
        cache_page_stride_bytes=cache_page_stride_bytes,
        cache_page_stride_f32=cache_page_stride_f32,
        page_table_stride=page_table.stride(0),
        max_seq_len=max_seq_len,
        page_table_width=page_table.shape[1],
        num_cache_pages=kvcache_u8.shape[0],
        programs_per_query=programs_per_query,
        page_size=DSV4_INDEXER_PAGE_SIZE,
        num_heads=DSV4_INDEXER_NUM_HEADS,
        head_dim=DSV4_INDEXER_HEAD_DIM,
        value_bytes=DSV4_INDEXER_VALUE_BYTES,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


__all__ = [
    "DSV4_INDEXER_PAGE_BYTES",
    "DSV4_INDEXER_PAGE_SIZE",
    "fp8_storage_paged_mqa_logits_triton",
]
