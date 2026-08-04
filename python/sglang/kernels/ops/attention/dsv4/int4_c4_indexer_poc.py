"""Signed-INT4 storage and scorer for the DeepSeek V4 C4 indexer.

The production SM86 opt-in path uses architecture-neutral ``torch.uint8``
pages and treats INT4 strictly as a storage format: keys are unpacked to BF16
inside the scorer immediately before the Ampere tensor-core dot.

Each 64-token page has the following exact layout::

    [0, 4096)       64 tokens x 64 packed bytes (two signed nibbles/byte)
    [4096, 4608)    64 tokens x four BF16 scales (one per 32 dimensions)

Within a packed byte, the even dimension occupies the low nibble and the odd
dimension occupies the high nibble.  Codes use four-bit two's-complement
storage, while the symmetric quantizer emits only ``[-7, 7]``.

The scorer implements the existing C4 semantics: BF16 query/key dot products,
ReLU per query head, then an FP32 weighted reduction over 64 heads.  Supplying
``out`` makes the hot wrapper allocation-free and suitable for CUDA-graph
capture after the Triton specialization has been primed eagerly.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    get_e4m3fn_decode_lut,
)

INT4_C4_POC_PAGE_SIZE = 64
INT4_C4_POC_NUM_HEADS = 64
INT4_C4_POC_HEAD_DIM = 128
INT4_C4_POC_GROUP_SIZE = 32
INT4_C4_POC_NUM_GROUPS = INT4_C4_POC_HEAD_DIM // INT4_C4_POC_GROUP_SIZE
INT4_C4_POC_PACKED_BYTES_PER_TOKEN = INT4_C4_POC_HEAD_DIM // 2
INT4_C4_POC_VALUE_BYTES = INT4_C4_POC_PAGE_SIZE * INT4_C4_POC_PACKED_BYTES_PER_TOKEN
INT4_C4_POC_SCALE_BYTES = (
    INT4_C4_POC_PAGE_SIZE * INT4_C4_POC_NUM_GROUPS * torch.bfloat16.itemsize
)
INT4_C4_POC_PAGE_BYTES = INT4_C4_POC_VALUE_BYTES + INT4_C4_POC_SCALE_BYTES
INT4_C4_BYTES_PER_TOKEN = (
    INT4_C4_POC_PACKED_BYTES_PER_TOKEN
    + INT4_C4_POC_NUM_GROUPS * torch.bfloat16.itemsize
)
INT4_C4_POC_TARGET_PROGRAMS = 2048
INT4_C4_POC_SINGLE_QUERY_PROGRAMS = 256
INT4_C4_POC_SMALL_BATCH_PROGRAMS_PER_QUERY = 64
INT4_C4_POC_SMALL_BATCH_LIMIT = 8

_INT4_MIN = -7
_INT4_MAX = 7


def int4_c4_page_bytes(page_size: int) -> int:
    """Return exact bytes for one production C4-indexer page."""
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError(f"page_size must be a positive integer, got {page_size!r}")
    return page_size * INT4_C4_BYTES_PER_TOKEN


def _validate_reference_pages(cache_u8: torch.Tensor) -> None:
    if cache_u8.dtype != torch.uint8:
        raise TypeError(f"cache must be torch.uint8, got {cache_u8.dtype}")
    if cache_u8.ndim != 2 or cache_u8.shape[1] != INT4_C4_POC_PAGE_BYTES:
        raise ValueError(
            "cache must have shape "
            f"[num_pages, {INT4_C4_POC_PAGE_BYTES}], got {tuple(cache_u8.shape)}"
        )
    if cache_u8.stride(1) != 1:
        raise ValueError("cache rows must have contiguous byte storage")


def store_int4_c4_indexer_cache(
    keys: torch.Tensor,
    cache_u8: torch.Tensor,
    locations: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Quantize/scatter complete BF16 C4 keys into caller-owned pages.

    Values occupy the leading ``page_size * 64`` bytes of every page and the
    four BF16 scales per token occupy the trailing ``page_size * 8`` bytes.
    Negative locations are padding and are skipped.  The operation allocates
    nothing and is CUDA-graph replay safe once its specialization is primed.
    """

    if (
        not keys.is_cuda
        or keys.dtype != torch.bfloat16
        or keys.ndim != 2
        or keys.shape[1] != INT4_C4_POC_HEAD_DIM
        or keys.stride(1) != 1
    ):
        raise ValueError(
            "keys must be CUDA BF16 with shape "
            f"[num_tokens, {INT4_C4_POC_HEAD_DIM}] and contiguous rows"
        )
    expected_page_bytes = int4_c4_page_bytes(page_size)
    if (
        not cache_u8.is_cuda
        or cache_u8.dtype != torch.uint8
        or cache_u8.ndim != 2
        or cache_u8.shape[1] != expected_page_bytes
        or cache_u8.stride(1) != 1
    ):
        raise ValueError(
            "cache must be CUDA uint8 with shape "
            f"[num_pages, {expected_page_bytes}] and contiguous rows"
        )
    if cache_u8.stride(0) % 2 or cache_u8.storage_offset() % 2:
        raise ValueError("cache pages and storage offset must be BF16 aligned")
    if (
        not locations.is_cuda
        or locations.ndim != 1
        or locations.dtype not in (torch.int32, torch.int64)
        or locations.numel() != keys.shape[0]
    ):
        raise ValueError("locations must be CUDA int32/int64 with one entry per key")
    if keys.device != cache_u8.device or locations.device != cache_u8.device:
        raise ValueError("keys, cache, and locations must share one CUDA device")
    if keys.shape[0] == 0:
        return

    cache_bf16 = cache_u8.view(torch.bfloat16)
    _store_int4_c4_indexer_cache_kernel[(keys.shape[0],)](
        keys,
        cache_u8,
        cache_bf16,
        locations,
        keys.stride(0),
        cache_u8.stride(0),
        cache_bf16.stride(0),
        page_size=page_size,
        head_dim=INT4_C4_POC_HEAD_DIM,
        group_size=INT4_C4_POC_GROUP_SIZE,
        num_groups=INT4_C4_POC_NUM_GROUPS,
        packed_bytes_per_token=INT4_C4_POC_PACKED_BYTES_PER_TOKEN,
        int4_min=_INT4_MIN,
        int4_max=_INT4_MAX,
        num_warps=1,
    )


@triton.jit
def _store_int4_c4_indexer_cache_kernel(
    keys_ptr,
    cache_u8_ptr,
    cache_bf16_ptr,
    locations_ptr,
    keys_stride,
    cache_page_stride_bytes,
    cache_page_stride_bf16,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    packed_bytes_per_token: tl.constexpr,
    int4_min: tl.constexpr,
    int4_max: tl.constexpr,
):
    input_row = tl.program_id(0)
    location = tl.load(locations_ptr + input_row).to(tl.int64)
    valid = location >= 0
    safe_location = tl.where(valid, location, 0)
    page = safe_location // page_size
    in_page = safe_location - page * page_size
    input_base = input_row * keys_stride
    value_base = page * cache_page_stride_bytes + in_page * packed_bytes_per_token
    scale_base = (
        page * cache_page_stride_bf16
        + (page_size * packed_bytes_per_token) // 2
        + in_page * num_groups
    )
    pair_offsets = tl.arange(0, group_size // 2)

    for group_id in tl.static_range(num_groups):
        group_base = input_base + group_id * group_size
        low_values = tl.load(keys_ptr + group_base + pair_offsets * 2).to(tl.float32)
        high_values = tl.load(keys_ptr + group_base + pair_offsets * 2 + 1).to(
            tl.float32
        )
        max_abs = tl.max(tl.maximum(tl.abs(low_values), tl.abs(high_values)))
        scale_fp32 = tl.where(max_abs == 0.0, 1.0, max_abs / int4_max)
        scale_bf16 = scale_fp32.to(tl.bfloat16)
        quant_scale = scale_bf16.to(tl.float32)
        low_scaled = low_values / quant_scale
        high_scaled = high_values / quant_scale
        low_rounded = tl.where(
            low_scaled >= 0.0,
            tl.floor(low_scaled + 0.5),
            tl.ceil(low_scaled - 0.5),
        )
        high_rounded = tl.where(
            high_scaled >= 0.0,
            tl.floor(high_scaled + 0.5),
            tl.ceil(high_scaled - 0.5),
        )
        low_quant = tl.maximum(tl.minimum(low_rounded, int4_max), int4_min).to(tl.int32)
        high_quant = tl.maximum(tl.minimum(high_rounded, int4_max), int4_min).to(
            tl.int32
        )
        packed = (low_quant & 0x0F) | ((high_quant & 0x0F) << 4)
        tl.store(
            cache_u8_ptr + value_base + group_id * (group_size // 2) + pair_offsets,
            packed.to(tl.uint8),
            mask=valid,
        )
        tl.store(
            cache_bf16_ptr + scale_base + group_id,
            scale_bf16,
            mask=valid,
        )


def pack_int4_c4_pages_reference(keys: torch.Tensor) -> torch.Tensor:
    """Quantize complete C4 pages into the portable PoC byte layout.

    This allocation-heavy Torch implementation is a CPU/reference utility,
    not a serving-path cache writer.  Quantization is symmetric per token and
    per 32-dimensional group.  Scales are rounded to BF16 *before* codes are
    chosen so the stored scale is exactly the scale used by the reference.
    """

    expected_tail = (INT4_C4_POC_PAGE_SIZE, INT4_C4_POC_HEAD_DIM)
    if keys.ndim != 3 or tuple(keys.shape[1:]) != expected_tail:
        raise ValueError(
            f"keys must have shape [num_pages, {expected_tail[0]}, "
            f"{expected_tail[1]}], got {tuple(keys.shape)}"
        )
    if not keys.dtype.is_floating_point:
        raise TypeError(f"keys must use a floating-point dtype, got {keys.dtype}")

    num_pages = keys.shape[0]
    grouped = keys.float().reshape(
        num_pages,
        INT4_C4_POC_PAGE_SIZE,
        INT4_C4_POC_NUM_GROUPS,
        INT4_C4_POC_GROUP_SIZE,
    )
    amax = grouped.abs().amax(dim=-1)
    scales = torch.where(amax > 0, amax / _INT4_MAX, torch.ones_like(amax))
    scales_bf16 = scales.to(torch.bfloat16)
    effective_scales = scales_bf16.float().unsqueeze(-1)
    scaled = grouped / effective_scales
    codes = torch.where(
        scaled >= 0,
        torch.floor(scaled + 0.5),
        torch.ceil(scaled - 0.5),
    ).clamp(_INT4_MIN, _INT4_MAX)
    codes = codes.to(torch.int16).reshape(
        num_pages, INT4_C4_POC_PAGE_SIZE, INT4_C4_POC_HEAD_DIM
    )

    low = (codes[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((codes[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = low | high

    cache = torch.empty(
        (num_pages, INT4_C4_POC_PAGE_BYTES),
        dtype=torch.uint8,
        device=keys.device,
    )
    cache[:, :INT4_C4_POC_VALUE_BYTES].copy_(packed.reshape(num_pages, -1))
    scale_bytes = scales_bf16.contiguous().view(torch.uint8).reshape(num_pages, -1)
    cache[:, INT4_C4_POC_VALUE_BYTES:].copy_(scale_bytes)
    return cache


def unpack_int4_c4_pages_reference(cache_u8: torch.Tensor) -> torch.Tensor:
    """Decode portable C4 pages to FP32 for numerical reference tests."""

    _validate_reference_pages(cache_u8)
    num_pages = cache_u8.shape[0]
    packed = cache_u8[:, :INT4_C4_POC_VALUE_BYTES].reshape(
        num_pages,
        INT4_C4_POC_PAGE_SIZE,
        INT4_C4_POC_PACKED_BYTES_PER_TOKEN,
    )
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    low = torch.where(low < 8, low, low - 16)
    high = torch.where(high < 8, high, high - 16)
    codes = torch.stack((low, high), dim=-1).reshape(
        num_pages, INT4_C4_POC_PAGE_SIZE, INT4_C4_POC_HEAD_DIM
    )
    scales = (
        cache_u8[:, INT4_C4_POC_VALUE_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(
            num_pages,
            INT4_C4_POC_PAGE_SIZE,
            INT4_C4_POC_NUM_GROUPS,
        )
        .float()
    )
    decoded = codes.float() * scales.repeat_interleave(INT4_C4_POC_GROUP_SIZE, dim=-1)
    # The hot scorer feeds BF16 keys to HMMA after applying each BF16 scale.
    return decoded.to(torch.bfloat16).float()


def int4_c4_paged_mqa_logits_reference(
    q_bf16: torch.Tensor,
    cache_u8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Allocation-heavy Torch reference for the paged C4 scorer.

    Negative and out-of-range physical page IDs contribute zero.  Sequence
    lengths are clamped to the captured output width, matching the Triton PoC.
    The function works on CPU and is deliberately intended for small tests.
    """

    _validate_reference_pages(cache_u8)
    if q_bf16.ndim != 4:
        raise ValueError(f"query must be rank 4, got {tuple(q_bf16.shape)}")
    batch_size = q_bf16.shape[0]
    expected_query_shape = (
        batch_size,
        1,
        INT4_C4_POC_NUM_HEADS,
        INT4_C4_POC_HEAD_DIM,
    )
    if tuple(q_bf16.shape) != expected_query_shape:
        raise ValueError(
            f"query must have shape {expected_query_shape}, got {tuple(q_bf16.shape)}"
        )
    if tuple(weight.shape) != (batch_size, INT4_C4_POC_NUM_HEADS):
        raise ValueError(
            f"weight must have shape [{batch_size}, {INT4_C4_POC_NUM_HEADS}], "
            f"got {tuple(weight.shape)}"
        )
    if tuple(seq_lens.shape) not in {(batch_size,), (batch_size, 1)}:
        raise ValueError(f"invalid seq_lens shape {tuple(seq_lens.shape)}")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError(f"invalid page_table shape {tuple(page_table.shape)}")
    if max_seq_len <= 0 or max_seq_len > page_table.shape[1] * INT4_C4_POC_PAGE_SIZE:
        raise ValueError("max_seq_len does not fit in the supplied page table")
    tensors = (cache_u8, weight, seq_lens, page_table)
    if any(tensor.device != q_bf16.device for tensor in tensors):
        raise ValueError("all reference tensors must reside on the same device")

    decoded_pages = unpack_int4_c4_pages_reference(cache_u8)
    page_ids = page_table.to(torch.int64)
    valid_pages = (page_ids >= 0) & (page_ids < cache_u8.shape[0])
    safe_page_ids = page_ids.clamp(0, max(cache_u8.shape[0] - 1, 0))
    if cache_u8.shape[0] == 0:
        raise ValueError("cache must contain at least one physical page")
    gathered_keys = decoded_pages[safe_page_ids].reshape(
        batch_size, -1, INT4_C4_POC_HEAD_DIM
    )[:, :max_seq_len]
    valid_tokens = valid_pages.repeat_interleave(INT4_C4_POC_PAGE_SIZE, dim=1)[
        :, :max_seq_len
    ]

    query = q_bf16[:, 0].float()
    logits = torch.bmm(gathered_keys, query.transpose(1, 2))
    weighted = (torch.relu(logits) * weight.float()[:, None, :]).sum(dim=2)
    positions = torch.arange(max_seq_len, device=q_bf16.device)[None, :]
    bounded_lengths = seq_lens.reshape(batch_size, 1).clamp(0, max_seq_len)
    return weighted.masked_fill(~valid_tokens | (positions >= bounded_lengths), 0.0)


@triton.jit
def _int4_c4_paged_mqa_logits_kernel(
    q_ptr,
    q_decode_lut_ptr,
    cache_u8_ptr,
    cache_bf16_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    output_ptr,
    cache_page_stride_bytes: tl.constexpr,
    cache_page_stride_bf16: tl.constexpr,
    page_table_stride: tl.constexpr,
    max_seq_len: tl.constexpr,
    page_table_width: tl.constexpr,
    num_cache_pages: tl.constexpr,
    programs_per_query: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    packed_bytes_per_token: tl.constexpr,
    value_bytes: tl.constexpr,
    query_is_e4m3_bytes: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    program_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), max_seq_len)
    active_pages = tl.minimum(tl.cdiv(bounded_seq_len, page_size), page_table_width)

    # The launch shape is capture-static, while runtime lengths bound the
    # strided page loop. Program p exclusively owns pages p, p + P, ... .
    if program_idx < active_pages:
        head_offsets = tl.arange(0, num_heads)
        dim_offsets = tl.arange(0, head_dim)
        packed_offsets = tl.arange(0, packed_bytes_per_token)
        token_offsets = tl.arange(0, page_size)
        weights = tl.load(weight_ptr + batch_idx * num_heads + head_offsets).to(
            tl.float32
        )
        query_offsets = (
            batch_idx * num_heads * head_dim
            + head_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        query_payload = tl.load(q_ptr + query_offsets)
        if query_is_e4m3_bytes:
            query_indices = query_payload.to(tl.uint32).to(tl.int32)
            query = tl.load(q_decode_lut_ptr + query_indices)
        else:
            query = query_payload

        for page_slot in tl.range(program_idx, active_pages, programs_per_query):
            page_id = tl.load(
                page_table_ptr + batch_idx * page_table_stride + page_slot
            )
            page_is_valid = (page_id >= 0) & (page_id < num_cache_pages)
            safe_page_id = tl.where(page_is_valid, page_id, 0)

            cache_offsets = (
                safe_page_id * cache_page_stride_bytes
                + token_offsets[:, None] * packed_bytes_per_token
                + packed_offsets[None, :]
            )
            packed = tl.load(
                cache_u8_ptr + cache_offsets,
                mask=page_is_valid,
                other=0,
            )
            low_nibbles = (packed & 0x0F).to(tl.int32)
            high_nibbles = ((packed >> 4) & 0x0F).to(tl.int32)
            signed_low = tl.where(low_nibbles < 8, low_nibbles, low_nibbles - 16)
            signed_high = tl.where(high_nibbles < 8, high_nibbles, high_nibbles - 16)
            signed_codes = tl.interleave(signed_low, signed_high)

            scale_base = (
                safe_page_id * cache_page_stride_bf16
                + value_bytes // 2
                + token_offsets * num_groups
            )
            scale0 = tl.load(
                cache_bf16_ptr + scale_base,
                mask=page_is_valid,
                other=0.0,
            ).to(tl.float32)
            scale1 = tl.load(
                cache_bf16_ptr + scale_base + 1,
                mask=page_is_valid,
                other=0.0,
            ).to(tl.float32)
            scale2 = tl.load(
                cache_bf16_ptr + scale_base + 2,
                mask=page_is_valid,
                other=0.0,
            ).to(tl.float32)
            scale3 = tl.load(
                cache_bf16_ptr + scale_base + 3,
                mask=page_is_valid,
                other=0.0,
            ).to(tl.float32)
            group_indices = dim_offsets[None, :] // group_size
            scales = tl.where(
                group_indices == 0,
                scale0[:, None],
                tl.where(
                    group_indices == 1,
                    scale1[:, None],
                    tl.where(
                        group_indices == 2,
                        scale2[:, None],
                        scale3[:, None],
                    ),
                ),
            )
            keys = (signed_codes.to(tl.float32) * scales).to(tl.bfloat16)
            logits = tl.dot(keys, tl.trans(query), out_dtype=tl.float32)

            logits = tl.maximum(logits, 0.0)
            reduced = tl.sum(logits * weights[None, :], axis=1)
            reduced = tl.where(page_is_valid, reduced, 0.0)
            output_positions = page_slot * page_size + token_offsets
            tl.store(
                output_ptr + batch_idx * max_seq_len + output_positions,
                reduced,
                mask=output_positions < bounded_seq_len,
            )


def _programs_per_query(batch_size: int, max_pages: int) -> int:
    if batch_size == 1:
        target = INT4_C4_POC_SINGLE_QUERY_PROGRAMS
    elif batch_size <= INT4_C4_POC_SMALL_BATCH_LIMIT:
        target = INT4_C4_POC_SMALL_BATCH_PROGRAMS_PER_QUERY
    else:
        target = triton.cdiv(INT4_C4_POC_TARGET_PROGRAMS, batch_size)
    return max(1, min(max_pages, target))


def int4_c4_paged_mqa_logits_triton(
    q_bf16: torch.Tensor,
    cache_u8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score signed-INT4 C4 pages without a cache-sized decode workspace.

    The positional arguments mirror the production paged-MQA scorer.  The
    DeepGEMM metadata value is accepted for call-site compatibility and is not
    used.  Before CUDA-graph capture, make one eager call for the same captured
    shapes and supply a caller-owned ``out`` tensor during both calls.
    """

    del deep_gemm_metadata

    if q_bf16.ndim != 4:
        raise ValueError(f"query must be rank 4, got shape {tuple(q_bf16.shape)}")
    batch_size = q_bf16.shape[0]
    if batch_size == 0:
        raise ValueError("query batch must not be empty")
    expected_query_shape = (
        batch_size,
        1,
        INT4_C4_POC_NUM_HEADS,
        INT4_C4_POC_HEAD_DIM,
    )
    supported_query_dtypes = {
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.uint8,
    }
    if (
        q_bf16.dtype not in supported_query_dtypes
        or tuple(q_bf16.shape) != expected_query_shape
    ):
        raise ValueError(
            "query must be BF16 or raw/native E4M3FN bytes with shape "
            f"{expected_query_shape}, got "
            f"{q_bf16.dtype} {tuple(q_bf16.shape)}"
        )
    if not q_bf16.is_contiguous():
        raise ValueError("query must be contiguous")

    _validate_reference_pages(cache_u8)
    if cache_u8.shape[0] == 0:
        raise ValueError("cache must contain at least one physical page")
    cache_page_stride_bytes = cache_u8.stride(0)
    if cache_page_stride_bytes < INT4_C4_POC_PAGE_BYTES:
        raise ValueError("cache page stride is smaller than its packed layout")
    if cache_page_stride_bytes % 2 or cache_u8.storage_offset() % 2:
        raise ValueError("cache pages and storage offset must be BF16-aligned")
    cache_bf16 = cache_u8.view(torch.bfloat16)
    cache_page_stride_bf16 = cache_bf16.stride(0)

    if weight.dtype != torch.float32 or tuple(weight.shape) != (
        batch_size,
        INT4_C4_POC_NUM_HEADS,
    ):
        raise ValueError(
            f"weight must be FP32 [{batch_size}, {INT4_C4_POC_NUM_HEADS}], got "
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
    if max_seq_len > page_table.shape[1] * INT4_C4_POC_PAGE_SIZE:
        raise ValueError("max_seq_len exceeds the supplied page-table capacity")

    device = q_bf16.device
    tensors = (cache_u8, weight, seq_lens, page_table)
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError("all scorer tensors must reside on the same CUDA device")

    if out is None:
        # PyTorch's graph-private allocator gives this tensor a replay-stable
        # address.  Tests and explicit callers can still supply ``out`` to
        # avoid the allocation entirely.
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

    query_is_e4m3_bytes = q_bf16.dtype in {
        torch.float8_e4m3fn,
        torch.uint8,
    }
    if query_is_e4m3_bytes:
        query_payload = (
            q_bf16 if q_bf16.dtype == torch.uint8 else q_bf16.view(torch.uint8)
        )
        query_decode_lut = get_e4m3fn_decode_lut(device)
    else:
        query_payload = q_bf16
        # Unused constexpr branch; passing an existing tensor keeps this
        # wrapper allocation-free under graph capture.
        query_decode_lut = q_bf16

    max_pages = triton.cdiv(max_seq_len, INT4_C4_POC_PAGE_SIZE)
    programs_per_query = _programs_per_query(batch_size, max_pages)
    num_warps = 8 if max_pages <= 64 else 4
    _int4_c4_paged_mqa_logits_kernel[(batch_size, programs_per_query)](
        query_payload,
        query_decode_lut,
        cache_u8,
        cache_bf16,
        weight,
        seq_lens_flat,
        page_table,
        out,
        cache_page_stride_bytes=cache_page_stride_bytes,
        cache_page_stride_bf16=cache_page_stride_bf16,
        page_table_stride=page_table.stride(0),
        max_seq_len=max_seq_len,
        page_table_width=page_table.shape[1],
        num_cache_pages=cache_u8.shape[0],
        programs_per_query=programs_per_query,
        page_size=INT4_C4_POC_PAGE_SIZE,
        num_heads=INT4_C4_POC_NUM_HEADS,
        head_dim=INT4_C4_POC_HEAD_DIM,
        group_size=INT4_C4_POC_GROUP_SIZE,
        num_groups=INT4_C4_POC_NUM_GROUPS,
        packed_bytes_per_token=INT4_C4_POC_PACKED_BYTES_PER_TOKEN,
        value_bytes=INT4_C4_POC_VALUE_BYTES,
        query_is_e4m3_bytes=query_is_e4m3_bytes,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


__all__ = [
    "INT4_C4_BYTES_PER_TOKEN",
    "INT4_C4_POC_GROUP_SIZE",
    "INT4_C4_POC_HEAD_DIM",
    "INT4_C4_POC_NUM_GROUPS",
    "INT4_C4_POC_NUM_HEADS",
    "INT4_C4_POC_PACKED_BYTES_PER_TOKEN",
    "INT4_C4_POC_PAGE_BYTES",
    "INT4_C4_POC_PAGE_SIZE",
    "INT4_C4_POC_SCALE_BYTES",
    "INT4_C4_POC_VALUE_BYTES",
    "int4_c4_page_bytes",
    "int4_c4_paged_mqa_logits_reference",
    "int4_c4_paged_mqa_logits_triton",
    "pack_int4_c4_pages_reference",
    "store_int4_c4_indexer_cache",
    "unpack_int4_c4_pages_reference",
]
