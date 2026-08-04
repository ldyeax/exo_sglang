"""DeepSeek V4 architecture-neutral symmetric-INT4 cache storage.

The 448 no-PE components are stored as architecture-neutral signed INT4
nibbles, while the 64 RoPE components remain exact BF16.  Seven BF16 scales
(one per 64-value group) make the logical layout 366 bytes/token; rows are
padded to 368 bytes for 16-byte alignment.  Production paged caches use the
same token-major row inside every page, so SWA, C4, and C128 share one layout.

An RTX 3090 supports integer INT4 Tensor Core operations, but these kernels use
INT4 as a *storage format* and decode to BF16.  They are therefore neither a
Tensor Core INT4 attention implementation nor OSCAR: OSCAR additionally needs
offline, model-specific rotations and calibration.  No uncalibrated rotation
is applied by the hot path.  A reference-only 64-wide normalized Hadamard
helper is included to test the orthogonal-transform building block used by
SAW-INT4.  SAW's fixed Hadamard transform does not require offline fitting,
but integrating it into DSV4 still requires an inverse output transform (or
an exactly absorbed ``wo_a`` transform) plus model-specific quality gates.

All hot APIs accept caller-owned destinations.  Passing those destinations is
required during CUDA-graph capture so capture cannot hide a dynamic allocation.
The raw nibble representation itself has no native ``torch.int4`` dependency.

Per-token byte layout::

    [0, 224)    448 signed INT4 no-PE values, even element in low nibble
    [224, 352)   64 BF16 RoPE values
    [352, 366)    7 BF16 symmetric scales
    [366, 368)    2 alignment bytes, never written by the kernels
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
GROUP_SIZE = 64
NUM_GROUPS = NOPE_DIM // GROUP_SIZE
INT4_VALUES_PER_BYTE = 2
PACKED_NOPE_BYTES = NOPE_DIM // INT4_VALUES_PER_BYTE
ROPE_BYTES = ROPE_DIM * torch.bfloat16.itemsize
SCALE_BYTES = NUM_GROUPS * torch.bfloat16.itemsize
ROPE_OFFSET_BYTES = PACKED_NOPE_BYTES
SCALE_OFFSET_BYTES = ROPE_OFFSET_BYTES + ROPE_BYTES
LOGICAL_BYTES_PER_TOKEN = SCALE_OFFSET_BYTES + SCALE_BYTES
STORAGE_BYTES_PER_TOKEN = 368
PADDING_BYTES_PER_TOKEN = STORAGE_BYTES_PER_TOKEN - LOGICAL_BYTES_PER_TOKEN

INT4_MIN = -7
INT4_MAX = 7


def int4_main_page_bytes(page_size: int) -> int:
    """Return exact bytes in one token-major INT4 main-cache page."""
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError(f"page_size must be a positive integer, got {page_size!r}")
    return page_size * STORAGE_BYTES_PER_TOKEN


def _round_half_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.where(
        values >= 0,
        torch.floor(values + 0.5),
        torch.ceil(values - 0.5),
    )


def apply_block_hadamard_nope_reference(values: torch.Tensor) -> torch.Tensor:
    """Apply normalized 64-wide Hadamard blocks to the no-PE dimensions.

    This allocation-heavy Torch implementation is a numerical reference, not
    a serving kernel.  ``values`` may end in either 448 no-PE elements or the
    complete 512-element DeepSeek V4 key/query geometry.  In the latter case,
    the 64 RoPE dimensions are copied without modification.
    """

    if values.shape[-1] not in (NOPE_DIM, HEAD_DIM):
        raise ValueError(
            f"the last dimension must be {NOPE_DIM} or {HEAD_DIM}, "
            f"got {values.shape[-1]}"
        )

    leading_shape = values.shape[:-1]
    transformed = (
        values[..., :NOPE_DIM].float().reshape(*leading_shape, NUM_GROUPS, GROUP_SIZE)
    )
    butterfly_width = 1
    while butterfly_width < GROUP_SIZE:
        transformed = transformed.reshape(
            *leading_shape,
            NUM_GROUPS,
            -1,
            2,
            butterfly_width,
        )
        low = transformed[..., 0, :]
        high = transformed[..., 1, :]
        transformed = torch.cat((low + high, low - high), dim=-1)
        butterfly_width *= 2
    transformed = transformed.reshape(*leading_shape, NOPE_DIM) / (GROUP_SIZE**0.5)
    if values.shape[-1] == NOPE_DIM:
        return transformed
    return torch.cat((transformed, values[..., NOPE_DIM:].float()), dim=-1)


def _validate_values(values: torch.Tensor) -> None:
    if not values.is_cuda:
        raise ValueError("the Triton INT4 storage writer requires a CUDA tensor")
    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(
            f"values must have shape (num_tokens, {HEAD_DIM}), got {values.shape}"
        )
    if values.dtype != torch.bfloat16:
        raise ValueError(f"values must be BF16, got {values.dtype}")
    if values.stride(1) != 1:
        raise ValueError("values must be contiguous along the head dimension")


def _validate_storage(storage: torch.Tensor) -> None:
    if not storage.is_cuda:
        raise ValueError("INT4 storage must be a CUDA tensor")
    if storage.ndim != 2 or storage.shape[1] != STORAGE_BYTES_PER_TOKEN:
        raise ValueError(
            "storage must have shape "
            f"(capacity, {STORAGE_BYTES_PER_TOKEN}), got {storage.shape}"
        )
    if storage.dtype != torch.uint8:
        raise ValueError(f"storage must use raw uint8 bytes, got {storage.dtype}")
    if not storage.is_contiguous():
        raise ValueError("storage must be contiguous")


def _validate_paged_storage(storage: torch.Tensor, page_size: int) -> None:
    if not storage.is_cuda:
        raise ValueError("INT4 paged storage must be a CUDA tensor")
    expected_page_bytes = int4_main_page_bytes(page_size)
    if storage.ndim != 2 or storage.shape[1] != expected_page_bytes:
        raise ValueError(
            "paged storage must have shape "
            f"(num_pages, {expected_page_bytes}), got {tuple(storage.shape)}"
        )
    if storage.dtype != torch.uint8:
        raise ValueError(f"paged storage must use raw uint8 bytes, got {storage.dtype}")
    if storage.stride(1) != 1:
        raise ValueError("paged storage rows must have contiguous byte storage")
    if storage.stride(0) < expected_page_bytes:
        raise ValueError("paged storage row stride is smaller than its INT4 layout")
    if storage.stride(0) % 2 or storage.storage_offset() % 2:
        raise ValueError("paged storage must be BF16 aligned")


def _validate_locations(
    locations: torch.Tensor | None,
    expected_size: int | None = None,
) -> None:
    if locations is None:
        return
    if not locations.is_cuda or locations.ndim != 1:
        raise ValueError("locations must be a one-dimensional CUDA tensor")
    if locations.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"locations must be int32 or int64, got {locations.dtype}")
    if expected_size is not None and locations.numel() != expected_size:
        raise ValueError(f"expected {expected_size} locations, got {locations.numel()}")


def _validate_output(output: torch.Tensor, num_tokens: int) -> None:
    if not output.is_cuda:
        raise ValueError("decoded output must be a CUDA tensor")
    if output.shape != (num_tokens, HEAD_DIM):
        raise ValueError(
            f"output must have shape ({num_tokens}, {HEAD_DIM}), got {output.shape}"
        )
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output must be BF16, got {output.dtype}")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")


def quantize_dsv4_int4_storage(
    values: torch.Tensor,
    *,
    storage: torch.Tensor | None = None,
    locations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize BF16 DeepSeek V4 keys into raw symmetric-INT4 storage.

    ``locations`` optionally scatters each input row into a caller-owned cache.
    A negative scatter location is treated as padding and skipped.  When no
    storage is supplied, a zero-initialized, tightly sized cache is allocated;
    that convenience form is forbidden during CUDA-graph capture.
    """

    _validate_values(values)
    _validate_locations(locations, values.shape[0])
    if locations is not None and locations.device != values.device:
        raise ValueError("values and locations must be on the same CUDA device")

    if storage is None:
        if locations is not None:
            raise ValueError("scatter locations require caller-owned storage")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "preallocated INT4 storage is required during CUDA-graph capture"
            )
        storage = torch.zeros(
            (values.shape[0], STORAGE_BYTES_PER_TOKEN),
            dtype=torch.uint8,
            device=values.device,
        )
    _validate_storage(storage)
    if storage.device != values.device:
        raise ValueError("values and storage must be on the same CUDA device")
    if locations is None and storage.shape[0] < values.shape[0]:
        raise ValueError("storage capacity is smaller than the input batch")

    if values.shape[0] == 0:
        return storage

    # Passing an alias is allocation-free and lets Triton issue typed BF16
    # stores into the RoPE/scale regions of the raw uint8 backing allocation.
    storage_bf16 = storage.view(torch.bfloat16)
    locations_ptr = locations if locations is not None else storage
    _quantize_dsv4_int4_storage_kernel[(values.shape[0],)](
        values,
        storage,
        storage_bf16,
        locations_ptr,
        values.stride(0),
        storage.stride(0),
        storage_bf16.stride(0),
        use_locations=locations is not None,
        group_size=GROUP_SIZE,
        num_groups=NUM_GROUPS,
        int4_values_per_byte=INT4_VALUES_PER_BYTE,
        int4_min=INT4_MIN,
        int4_max=INT4_MAX,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        rope_offset_bytes=ROPE_OFFSET_BYTES,
        scale_offset_bytes=SCALE_OFFSET_BYTES,
        num_warps=1,
    )
    return storage


def dequantize_dsv4_int4_storage(
    storage: torch.Tensor,
    *,
    locations: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather and decode raw INT4 cache rows to BF16.

    Supplying ``locations`` decodes only those cache rows, in that order.  The
    fused decoder unpacks nibbles, sign-extends, applies each BF16 scale, and
    copies the BF16 RoPE tail in one Triton program per output row.
    """

    _validate_storage(storage)
    _validate_locations(locations)
    if locations is not None and locations.device != storage.device:
        raise ValueError("storage and locations must be on the same CUDA device")
    num_tokens = storage.shape[0] if locations is None else locations.numel()

    if output is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "preallocated INT4 decode output is required during CUDA-graph capture"
            )
        output = torch.empty(
            (num_tokens, HEAD_DIM),
            dtype=torch.bfloat16,
            device=storage.device,
        )
    _validate_output(output, num_tokens)
    if output.device != storage.device:
        raise ValueError("storage and output must be on the same CUDA device")

    if num_tokens == 0:
        return output

    storage_bf16 = storage.view(torch.bfloat16)
    locations_ptr = locations if locations is not None else storage
    _dequantize_dsv4_int4_storage_kernel[(num_tokens,)](
        storage,
        storage_bf16,
        locations_ptr,
        output,
        storage.stride(0),
        storage_bf16.stride(0),
        output.stride(0),
        use_locations=locations is not None,
        group_size=GROUP_SIZE,
        num_groups=NUM_GROUPS,
        int4_values_per_byte=INT4_VALUES_PER_BYTE,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        rope_offset_bytes=ROPE_OFFSET_BYTES,
        scale_offset_bytes=SCALE_OFFSET_BYTES,
        num_warps=1,
    )
    return output


def quantize_dsv4_int4_cache_paged(
    values: torch.Tensor,
    storage: torch.Tensor,
    locations: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Quantize and scatter BF16 keys into production token-major pages.

    ``locations`` contains logical token slots.  Negative entries are padding
    and do not modify the cache.  The wrapper allocates nothing and is safe to
    replay from a CUDA graph after its Triton specialization is primed.
    """

    _validate_values(values)
    _validate_locations(locations, values.shape[0])
    _validate_paged_storage(storage, page_size)
    if values.device != storage.device or locations.device != storage.device:
        raise ValueError("values, storage, and locations must share one CUDA device")
    if values.shape[0] == 0:
        return

    storage_bf16 = storage.view(torch.bfloat16)
    _quantize_dsv4_int4_cache_paged_kernel[(values.shape[0],)](
        values,
        storage,
        storage_bf16,
        locations,
        values.stride(0),
        storage.stride(0),
        storage_bf16.stride(0),
        page_size=page_size,
        group_size=GROUP_SIZE,
        num_groups=NUM_GROUPS,
        int4_values_per_byte=INT4_VALUES_PER_BYTE,
        int4_min=INT4_MIN,
        int4_max=INT4_MAX,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        rope_offset_bytes=ROPE_OFFSET_BYTES,
        scale_offset_bytes=SCALE_OFFSET_BYTES,
        token_bytes=STORAGE_BYTES_PER_TOKEN,
        num_warps=1,
    )


@triton.jit
def _quantize_dsv4_int4_storage_kernel(
    values_ptr,
    storage_u8_ptr,
    storage_bf16_ptr,
    locations_ptr,
    values_stride,
    storage_u8_stride,
    storage_bf16_stride,
    use_locations: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    int4_values_per_byte: tl.constexpr,
    int4_min: tl.constexpr,
    int4_max: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rope_offset_bytes: tl.constexpr,
    scale_offset_bytes: tl.constexpr,
):
    input_row = tl.program_id(0)
    output_row = (
        tl.load(locations_ptr + input_row).to(tl.int64) if use_locations else input_row
    )
    valid = output_row >= 0
    value_base = input_row * values_stride
    storage_u8_base = output_row * storage_u8_stride
    storage_bf16_base = output_row * storage_bf16_stride
    pair_offsets = tl.arange(0, group_size // int4_values_per_byte)

    for group_id in tl.static_range(num_groups):
        group_base = value_base + group_id * group_size
        low_values = tl.load(values_ptr + group_base + pair_offsets * 2).to(tl.float32)
        high_values = tl.load(values_ptr + group_base + pair_offsets * 2 + 1).to(
            tl.float32
        )
        max_abs = tl.max(tl.maximum(tl.abs(low_values), tl.abs(high_values)))
        scale_fp32 = tl.where(max_abs == 0.0, 1.0, max_abs / int4_max)
        # Quantize with the scale that is actually persisted.  BF16 rounding
        # can otherwise make the encoder and decoder disagree at a bin edge.
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

        packed_base = group_id * (group_size // int4_values_per_byte)
        tl.store(
            storage_u8_ptr + storage_u8_base + packed_base + pair_offsets,
            packed.to(tl.uint8),
            mask=valid,
        )
        scale_offset_bf16 = scale_offset_bytes // 2 + group_id
        tl.store(
            storage_bf16_ptr + storage_bf16_base + scale_offset_bf16,
            scale_bf16,
            mask=valid,
        )

    rope_offsets = tl.arange(0, rope_dim)
    rope = tl.load(values_ptr + value_base + nope_dim + rope_offsets)
    tl.store(
        storage_bf16_ptr + storage_bf16_base + rope_offset_bytes // 2 + rope_offsets,
        rope,
        mask=valid,
    )


@triton.jit
def _dequantize_dsv4_int4_storage_kernel(
    storage_u8_ptr,
    storage_bf16_ptr,
    locations_ptr,
    output_ptr,
    storage_u8_stride,
    storage_bf16_stride,
    output_stride,
    use_locations: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    int4_values_per_byte: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rope_offset_bytes: tl.constexpr,
    scale_offset_bytes: tl.constexpr,
):
    output_row = tl.program_id(0)
    storage_row = (
        tl.load(locations_ptr + output_row).to(tl.int64)
        if use_locations
        else output_row
    )
    storage_u8_base = storage_row * storage_u8_stride
    storage_bf16_base = storage_row * storage_bf16_stride
    output_base = output_row * output_stride
    pair_offsets = tl.arange(0, group_size // int4_values_per_byte)

    for group_id in tl.static_range(num_groups):
        packed_base = group_id * (group_size // int4_values_per_byte)
        packed = tl.load(
            storage_u8_ptr + storage_u8_base + packed_base + pair_offsets
        ).to(tl.int32)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        low = tl.where(low >= 8, low - 16, low).to(tl.float32)
        high = tl.where(high >= 8, high - 16, high).to(tl.float32)
        scale_offset_bf16 = scale_offset_bytes // 2 + group_id
        scale = tl.load(storage_bf16_ptr + storage_bf16_base + scale_offset_bf16).to(
            tl.float32
        )

        group_base = output_base + group_id * group_size
        tl.store(
            output_ptr + group_base + pair_offsets * 2,
            (low * scale).to(output_ptr.dtype.element_ty),
        )
        tl.store(
            output_ptr + group_base + pair_offsets * 2 + 1,
            (high * scale).to(output_ptr.dtype.element_ty),
        )

    rope_offsets = tl.arange(0, rope_dim)
    rope = tl.load(
        storage_bf16_ptr + storage_bf16_base + rope_offset_bytes // 2 + rope_offsets
    )
    tl.store(output_ptr + output_base + nope_dim + rope_offsets, rope)


@triton.jit
def _quantize_dsv4_int4_cache_paged_kernel(
    values_ptr,
    storage_u8_ptr,
    storage_bf16_ptr,
    locations_ptr,
    values_stride,
    storage_u8_page_stride,
    storage_bf16_page_stride,
    page_size: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    int4_values_per_byte: tl.constexpr,
    int4_min: tl.constexpr,
    int4_max: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rope_offset_bytes: tl.constexpr,
    scale_offset_bytes: tl.constexpr,
    token_bytes: tl.constexpr,
):
    input_row = tl.program_id(0)
    location = tl.load(locations_ptr + input_row).to(tl.int64)
    valid = location >= 0
    safe_location = tl.where(valid, location, 0)
    page = safe_location // page_size
    in_page = safe_location - page * page_size
    value_base = input_row * values_stride
    storage_u8_base = page * storage_u8_page_stride + in_page * token_bytes
    storage_bf16_base = page * storage_bf16_page_stride + in_page * (token_bytes // 2)
    pair_offsets = tl.arange(0, group_size // int4_values_per_byte)

    for group_id in tl.static_range(num_groups):
        group_base = value_base + group_id * group_size
        low_values = tl.load(values_ptr + group_base + pair_offsets * 2).to(tl.float32)
        high_values = tl.load(values_ptr + group_base + pair_offsets * 2 + 1).to(
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
        packed_base = group_id * (group_size // int4_values_per_byte)
        tl.store(
            storage_u8_ptr + storage_u8_base + packed_base + pair_offsets,
            packed.to(tl.uint8),
            mask=valid,
        )
        tl.store(
            storage_bf16_ptr + storage_bf16_base + scale_offset_bytes // 2 + group_id,
            scale_bf16,
            mask=valid,
        )

    rope_offsets = tl.arange(0, rope_dim)
    rope = tl.load(values_ptr + value_base + nope_dim + rope_offsets)
    tl.store(
        storage_bf16_ptr + storage_bf16_base + rope_offset_bytes // 2 + rope_offsets,
        rope,
        mask=valid,
    )


def quantize_dsv4_int4_reference(values: torch.Tensor) -> torch.Tensor:
    """Portable Torch reference encoder for parity tests and calibration work."""

    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(
            f"values must have shape (num_tokens, {HEAD_DIM}), got {values.shape}"
        )
    if values.dtype != torch.bfloat16:
        raise ValueError(f"values must be BF16, got {values.dtype}")

    num_tokens = values.shape[0]
    no_pe = values[:, :NOPE_DIM].float().reshape(num_tokens, NUM_GROUPS, GROUP_SIZE)
    max_abs = no_pe.abs().amax(dim=-1)
    scales = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / INT4_MAX)
    scales_bf16 = scales.to(torch.bfloat16)
    scaled = no_pe / scales_bf16.float().unsqueeze(-1)
    quantized = _round_half_away_from_zero(scaled).clamp(INT4_MIN, INT4_MAX)
    quantized = quantized.to(torch.int8)
    low = quantized[..., 0::2].to(torch.int16) & 0x0F
    high = quantized[..., 1::2].to(torch.int16) & 0x0F
    packed = (low | (high << 4)).to(torch.uint8).reshape(num_tokens, -1)

    storage = torch.zeros(
        (num_tokens, STORAGE_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=values.device,
    )
    storage[:, :PACKED_NOPE_BYTES].copy_(packed)
    storage[:, ROPE_OFFSET_BYTES:SCALE_OFFSET_BYTES].view(torch.bfloat16).copy_(
        values[:, NOPE_DIM:]
    )
    storage[:, SCALE_OFFSET_BYTES:LOGICAL_BYTES_PER_TOKEN].view(torch.bfloat16).copy_(
        scales_bf16
    )
    return storage


def dequantize_dsv4_int4_reference(storage: torch.Tensor) -> torch.Tensor:
    """Portable Torch reference decoder for contiguous cache rows."""

    if storage.ndim != 2 or storage.shape[1] != STORAGE_BYTES_PER_TOKEN:
        raise ValueError(
            "storage must have shape "
            f"(num_tokens, {STORAGE_BYTES_PER_TOKEN}), got {storage.shape}"
        )
    if storage.dtype != torch.uint8:
        raise ValueError(f"storage must use raw uint8 bytes, got {storage.dtype}")

    num_tokens = storage.shape[0]
    packed = storage[:, :PACKED_NOPE_BYTES].reshape(
        num_tokens, NUM_GROUPS, GROUP_SIZE // INT4_VALUES_PER_BYTE
    )
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    quantized = torch.empty(
        (num_tokens, NUM_GROUPS, GROUP_SIZE),
        dtype=torch.int8,
        device=storage.device,
    )
    quantized[..., 0::2] = low
    quantized[..., 1::2] = high
    scales = storage[:, SCALE_OFFSET_BYTES:LOGICAL_BYTES_PER_TOKEN].view(torch.bfloat16)

    output = torch.empty(
        (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=storage.device
    )
    output[:, :NOPE_DIM] = (
        (quantized.float() * scales.float().unsqueeze(-1))
        .reshape(num_tokens, NOPE_DIM)
        .to(torch.bfloat16)
    )
    output[:, NOPE_DIM:] = storage[:, ROPE_OFFSET_BYTES:SCALE_OFFSET_BYTES].view(
        torch.bfloat16
    )
    return output
