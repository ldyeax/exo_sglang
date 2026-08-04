from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    get_e4m3fn_decode_lut,
)
from sglang.kernels.ops.attention.dsv4.int4_storage import (
    GROUP_SIZE as INT4_GROUP_SIZE,
    NUM_GROUPS as INT4_NUM_GROUPS,
    ROPE_OFFSET_BYTES as INT4_ROPE_OFFSET_BYTES,
    SCALE_OFFSET_BYTES as INT4_SCALE_OFFSET_BYTES,
    STORAGE_BYTES_PER_TOKEN as INT4_STORAGE_BYTES_PER_TOKEN,
)
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.mem_cache.dsv4_kv_cache_dtype import (
    dsv4_uses_ampere_fp8_kv_storage,
)

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# v4 KV cache layout (see dsv4.index_buf_accessor._set_k_and_s_triton_kernel):
#   per-token: 448 fp8 nope + 64 bf16 rope (= 576 contiguous bytes) +
#              7 ue8m0 scales padded to 8 bytes.
#   per-page:  [token 0..P-1 nope+rope (P*576 bytes)] [token 0..P-1 scale (P*8 bytes)]
#              padded up to a multiple of 576.
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64  # one nope scale tile = 64 fp8 values
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE  # 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2  # 576
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1  # 8


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
    is_bf16: bool = False,
    is_int4: bool = False,
) -> torch.Tensor:
    """Dequantize the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: (num_pages, bytes_per_page_padded) uint8.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.
            May be a slice of a larger workspace; the kernel uses out.stride(0)
            so contiguous-along-dim-0 slices work.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    if is_bf16 and is_int4:
        raise ValueError("BF16 and INT4 DSV4 cache layouts are mutually exclusive")

    # The buffer's dtype is whatever the pool exposes (often bf16); the
    # underlying storage is uint8. Reinterpret to byte-space first.
    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    if is_bf16:
        _gather_bf16_k_cache_paged_kernel[(num_tokens,)](
            out,
            quant_k_cache_u8.view(torch.bfloat16).reshape(-1),
            page_table_1_flattened,
            out.stride(0),
            BF16_PER_PAGE=bytes_per_page // 2,
            PAGE_SIZE=page_size,
            TOKEN_ELEMS=DIM_NOPE + DIM_ROPE,
            BLOCK_SIZE=512,
        )
        return out

    if is_int4:
        expected_page_bytes = page_size * INT4_STORAGE_BYTES_PER_TOKEN
        if bytes_per_page != expected_page_bytes:
            raise ValueError(
                f"INT4 page has {bytes_per_page} bytes, expected {expected_page_bytes}"
            )
        _dequantize_k_cache_paged_int4_kernel[(num_tokens,)](
            out,
            quant_k_cache_u8.reshape(-1),
            quant_k_cache_u8.view(torch.bfloat16).reshape(-1),
            page_table_1_flattened,
            out.stride(0),
            BYTES_PER_PAGE=bytes_per_page,
            PAGE_SIZE=page_size,
            DIM_NOPE=DIM_NOPE,
            DIM_ROPE=DIM_ROPE,
            GROUP_SIZE=INT4_GROUP_SIZE,
            NUM_GROUPS=INT4_NUM_GROUPS,
            TOKEN_BYTES=INT4_STORAGE_BYTES_PER_TOKEN,
            ROPE_OFFSET_BYTES=INT4_ROPE_OFFSET_BYTES,
            SCALE_OFFSET_BYTES=INT4_SCALE_OFFSET_BYTES,
        )
        return out

    device_capability = torch.cuda.get_device_capability(quant_k_cache.device)
    if dsv4_uses_ampere_fp8_kv_storage(device_capability):
        lut = get_e4m3fn_decode_lut(quant_k_cache.device)
        _dequantize_k_cache_paged_byte_kernel[(num_tokens,)](
            out,
            quant_k_cache_u8.reshape(-1),
            quant_k_cache_u8.view(torch.bfloat16).reshape(-1),
            page_table_1_flattened,
            lut,
            out.stride(0),
            BYTES_PER_PAGE=bytes_per_page,
            PAGE_SIZE=page_size,
            DIM_NOPE=DIM_NOPE,
            DIM_ROPE=DIM_ROPE,
            TILE_SIZE=TILE_SIZE,
            NUM_SCALE_TILES=NUM_SCALE_TILES,
            NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
            PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
            S_OFFSET_BYTES=s_offset_bytes,
        )
        return out

    # Three typed views over the same underlying bytes.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    _dequantize_k_cache_paged_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


@triton.jit
def _gather_bf16_k_cache_paged_kernel(
    output_ptr,
    buf_bf16_ptr,
    page_table_ptr,
    output_stride_0,
    BF16_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOKEN_ELEMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather unquantized BF16 cache rows without introducing FP8 IR types."""
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    src_base = page_idx * BF16_PER_PAGE + in_page * TOKEN_ELEMS
    dst_base = token_id * output_stride_0
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(buf_bf16_ptr + src_base + offsets)
    tl.store(output_ptr + dst_base + offsets, values)


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


@triton.jit
def _dequantize_k_cache_paged_byte_kernel(
    output_ptr,
    buf_uint8_ptr,
    buf_bf16_ptr,
    page_table_ptr,
    lut_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    """SM86 byte-storage decoder; contains no native FP8 Triton IR."""
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        raw = tl.load(buf_uint8_ptr + fp8_off)
        lut_index = raw.to(tl.uint32).to(tl.int32)
        fp8_vals = tl.load(lut_ptr + lut_index).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


@triton.jit
def _dequantize_k_cache_paged_int4_kernel(
    output_ptr,
    buf_uint8_ptr,
    buf_bf16_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    ROPE_OFFSET_BYTES: tl.constexpr,
    SCALE_OFFSET_BYTES: tl.constexpr,
):
    """Gather signed-INT4 keys directly into a caller-owned BF16 workspace."""
    token_id = tl.program_id(0)
    location = tl.load(page_table_ptr + token_id).to(tl.int64)
    page = location // PAGE_SIZE
    in_page = location - page * PAGE_SIZE
    token_base = page * BYTES_PER_PAGE + in_page * TOKEN_BYTES
    output_base = token_id * output_stride_0
    pair_offsets = tl.arange(0, GROUP_SIZE // 2)

    for group_id in tl.static_range(NUM_GROUPS):
        packed = tl.load(
            buf_uint8_ptr
            + token_base
            + group_id * (GROUP_SIZE // 2)
            + pair_offsets
        ).to(tl.int32)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        low = tl.where(low < 8, low, low - 16).to(tl.float32)
        high = tl.where(high < 8, high, high - 16).to(tl.float32)
        scale = tl.load(
            buf_bf16_ptr + (token_base + SCALE_OFFSET_BYTES) // 2 + group_id
        ).to(tl.float32)
        group_base = output_base + group_id * GROUP_SIZE
        tl.store(
            output_ptr + group_base + pair_offsets * 2,
            (low * scale).to(tl.bfloat16),
        )
        tl.store(
            output_ptr + group_base + pair_offsets * 2 + 1,
            (high * scale).to(tl.bfloat16),
        )

    rope_offsets = tl.arange(0, DIM_ROPE)
    rope = tl.load(
        buf_bf16_ptr + (token_base + ROPE_OFFSET_BYTES) // 2 + rope_offsets
    )
    tl.store(output_ptr + output_base + DIM_NOPE + rope_offsets, rope)


def dequantize_k_cache_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch reference for :func:`dequantize_k_cache_paged`.

    Decodes the same v4 paged layout with vectorized torch indexing instead of
    a Triton kernel. Used to validate the kernel (see the ``__main__`` block
    below); not on any hot path.
    """
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    flat_u8 = u8.reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
    )

    device = quant_k_cache.device
    nope_byte = (
        token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    )
    decode_lut = get_e4m3fn_decode_lut(quant_k_cache.device)
    nope_fp8 = decode_lut[flat_u8[nope_byte].long()].to(torch.float32)
    scale_byte = (
        token_scale_base[:, None]
        + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
    )
    scale_u8 = flat_u8[scale_byte].to(torch.int32)
    scale_pow2 = torch.exp2((scale_u8 - 127).to(torch.float32))
    scale_pow2 = torch.where(
        scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
    )
    scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
    nope = nope_fp8 * scale_full

    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope_idx = rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty(
        (loc.shape[0], 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=device,
    )
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this self-test needs a CUDA device"
    torch.manual_seed(0)
    device = "cuda"

    page_size = 64
    num_pages = 8
    num_tokens = 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES

    quant_k_cache = torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )
    page_table = torch.randint(
        0, num_pages * page_size, (num_tokens,), dtype=torch.int32, device=device
    )

    out_kernel = dequantize_k_cache_paged(quant_k_cache, page_table, page_size)
    out_ref = dequantize_k_cache_paged_ref(quant_k_cache, page_table, page_size)

    torch.testing.assert_close(out_kernel, out_ref, atol=0, rtol=0, equal_nan=True)
    print(
        f"OK: kernel matches torch ref for {num_tokens} tokens "
        f"(page_size={page_size}, bytes_per_page={bytes_per_page})"
    )
