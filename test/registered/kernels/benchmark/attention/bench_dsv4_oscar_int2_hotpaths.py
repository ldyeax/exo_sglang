"""CUDA-event benchmark for the DSV4 OSCAR masked writers and C4 scorer.

The ``legacy_*`` kernels freeze the immediately preceding behavior: masked
writer rows still execute rotation/sort/pack work, and each C4 page program
recomputes ``query @ R``.  The production calls exercise device-uniform live
mask bypass and the caller-owned once-per-query rotation workspace.  Timings
are CUDA-graph replay medians so Python-side validation differences are outside
the measured region, as they are in production decode graphs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4 import oscar_int2_c4_indexer as c4
from sglang.kernels.ops.attention.dsv4 import oscar_int2_storage as shared


@triton.jit
def _legacy_shared_masked_writer_kernel(
    values_ptr,
    rotation_ptr,
    clip_indices_ptr,
    storage_u8_ptr,
    storage_bf16_ptr,
    locations_ptr,
    write_mask_ptr,
    num_tokens,
    capacity,
    values_stride,
    rotation_stride_in,
    rotation_stride_out,
    storage_u8_page_stride,
    storage_bf16_page_stride,
    page_size: tl.constexpr,
    token_bytes: tl.constexpr,
    nope_dim: tl.constexpr,
    rotation_input_tile: tl.constexpr,
    rope_dim: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    rope_offset_bytes: tl.constexpr,
    scale_zero_offset_bytes: tl.constexpr,
    block_tokens: tl.constexpr,
    int2_max: tl.constexpr,
):
    token_block = tl.program_id(0)
    group_id = tl.program_id(1)
    token_offsets = token_block * block_tokens + tl.arange(0, block_tokens)
    token_mask = token_offsets < num_tokens
    locations = tl.load(locations_ptr + token_offsets, mask=token_mask, other=-1).to(
        tl.int64
    )
    requested = tl.load(write_mask_ptr + token_offsets, mask=token_mask, other=0).to(
        tl.int1
    )
    active = token_mask & requested & (locations >= 0) & (locations < capacity)
    safe_locations = tl.where(active, locations, 0)

    # Frozen pre-optimization behavior: these loads, HMMA, sort, and packing
    # execute even when every live-mask bit is false.
    input_offsets = tl.arange(0, rotation_input_tile)
    group_offsets = tl.arange(0, group_size)
    input_tile = tl.load(
        values_ptr + token_offsets[:, None] * values_stride + input_offsets[None, :],
        mask=token_mask[:, None] & (input_offsets[None, :] < nope_dim),
        other=0.0,
    )
    rotation_tile = tl.load(
        rotation_ptr
        + input_offsets[:, None] * rotation_stride_in
        + (group_id * group_size + group_offsets[None, :]) * rotation_stride_out,
        mask=input_offsets[:, None] < nope_dim,
        other=0.0,
    )
    rotated = tl.dot(input_tile, rotation_tile, out_dtype=tl.float32)
    sorted_absolute = tl.sort(tl.abs(rotated), dim=1)
    clip_index = tl.load(clip_indices_ptr + group_id).to(tl.int32)
    selected = group_offsets[None, :] == clip_index
    threshold = tl.sum(tl.where(selected, sorted_absolute, 0.0), axis=1)
    clipped = tl.minimum(tl.maximum(rotated, -threshold[:, None]), threshold[:, None])
    minimum = tl.min(clipped, axis=1)
    maximum = tl.max(clipped, axis=1)
    scale_fp32 = tl.where(maximum == minimum, 1.0, (maximum - minimum) / int2_max)
    scale_bf16 = scale_fp32.to(tl.bfloat16)
    persisted_scale = scale_bf16.to(tl.float32)
    zero_bf16 = (-minimum / persisted_scale).to(tl.bfloat16)
    persisted_zero = zero_bf16.to(tl.float32)
    codes = tl.floor(clipped / persisted_scale[:, None] + persisted_zero[:, None] + 0.5)
    codes = tl.maximum(tl.minimum(codes, int2_max), 0.0).to(tl.uint8)
    shaped = tl.reshape(codes, (block_tokens, packed_group_bytes, 2, 2))
    even, odd = tl.split(shaped)
    code0, code2 = tl.split(even)
    code1, code3 = tl.split(odd)
    packed = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)

    page = safe_locations // page_size
    in_page = safe_locations - page * page_size
    storage_u8_base = page * storage_u8_page_stride + in_page * token_bytes
    storage_bf16_base = page * storage_bf16_page_stride + in_page * (token_bytes // 2)
    byte_offsets = tl.arange(0, packed_group_bytes)
    tl.store(
        storage_u8_ptr
        + storage_u8_base[:, None]
        + group_id * packed_group_bytes
        + byte_offsets[None, :],
        packed,
        mask=active[:, None],
    )
    metadata_base = scale_zero_offset_bytes // 2 + group_id * 2
    tl.store(
        storage_bf16_ptr + storage_bf16_base + metadata_base,
        scale_bf16,
        mask=active,
    )
    tl.store(
        storage_bf16_ptr + storage_bf16_base + metadata_base + 1,
        zero_bf16,
        mask=active,
    )
    rope_offsets = tl.arange(0, rope_dim)
    rope = tl.load(
        values_ptr
        + token_offsets[:, None] * values_stride
        + nope_dim
        + rope_offsets[None, :],
        mask=token_mask[:, None] & (group_id == 0),
        other=0.0,
    )
    tl.store(
        storage_bf16_ptr
        + storage_bf16_base[:, None]
        + rope_offset_bytes // 2
        + rope_offsets[None, :],
        rope,
        mask=active[:, None] & (group_id == 0),
    )


@triton.jit
def _legacy_c4_masked_writer_kernel(
    keys_ptr,
    rotation_ptr,
    storage_u8_ptr,
    storage_f32_ptr,
    locations_ptr,
    write_mask_ptr,
    num_tokens,
    capacity,
    keys_stride,
    rotation_stride_in,
    rotation_stride_out,
    storage_u8_page_stride,
    storage_f32_page_stride,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    codes_bytes_per_token: tl.constexpr,
    metadata_values_per_token: tl.constexpr,
    block_tokens: tl.constexpr,
    int2_max: tl.constexpr,
    clip_index: tl.constexpr,
):
    token_block = tl.program_id(0)
    group_id = tl.program_id(1)
    token_offsets = token_block * block_tokens + tl.arange(0, block_tokens)
    token_mask = token_offsets < num_tokens
    locations = tl.load(locations_ptr + token_offsets, mask=token_mask, other=-1).to(
        tl.int64
    )
    requested = tl.load(write_mask_ptr + token_offsets, mask=token_mask, other=0).to(
        tl.int1
    )
    active = token_mask & requested & (locations >= 0) & (locations < capacity)
    safe_locations = tl.where(active, locations, 0)

    input_offsets = tl.arange(0, head_dim)
    group_offsets = tl.arange(0, group_size)
    input_tile = tl.load(
        keys_ptr + token_offsets[:, None] * keys_stride + input_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    )
    rotation_tile = tl.load(
        rotation_ptr
        + input_offsets[:, None] * rotation_stride_in
        + (group_id * group_size + group_offsets[None, :]) * rotation_stride_out
    )
    rotated = tl.dot(input_tile, rotation_tile, out_dtype=tl.float32)
    sorted_absolute = tl.sort(tl.abs(rotated), dim=1)
    selected = group_offsets[None, :] == clip_index
    threshold = tl.sum(tl.where(selected, sorted_absolute, 0.0), axis=1)
    clipped = tl.minimum(tl.maximum(rotated, -threshold[:, None]), threshold[:, None])
    minimum = tl.min(clipped, axis=1)
    maximum = tl.max(clipped, axis=1)
    scale = tl.where(maximum == minimum, 1.0, (maximum - minimum) / int2_max)
    zero = -minimum / scale
    codes = tl.floor(clipped / scale[:, None] + zero[:, None] + 0.5)
    codes = tl.maximum(tl.minimum(codes, int2_max), 0.0).to(tl.uint8)
    shaped = tl.reshape(codes, (block_tokens, packed_group_bytes, 2, 2))
    even, odd = tl.split(shaped)
    code0, code2 = tl.split(even)
    code1, code3 = tl.split(odd)
    packed = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)

    page = safe_locations // page_size
    in_page = safe_locations - page * page_size
    code_base = (
        page * storage_u8_page_stride
        + in_page * codes_bytes_per_token
        + group_id * packed_group_bytes
    )
    byte_offsets = tl.arange(0, packed_group_bytes)
    tl.store(
        storage_u8_ptr + code_base[:, None] + byte_offsets[None, :],
        packed,
        mask=active[:, None],
    )
    metadata_base = (
        page * storage_f32_page_stride
        + (page_size * codes_bytes_per_token) // 4
        + in_page * metadata_values_per_token
        + group_id * 2
    )
    tl.store(storage_f32_ptr + metadata_base, scale, mask=active)
    tl.store(storage_f32_ptr + metadata_base + 1, zero, mask=active)


@triton.jit
def _legacy_c4_repeated_query_rotation_kernel(
    query_ptr,
    rotation_ptr,
    storage_u8_ptr,
    storage_f32_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    output_ptr,
    rotation_stride_in,
    rotation_stride_out,
    storage_u8_page_stride,
    storage_f32_page_stride,
    page_table_stride,
    max_seq_len: tl.constexpr,
    page_table_width: tl.constexpr,
    num_cache_pages: tl.constexpr,
    programs_per_query: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    packed_bytes_per_token: tl.constexpr,
    metadata_values_per_token: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    program_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), max_seq_len)
    active_pages = tl.minimum(tl.cdiv(bounded_seq_len, page_size), page_table_width)

    if program_idx < active_pages:
        head_offsets = tl.arange(0, num_heads)
        dim_offsets = tl.arange(0, head_dim)
        token_offsets = tl.arange(0, page_size)
        packed_offsets = tl.arange(0, packed_bytes_per_token)
        query_offsets = (
            batch_idx * num_heads * head_dim
            + head_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        query = tl.load(query_ptr + query_offsets)
        rotation = tl.load(
            rotation_ptr
            + dim_offsets[:, None] * rotation_stride_in
            + dim_offsets[None, :] * rotation_stride_out
        )
        rotated_query = tl.dot(query, rotation, out_dtype=tl.float32).to(tl.bfloat16)
        weights = tl.load(weight_ptr + batch_idx * num_heads + head_offsets).to(
            tl.float32
        )

        for page_slot in tl.range(program_idx, active_pages, programs_per_query):
            page_id = tl.load(
                page_table_ptr + batch_idx * page_table_stride + page_slot
            )
            page_is_valid = (page_id >= 0) & (page_id < num_cache_pages)
            safe_page_id = tl.where(page_is_valid, page_id, 0)
            packed = tl.load(
                storage_u8_ptr
                + safe_page_id * storage_u8_page_stride
                + token_offsets[:, None] * packed_bytes_per_token
                + packed_offsets[None, :],
                mask=page_is_valid,
                other=0,
            ).to(tl.uint8)
            code0 = packed & 0x03
            code1 = (packed >> 2) & 0x03
            code2 = (packed >> 4) & 0x03
            code3 = (packed >> 6) & 0x03
            even_codes = tl.interleave(code0, code2)
            odd_codes = tl.interleave(code1, code3)
            codes = tl.interleave(even_codes, odd_codes)
            metadata_base = (
                safe_page_id * storage_f32_page_stride
                + (page_size * packed_bytes_per_token) // 4
                + token_offsets * metadata_values_per_token
            )
            scale = tl.load(
                storage_f32_ptr + metadata_base, mask=page_is_valid, other=1.0
            )
            zero = tl.load(
                storage_f32_ptr + metadata_base + 1, mask=page_is_valid, other=0.0
            )
            keys = ((codes.to(tl.float32) - zero[:, None]) * scale[:, None]).to(
                tl.bfloat16
            )
            logits = tl.dot(keys, tl.trans(rotated_query), out_dtype=tl.float32)
            reduced = tl.sum(tl.maximum(logits, 0.0) * weights[None, :], axis=1)
            reduced = tl.where(page_is_valid, reduced, 0.0)
            output_positions = page_slot * page_size + token_offsets
            tl.store(
                output_ptr + batch_idx * max_seq_len + output_positions,
                reduced,
                mask=output_positions < bounded_seq_len,
            )


def _median_cuda_ms(
    function: Callable[[], object],
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    samples: int,
) -> float:
    for _ in range(warmup_iterations):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    sample_ms: list[float] = []
    for _ in range(samples):
        start.record()
        for _ in range(measurement_iterations):
            function()
        end.record()
        end.synchronize()
        sample_ms.append(start.elapsed_time(end) / measurement_iterations)
    return float(statistics.median(sample_ms))


def _capture(function: Callable[[], object]) -> torch.cuda.CUDAGraph:
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def _signed_permutation(size: int, device: torch.device) -> torch.Tensor:
    rotation = torch.zeros((size, size), dtype=torch.bfloat16, device=device)
    rows = torch.arange(size, device=device)
    columns = torch.remainder(rows + 1, size)
    signs = torch.where(rows % 3 == 0, -1.0, 1.0).to(torch.bfloat16)
    rotation[rows, columns] = signs
    return rotation


def _make_calibrations(
    device: torch.device,
) -> tuple[shared.OscarInt2Calibration, c4.OscarInt2C4Calibration]:
    shared_ratio = 0.95
    shared_ratios = torch.full(
        (shared.NUM_GROUPS,), shared_ratio, dtype=torch.float32, device=device
    )
    shared_calibration = shared.validate_oscar_int2_calibration(
        _signed_permutation(shared.NOPE_DIM, device),
        torch.full((shared.NUM_GROUPS,), 2.5, dtype=torch.float32, device=device),
        shared_ratios,
        torch.floor(shared_ratios * shared.GROUP_SIZE).to(torch.int16),
        layer_id=2,
        clip_mode=shared.CLIP_PER_ROW_QUANTILE,
        clip_provenance="hotpath-benchmark",
    )
    c4_ratios = torch.tensor([0.95], dtype=torch.float32, device=device)
    c4_calibration = c4.validate_oscar_int2_c4_calibration(
        _signed_permutation(c4.HEAD_DIM, device),
        torch.tensor([2.5], dtype=torch.float32, device=device),
        c4_ratios,
        torch.floor(c4_ratios * c4.GROUP_SIZE).to(torch.int16),
        layer_id=2,
        clip_mode=c4.CLIP_MODE,
        clip_provenance="hotpath-benchmark",
    )
    return shared_calibration, c4_calibration


def _legacy_shared_call(
    values: torch.Tensor,
    calibration: shared.OscarInt2Calibration,
    storage: torch.Tensor,
    locations: torch.Tensor,
    write_mask: torch.Tensor,
    *,
    page_size: int,
) -> None:
    block_tokens = 16
    storage_bf16 = storage.view(torch.bfloat16)
    grid = (triton.cdiv(values.shape[0], block_tokens), shared.NUM_GROUPS)
    _legacy_shared_masked_writer_kernel[grid](
        values,
        calibration.rotation,
        calibration.clip_indices,
        storage,
        storage_bf16,
        locations,
        write_mask,
        values.shape[0],
        storage.shape[0] * page_size,
        values.stride(0),
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        storage.stride(0),
        storage_bf16.stride(0),
        page_size=page_size,
        token_bytes=shared.STORAGE_BYTES_PER_TOKEN,
        nope_dim=shared.NOPE_DIM,
        rotation_input_tile=512,
        rope_dim=shared.ROPE_DIM,
        group_size=shared.GROUP_SIZE,
        packed_group_bytes=shared.PACKED_GROUP_BYTES,
        rope_offset_bytes=shared.ROPE_OFFSET_BYTES,
        scale_zero_offset_bytes=shared.SCALE_ZERO_OFFSET_BYTES,
        block_tokens=block_tokens,
        int2_max=shared.INT2_MAX,
        num_warps=8,
        num_stages=1,
    )


def _legacy_c4_writer_call(
    keys: torch.Tensor,
    calibration: c4.OscarInt2C4Calibration,
    storage: torch.Tensor,
    locations: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    block_tokens = 16
    storage_f32 = storage.view(torch.float32)
    grid = (triton.cdiv(keys.shape[0], block_tokens), c4.NUM_GROUPS)
    _legacy_c4_masked_writer_kernel[grid](
        keys,
        calibration.rotation,
        storage,
        storage_f32,
        locations,
        write_mask,
        keys.shape[0],
        storage.shape[0] * c4.PAGE_SIZE,
        keys.stride(0),
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        storage.stride(0),
        storage_f32.stride(0),
        page_size=c4.PAGE_SIZE,
        head_dim=c4.HEAD_DIM,
        group_size=c4.GROUP_SIZE,
        packed_group_bytes=c4.PACKED_GROUP_BYTES,
        codes_bytes_per_token=c4.CODES_BYTES_PER_TOKEN,
        metadata_values_per_token=c4.METADATA_VALUES_PER_TOKEN,
        block_tokens=block_tokens,
        int2_max=c4.INT2_MAX,
        clip_index=calibration.clip_index,
        num_warps=4,
        num_stages=1,
    )


def _legacy_c4_scorer_call(
    query: torch.Tensor,
    calibration: c4.OscarInt2C4Calibration,
    storage: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    output: torch.Tensor,
    *,
    max_seq_len: int,
) -> None:
    storage_f32 = storage.view(torch.float32)
    max_pages = triton.cdiv(max_seq_len, c4.PAGE_SIZE)
    programs_per_query = c4._programs_per_query(query.shape[0], max_pages)
    _legacy_c4_repeated_query_rotation_kernel[(query.shape[0], programs_per_query)](
        query,
        calibration.rotation,
        storage,
        storage_f32,
        weights,
        seq_lens,
        page_table,
        output,
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        storage.stride(0),
        storage_f32.stride(0),
        page_table.stride(0),
        max_seq_len=max_seq_len,
        page_table_width=page_table.shape[1],
        num_cache_pages=storage.shape[0],
        programs_per_query=programs_per_query,
        page_size=c4.PAGE_SIZE,
        num_heads=c4.NUM_HEADS,
        head_dim=c4.HEAD_DIM,
        packed_bytes_per_token=c4.CODES_BYTES_PER_TOKEN,
        metadata_values_per_token=c4.METADATA_VALUES_PER_TOKEN,
        num_warps=8,
        num_stages=1,
    )


def _compare(
    before: Callable[[], object],
    after: Callable[[], object],
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    samples: int,
) -> dict[str, float]:
    before()
    after()
    torch.cuda.synchronize()
    before_graph = _capture(before)
    after_graph = _capture(after)
    before_graph_ms = _median_cuda_ms(
        before_graph.replay,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        samples=samples,
    )
    after_graph_ms = _median_cuda_ms(
        after_graph.replay,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        samples=samples,
    )
    return {
        "before_graph_ms": before_graph_ms,
        "after_graph_ms": after_graph_ms,
        "graph_speedup": before_graph_ms / after_graph_ms,
    }


def benchmark(
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    warmup_iterations: int,
    measurement_iterations: int,
    samples: int,
) -> dict[str, Any]:
    if not 0 < batch_size <= 16:
        raise ValueError("batch_size must be in [1, 16] for masked-row specialization")
    if sequence_length <= 0 or sequence_length % c4.PAGE_SIZE:
        raise ValueError("sequence_length must be a positive multiple of 64")
    if warmup_iterations < 0 or measurement_iterations <= 0 or samples <= 0:
        raise ValueError(
            "benchmark iteration counts must be positive (warmup may be 0)"
        )
    shared_calibration, c4_calibration = _make_calibrations(device)
    generator = torch.Generator(device=device).manual_seed(611)
    locations = torch.arange(batch_size, dtype=torch.int32, device=device)
    write_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    shared_values = torch.randn(
        (batch_size, shared.HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    shared_storage = torch.full(
        (1, 16 * shared.STORAGE_BYTES_PER_TOKEN),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )

    def shared_before() -> None:
        _legacy_shared_call(
            shared_values,
            shared_calibration,
            shared_storage,
            locations,
            write_mask,
            page_size=16,
        )

    def shared_after() -> None:
        shared.quantize_dsv4_oscar_int2_cache_paged(
            shared_values,
            shared_calibration,
            shared_storage,
            locations,
            page_size=16,
            write_mask=write_mask,
        )

    c4_keys = torch.randn(
        (batch_size, c4.HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    c4_writer_storage = torch.full(
        (1, c4.PAGE_BYTES), 0xA5, dtype=torch.uint8, device=device
    )

    def c4_writer_before() -> None:
        _legacy_c4_writer_call(
            c4_keys,
            c4_calibration,
            c4_writer_storage,
            locations,
            write_mask,
        )

    def c4_writer_after() -> None:
        c4.store_oscar_int2_c4_indexer_cache(
            c4_keys,
            c4_writer_storage,
            locations,
            calibration=c4_calibration,
            write_mask=write_mask,
        )

    num_pages = sequence_length // c4.PAGE_SIZE
    query = torch.randn(
        (1, 1, c4.NUM_HEADS, c4.HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.randn(
        (1, c4.NUM_HEADS),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    seq_lens = torch.tensor([sequence_length], dtype=torch.int32, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device)[None, :]
    scorer_storage = torch.empty(
        (num_pages, c4.PAGE_BYTES), dtype=torch.uint8, device=device
    )
    scorer_storage[:, : c4.CODES_BYTES_PER_PAGE].random_(0, 256, generator=generator)
    scorer_metadata = scorer_storage[:, c4.METADATA_OFFSET_BYTES :].view(torch.float32)
    scorer_metadata[:, 0::2].fill_(0.125)
    scorer_metadata[:, 1::2].fill_(1.5)
    before_output = torch.empty(
        (1, sequence_length), dtype=torch.float32, device=device
    )
    after_output = torch.empty_like(before_output)
    rotated_query_out = torch.empty(
        (1, c4.NUM_HEADS, c4.HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    def scorer_before() -> None:
        _legacy_c4_scorer_call(
            query,
            c4_calibration,
            scorer_storage,
            weights,
            seq_lens,
            page_table,
            before_output,
            max_seq_len=sequence_length,
        )

    def scorer_after() -> None:
        c4.oscar_int2_c4_paged_mqa_logits_triton(
            query,
            scorer_storage,
            weights,
            seq_lens,
            page_table,
            None,
            sequence_length,
            False,
            calibration=c4_calibration,
            out=after_output,
            rotated_query_out=rotated_query_out,
        )

    results = {
        "shared_masked_writer": _compare(
            shared_before,
            shared_after,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            samples=samples,
        ),
        "c4_masked_writer": _compare(
            c4_writer_before,
            c4_writer_after,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            samples=samples,
        ),
        "c4_scorer_query_rotation": _compare(
            scorer_before,
            scorer_after,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            samples=samples,
        ),
    }
    torch.testing.assert_close(after_output, before_output, rtol=0.0, atol=0.0)
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "c4_programs_per_query": c4._programs_per_query(1, num_pages),
        "warmup_iterations_per_provider": warmup_iterations,
        "measurement_iterations_per_sample": measurement_iterations,
        "samples_per_provider": samples,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=524_288)
    parser.add_argument("--warmup-iterations", type=int, default=256)
    parser.add_argument("--measurement-iterations", type=int, default=512)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 6):
        raise SystemExit(f"OSCAR hotpaths require exact SM86, got {capability}")
    result = benchmark(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        samples=args.samples,
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "compute_capability": "8.6",
                **result,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
