from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    compress_norm_rope_store,
)
from sglang.kernels.ops.attention.dsv4.int4_c4_indexer_poc import (
    INT4_C4_BYTES_PER_TOKEN,
    INT4_C4_POC_GROUP_SIZE,
    INT4_C4_POC_HEAD_DIM,
    INT4_C4_POC_NUM_GROUPS,
    INT4_C4_POC_NUM_HEADS,
    INT4_C4_POC_PACKED_BYTES_PER_TOKEN,
    INT4_C4_POC_PAGE_BYTES,
    INT4_C4_POC_PAGE_SIZE,
    INT4_C4_POC_SCALE_BYTES,
    INT4_C4_POC_VALUE_BYTES,
    int4_c4_page_bytes,
    int4_c4_paged_mqa_logits_reference,
    int4_c4_paged_mqa_logits_triton,
    pack_int4_c4_pages_reference,
    store_int4_c4_indexer_cache,
    unpack_int4_c4_pages_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm80_or_newer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("the PoC targets NVIDIA CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("the PoC scorer requires BF16 tensor cores")


def _random_pages(num_pages: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (
            num_pages,
            INT4_C4_POC_PAGE_SIZE,
            INT4_C4_POC_HEAD_DIM,
        ),
        generator=generator,
        dtype=torch.float32,
    ).clamp_(-3.0, 3.0)


def _identity_freqs(num_positions: int, *, device: str) -> torch.Tensor:
    interleaved = torch.zeros((num_positions, 64), dtype=torch.float32, device=device)
    interleaved[:, 0::2] = 1.0
    return torch.view_as_complex(interleaved.view(num_positions, 32, 2))


def _make_cuda_case(
    batch_size: int,
    max_seq_len: int,
    seq_lens: list[int],
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    max_pages = (max_seq_len + INT4_C4_POC_PAGE_SIZE - 1) // INT4_C4_POC_PAGE_SIZE
    physical_pages = max_pages + 7
    cache = pack_int4_c4_pages_reference(
        _random_pages(physical_pages, seed=seed)
    ).cuda()

    generator = torch.Generator().manual_seed(seed + 1)
    query = torch.randn(
        (
            batch_size,
            1,
            INT4_C4_POC_NUM_HEADS,
            INT4_C4_POC_HEAD_DIM,
        ),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    weights = (
        torch.randn(
            (batch_size, INT4_C4_POC_NUM_HEADS),
            generator=generator,
            dtype=torch.float32,
        )
        / INT4_C4_POC_NUM_HEADS
    )
    logical_pages = torch.arange(max_pages, dtype=torch.int32)
    batch_offsets = torch.arange(batch_size, dtype=torch.int32)[:, None] * 3
    page_table = (logical_pages[None, :] + batch_offsets) % physical_pages
    if max_pages >= 2:
        page_table[0, 1] = -1
    if max_pages >= 3 and batch_size >= 2:
        page_table[1, 2] = physical_pages + 5

    return (
        query.cuda(),
        cache,
        weights.cuda(),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        page_table.contiguous().cuda(),
    )


def _cpu_reference_from_cuda(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    return int4_c4_paged_mqa_logits_reference(
        query.cpu(),
        cache.cpu(),
        weights.cpu(),
        lengths.cpu(),
        page_table.cpu(),
        max_seq_len,
    )


def test_int4_c4_poc_layout_is_exactly_4608_bytes_per_page() -> None:
    assert INT4_C4_POC_PAGE_SIZE == 64
    assert INT4_C4_POC_HEAD_DIM == 128
    assert INT4_C4_POC_GROUP_SIZE == 32
    assert INT4_C4_POC_NUM_GROUPS == 4
    assert INT4_C4_POC_PACKED_BYTES_PER_TOKEN == 64
    assert INT4_C4_POC_VALUE_BYTES == 4096
    assert INT4_C4_POC_SCALE_BYTES == 512
    assert INT4_C4_POC_PAGE_BYTES == 4608
    assert INT4_C4_BYTES_PER_TOKEN == 72
    assert int4_c4_page_bytes(64) == 4608


def test_int4_c4_poc_reference_uses_signed_low_then_high_nibbles() -> None:
    keys = torch.zeros(
        (1, INT4_C4_POC_PAGE_SIZE, INT4_C4_POC_HEAD_DIM),
        dtype=torch.float32,
    )
    keys[0, 0, :4] = torch.tensor([7.0, -7.0, -1.0, 1.0])

    cache = pack_int4_c4_pages_reference(keys)
    decoded = unpack_int4_c4_pages_reference(cache)
    scales = (
        cache[:, INT4_C4_POC_VALUE_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(1, INT4_C4_POC_PAGE_SIZE, INT4_C4_POC_NUM_GROUPS)
    )

    assert cache[0, 0].item() == 0x97
    assert cache[0, 1].item() == 0x1F
    assert scales[0, 0, 0].item() == 1.0
    torch.testing.assert_close(decoded[0, 0, :4], keys[0, 0, :4], rtol=0, atol=0)


def test_int4_c4_poc_reference_quantization_error_is_bounded() -> None:
    keys = _random_pages(3, seed=11)
    cache = pack_int4_c4_pages_reference(keys)
    decoded = unpack_int4_c4_pages_reference(cache)
    scales = (
        cache[:, INT4_C4_POC_VALUE_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(
            3,
            INT4_C4_POC_PAGE_SIZE,
            INT4_C4_POC_NUM_GROUPS,
        )
        .float()
    )
    error = (
        (decoded - keys)
        .abs()
        .reshape(
            3,
            INT4_C4_POC_PAGE_SIZE,
            INT4_C4_POC_NUM_GROUPS,
            INT4_C4_POC_GROUP_SIZE,
        )
    )

    # BF16 dequantization adds a small rounding term beyond the ideal half-step.
    assert torch.all(error <= scales.unsqueeze(-1) * 0.52 + 1.0e-3)


def test_int4_c4_poc_cpu_reference_masks_sequence_and_invalid_pages() -> None:
    keys = torch.zeros(
        (2, INT4_C4_POC_PAGE_SIZE, INT4_C4_POC_HEAD_DIM),
        dtype=torch.float32,
    )
    keys[0, :, 0] = 1.0
    keys[1, :, 0] = 2.0
    cache = pack_int4_c4_pages_reference(keys)
    query = torch.zeros(
        (1, 1, INT4_C4_POC_NUM_HEADS, INT4_C4_POC_HEAD_DIM),
        dtype=torch.bfloat16,
    )
    query[0, 0, 0, 0] = 1.0
    weights = torch.zeros((1, INT4_C4_POC_NUM_HEADS), dtype=torch.float32)
    weights[0, 0] = 1.0
    lengths = torch.tensor([130], dtype=torch.int32)
    page_table = torch.tensor([[1, -1, 0]], dtype=torch.int32)

    result = int4_c4_paged_mqa_logits_reference(
        query, cache, weights, lengths, page_table, 192
    )
    decoded = unpack_int4_c4_pages_reference(cache)

    torch.testing.assert_close(
        result[0, :64],
        decoded[1, :, 0],
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(result[0, 64:128]) == 0
    torch.testing.assert_close(
        result[0, 128:130],
        decoded[0, :2, 0],
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(result[0, 130:]) == 0


@pytest.mark.parametrize(
    "batch_size,max_seq_len,seq_lens",
    [
        (8, 256, [0, 1, 31, 63, 64, 65, 127, 191]),
        (3, 320, [319, 320, 400]),
    ],
)
def test_int4_c4_poc_triton_matches_cpu_reference_at_boundaries(
    batch_size: int,
    max_seq_len: int,
    seq_lens: list[int],
) -> None:
    _require_sm80_or_newer()
    query, cache, weights, lengths, page_table = _make_cuda_case(
        batch_size,
        max_seq_len,
        seq_lens,
        seed=23 + batch_size,
    )
    scorer_lengths = lengths[:, None] if batch_size == 3 else lengths
    output = torch.full(
        (batch_size, max_seq_len),
        torch.nan,
        dtype=torch.float32,
        device="cuda",
    )

    actual = int4_c4_paged_mqa_logits_triton(
        query,
        cache,
        weights,
        scorer_lengths,
        page_table,
        None,
        max_seq_len,
        clean_logits=True,
        out=output,
    )
    expected = _cpu_reference_from_cuda(
        query, cache, weights, lengths, page_table, max_seq_len
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.cpu(), expected, rtol=2.0e-2, atol=2.0e-2)


def test_int4_c4_poc_replays_chunked_kernel_in_real_cuda_graph() -> None:
    _require_sm80_or_newer()
    batch_size = 33
    max_seq_len = 4096
    initial_lengths = [((index * 127) % max_seq_len) + 1 for index in range(batch_size)]
    query, cache, weights, lengths, page_table = _make_cuda_case(
        batch_size,
        max_seq_len,
        initial_lengths,
        seed=71,
    )
    lengths_column = lengths[:, None]
    output = torch.full(
        (batch_size, max_seq_len),
        torch.nan,
        dtype=torch.float32,
        device="cuda",
    )

    def run() -> torch.Tensor:
        return int4_c4_paged_mqa_logits_triton(
            query,
            cache,
            weights,
            lengths_column,
            page_table,
            None,
            max_seq_len,
            clean_logits=False,
            out=output,
        )

    # Prime compilation and launch metadata before capture.
    run()
    torch.cuda.synchronize()
    output_pointer = output.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()

    generator = torch.Generator().manual_seed(73)
    query.copy_(
        torch.randn(query.shape, generator=generator, dtype=torch.float32).to(
            device="cuda", dtype=torch.bfloat16
        )
    )
    weights.copy_(
        torch.randn(weights.shape, generator=generator, dtype=torch.float32).cuda()
        / INT4_C4_POC_NUM_HEADS
    )
    lengths.copy_(torch.tensor(initial_lengths[::-1], dtype=torch.int32, device="cuda"))
    page_table.copy_(page_table.roll(shifts=1, dims=1))
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output_pointer
    expected = _cpu_reference_from_cuda(
        query, cache, weights, lengths, page_table, max_seq_len
    )
    positions = torch.arange(max_seq_len)[None, :]
    valid_positions = positions < lengths.cpu()[:, None]
    torch.testing.assert_close(
        captured_output.cpu()[valid_positions],
        expected[valid_positions],
        rtol=2.0e-2,
        atol=2.0e-2,
    )


@pytest.mark.parametrize("writer", ["triton", "cuda_jit"])
def test_int4_c4_production_writers_match_reference(writer: str) -> None:
    _require_sm80_or_newer()
    page_size = INT4_C4_POC_PAGE_SIZE
    keys = _random_pages(2, seed=83).reshape(-1, INT4_C4_POC_HEAD_DIM)
    keys = keys.to(device="cuda", dtype=torch.bfloat16).contiguous()
    locations = torch.arange(keys.shape[0], dtype=torch.int32, device="cuda")
    cache = torch.full(
        (2, int4_c4_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    if writer == "triton":
        store_int4_c4_indexer_cache(keys, cache, locations, page_size=page_size)
    else:
        fused_store_cache(
            keys,
            cache,
            locations,
            page_size=page_size,
            type="indexer",
            int4_store=True,
        )
    torch.cuda.synchronize()

    expected = pack_int4_c4_pages_reference(
        keys.reshape(2, page_size, INT4_C4_POC_HEAD_DIM)
    )
    assert torch.equal(cache, expected)


def test_int4_c4_writer_handles_page_edges_negative_locations_and_slot_reuse() -> None:
    _require_sm80_or_newer()
    page_size = INT4_C4_POC_PAGE_SIZE
    keys = _random_pages(1, seed=87)[0, :5]
    keys = keys.to(device="cuda", dtype=torch.bfloat16).contiguous()
    locations = torch.tensor([0, 63, 64, 127, -1], dtype=torch.int32, device="cuda")
    cache = torch.full(
        (2, int4_c4_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    store_int4_c4_indexer_cache(keys, cache, locations, page_size=page_size)
    for key_id, location in enumerate(locations[:-1].cpu().tolist()):
        page, in_page = divmod(location, page_size)
        reference_keys = torch.zeros(
            (1, page_size, INT4_C4_POC_HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference_keys[0, in_page] = keys[key_id]
        reference = pack_int4_c4_pages_reference(reference_keys)[0]
        value_start = in_page * INT4_C4_POC_PACKED_BYTES_PER_TOKEN
        value_end = value_start + INT4_C4_POC_PACKED_BYTES_PER_TOKEN
        scale_start = INT4_C4_POC_VALUE_BYTES + in_page * 8
        scale_end = scale_start + 8
        assert torch.equal(
            cache[page, value_start:value_end], reference[value_start:value_end]
        )
        assert torch.equal(
            cache[page, scale_start:scale_end], reference[scale_start:scale_end]
        )

    # The skipped negative entry must not alias slot zero or an arbitrary row.
    assert torch.all(
        cache[
            0,
            1 * INT4_C4_POC_PACKED_BYTES_PER_TOKEN : 2
            * INT4_C4_POC_PACKED_BYTES_PER_TOKEN,
        ]
        == 0xA5
    )
    assert torch.all(
        cache[0, INT4_C4_POC_VALUE_BYTES + 8 : INT4_C4_POC_VALUE_BYTES + 16] == 0xA5
    )

    replacement = _random_pages(1, seed=91)[0, :1].to(
        device="cuda", dtype=torch.bfloat16
    )
    replacement_location = torch.tensor([63], dtype=torch.int32, device="cuda")
    store_int4_c4_indexer_cache(
        replacement,
        cache,
        replacement_location,
        page_size=page_size,
    )
    reference_keys = torch.zeros(
        (1, page_size, INT4_C4_POC_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    reference_keys[0, 63] = replacement[0]
    reference = pack_int4_c4_pages_reference(reference_keys)[0]
    value_start = 63 * INT4_C4_POC_PACKED_BYTES_PER_TOKEN
    scale_start = INT4_C4_POC_VALUE_BYTES + 63 * 8
    assert torch.equal(
        cache[0, value_start : value_start + INT4_C4_POC_PACKED_BYTES_PER_TOKEN],
        reference[value_start : value_start + INT4_C4_POC_PACKED_BYTES_PER_TOKEN],
    )
    assert torch.equal(
        cache[0, scale_start : scale_start + 8],
        reference[scale_start : scale_start + 8],
    )


def test_int4_c4_compressor_writer_matches_bf16_production_path() -> None:
    _require_sm80_or_newer()
    page_size = INT4_C4_POC_PAGE_SIZE
    compress_ratio = 4
    num_tokens = 9
    generator = torch.Generator().manual_seed(97)
    keys = torch.randn(
        (num_tokens, INT4_C4_POC_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(
        (INT4_C4_POC_HEAD_DIM,), generator=generator, dtype=torch.float32
    ).to(device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.arange(
        compress_ratio,
        (num_tokens + 1) * compress_ratio,
        compress_ratio,
        dtype=torch.int64,
        device="cuda",
    )
    req_pool_indices = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio, req_pool_indices, seq_lens
    )
    locations = torch.tensor(
        [0, 1, 7, 31, 63, 64, 65, 95, 127],
        dtype=torch.int64,
        device="cuda",
    )
    freqs_cis = _identity_freqs(int(seq_lens.max().item()), device="cuda")
    int4_cache = torch.full(
        (2, int4_c4_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    bf16_cache = torch.zeros(
        (2, page_size * INT4_C4_POC_HEAD_DIM * 2),
        dtype=torch.uint8,
        device="cuda",
    )

    common = dict(
        kv=keys,
        plan=plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=locations,
        page_size=page_size,
    )
    compress_norm_rope_store(kvcache=int4_cache, int4_store=True, **common)
    compress_norm_rope_store(kvcache=bf16_cache, bf16_store=True, **common)
    torch.cuda.synchronize()

    int4_pages = unpack_int4_c4_pages_reference(int4_cache).float()
    bf16_pages = bf16_cache.view(torch.bfloat16).reshape(
        2, page_size, INT4_C4_POC_HEAD_DIM
    )
    page_ids = locations // page_size
    in_page = locations % page_size
    actual = int4_pages[page_ids, in_page]
    expected = bf16_pages[page_ids, in_page].float()
    stored_scales = (
        int4_cache[:, INT4_C4_POC_VALUE_BYTES:]
        .view(torch.bfloat16)
        .reshape(2, page_size, INT4_C4_POC_NUM_GROUPS)
    )
    scale_matrix = (
        stored_scales[page_ids, in_page]
        .float()
        .repeat_interleave(INT4_C4_POC_GROUP_SIZE, dim=1)
    )
    assert torch.all((actual - expected).abs() <= scale_matrix * 0.53 + 1.0e-3)


def test_int4_c4_scorer_accepts_fp8_query_bytes() -> None:
    _require_sm80_or_newer()
    query, cache, weights, lengths, page_table = _make_cuda_case(
        2, 256, [193, 256], seed=89
    )
    query_fp32 = query.float()
    query_scale = query_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 448.0
    query_fp8 = (query_fp32 / query_scale).to(torch.float8_e4m3fn)
    scaled_weights = weights * query_scale[:, 0, :, 0]
    output = torch.empty((2, 256), dtype=torch.float32, device="cuda")

    actual = int4_c4_paged_mqa_logits_triton(
        query_fp8,
        cache,
        scaled_weights,
        lengths,
        page_table,
        None,
        256,
        clean_logits=True,
        out=output,
    )
    decoded_query = query_fp8.float().to(torch.bfloat16)
    expected = _cpu_reference_from_cuda(
        decoded_query,
        cache,
        scaled_weights,
        lengths,
        page_table,
        256,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2.0e-2, atol=2.0e-2)


def test_int4_c4_production_no_out_allocation_replays_in_cuda_graph() -> None:
    """The production indexer call omits ``out`` and relies on graph memory."""

    _require_sm80_or_newer()
    batch_size = 2
    max_seq_len = 256
    query, cache, weights, lengths, page_table = _make_cuda_case(
        batch_size, max_seq_len, [193, 256], seed=103
    )

    def run() -> torch.Tensor:
        return int4_c4_paged_mqa_logits_triton(
            query,
            cache,
            weights,
            lengths,
            page_table,
            None,
            max_seq_len,
            clean_logits=False,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()

    generator = torch.Generator().manual_seed(107)
    query.copy_(
        torch.randn(query.shape, generator=generator, dtype=torch.float32).to(
            device="cuda", dtype=torch.bfloat16
        )
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _cpu_reference_from_cuda(
        query, cache, weights, lengths, page_table, max_seq_len
    )
    torch.testing.assert_close(
        captured_output.cpu(), expected, rtol=2.0e-2, atol=2.0e-2
    )
