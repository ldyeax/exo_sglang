from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.attention.dsv4 import fused_store_cache
from sglang.kernels.ops.attention.dsv4.fp8_storage_indexer import (
    DSV4_INDEXER_PAGE_BYTES,
    fp8_storage_paged_mqa_logits_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA CUDA test"),
]

_PAGE_SIZE = 64
_NUM_HEADS = 64
_HEAD_DIM = 128
_VALUE_BYTES = _PAGE_SIZE * _HEAD_DIM


def _require_bf16_tensor_cores() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("the byte-storage scorer requires BF16 tensor cores")


def _make_case(
    batch_size: int,
    max_seq_len: int,
    seq_lens: list[int],
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    max_pages = (max_seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    physical_pages = max_pages + 7

    query = torch.randn(
        batch_size,
        1,
        _NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    key_source = torch.randn(
        physical_pages,
        _PAGE_SIZE,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    key_scales = (
        torch.rand(
            physical_pages,
            _PAGE_SIZE,
            dtype=torch.float32,
            device=device,
        )
        * 0.75
        + 0.25
    )
    keys = (key_source / key_scales[:, :, None]).to(torch.float8_e4m3fn)

    cache = torch.empty(
        physical_pages,
        DSV4_INDEXER_PAGE_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    cache[:, :_VALUE_BYTES].copy_(
        keys.view(torch.uint8).reshape(physical_pages, _VALUE_BYTES)
    )
    cache[:, _VALUE_BYTES:].view(torch.float32).copy_(key_scales)

    weights = (
        torch.randn(
            batch_size,
            _NUM_HEADS,
            dtype=torch.float32,
            device=device,
        )
        / _NUM_HEADS
    )
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    logical_pages = torch.arange(max_pages, dtype=torch.int32, device=device)
    batch_offsets = (
        torch.arange(batch_size, dtype=torch.int32, device=device)[:, None] * 3
    )
    page_table = (logical_pages[None, :] + batch_offsets) % physical_pages
    return query, cache, weights, lengths, page_table.contiguous()


def _reference(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    physical_pages = cache.shape[0]
    values = (
        cache[:, :_VALUE_BYTES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(physical_pages, _PAGE_SIZE, _HEAD_DIM)
        .to(torch.float32)
    )
    scales = (
        cache[:, _VALUE_BYTES:]
        .contiguous()
        .view(torch.float32)
        .reshape(physical_pages, _PAGE_SIZE)
    )
    needed_pages = (max_seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    page_ids = page_table[:, :needed_pages].to(torch.int64)
    valid_pages = (page_ids >= 0) & (page_ids < physical_pages)
    safe_page_ids = page_ids.clamp(0, physical_pages - 1)
    gathered_values = values[safe_page_ids].reshape(query.shape[0], -1, _HEAD_DIM)[
        :, :max_seq_len
    ]
    gathered_scales = scales[safe_page_ids].reshape(query.shape[0], -1)[:, :max_seq_len]
    valid_tokens = (
        valid_pages[:, :, None]
        .expand(-1, -1, _PAGE_SIZE)
        .reshape(query.shape[0], -1)[:, :max_seq_len]
    )
    gathered_values = gathered_values.masked_fill(~valid_tokens[:, :, None], 0.0)
    gathered_scales = gathered_scales.masked_fill(~valid_tokens, 0.0)
    query_values = query[:, 0].to(torch.float32)

    logits = torch.bmm(gathered_values, query_values.transpose(1, 2))
    logits = torch.relu(logits)
    result = (logits * weights[:, None, :]).sum(dim=2) * gathered_scales
    positions = torch.arange(max_seq_len, device=query.device)[None, :]
    return result.masked_fill(positions >= seq_lens.reshape(-1, 1), 0.0)


@pytest.mark.parametrize(
    "batch_size,max_seq_len,seq_lens",
    [
        (5, 256, [0, 1, 63, 64, 191]),
        # 33 * 128 pages exceeds the 2,048-program budget and therefore tests
        # the runtime-bounded strided program rather than the direct path.
        (33, 8192, [((index * 127) % 8192) + 1 for index in range(33)]),
    ],
)
def test_fp8_storage_indexer_matches_fp32_reference(
    batch_size: int,
    max_seq_len: int,
    seq_lens: list[int],
) -> None:
    _require_bf16_tensor_cores()
    query, cache, weights, lengths, page_table = _make_case(
        batch_size,
        max_seq_len,
        seq_lens,
        seed=17 + batch_size,
    )
    # Exercise both production sequence-length shapes.
    scorer_lengths = lengths if batch_size == 5 else lengths[:, None]

    actual = fp8_storage_paged_mqa_logits_triton(
        query,
        cache,
        weights,
        scorer_lengths,
        page_table,
        None,
        max_seq_len,
        clean_logits=True,
    )
    expected = _reference(
        query,
        cache,
        weights,
        lengths,
        page_table,
        max_seq_len,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_fp8_storage_indexer_large_capture_bounds_runtime_work() -> None:
    """A 128K graph width must not write pages beyond short live lengths."""
    _require_bf16_tensor_cores()
    batch_size = 6
    max_seq_len = 131072
    runtime_seq_len = 704
    query, cache, weights, lengths, page_table = _make_case(
        batch_size,
        max_seq_len,
        [runtime_seq_len] * batch_size,
        seed=52,
    )
    sentinel = -913.25
    output = torch.full(
        (batch_size, max_seq_len),
        sentinel,
        dtype=torch.float32,
        device=query.device,
    )

    returned = fp8_storage_paged_mqa_logits_triton(
        query,
        cache,
        weights,
        lengths,
        page_table,
        None,
        max_seq_len,
        clean_logits=False,
        out=output,
    )
    expected = _reference(
        query,
        cache,
        weights,
        lengths,
        page_table,
        runtime_seq_len,
    )

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(
        output[:, :runtime_seq_len], expected, rtol=2e-3, atol=2e-3
    )
    assert torch.all(output[:, runtime_seq_len:] == sentinel)


def test_fp8_storage_indexer_strided_full_coverage_and_invalid_pages() -> None:
    """Persistent lanes cover multiple pages and zero invalid physical IDs."""
    _require_bf16_tensor_cores()
    batch_size = 2
    max_seq_len = 8192
    query, cache, weights, lengths, page_table = _make_case(
        batch_size,
        max_seq_len,
        [max_seq_len, max_seq_len - 17],
        seed=68,
    )
    page_table[0, 3] = -1
    page_table[0, 97] = cache.shape[0]
    page_table[1, 64] = cache.shape[0] + 11

    actual = fp8_storage_paged_mqa_logits_triton(
        query,
        cache,
        weights,
        lengths[:, None],
        page_table,
        None,
        max_seq_len,
        clean_logits=True,
    )
    expected = _reference(
        query,
        cache,
        weights,
        lengths,
        page_table,
        max_seq_len,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert torch.count_nonzero(actual[0, 3 * _PAGE_SIZE : 4 * _PAGE_SIZE]) == 0
    assert torch.count_nonzero(actual[0, 97 * _PAGE_SIZE : 98 * _PAGE_SIZE]) == 0
    assert torch.count_nonzero(actual[1, 64 * _PAGE_SIZE : 65 * _PAGE_SIZE]) == 0


def test_fp8_storage_indexer_full_128k_persistent_coverage() -> None:
    """A single query covers all 2,048 pages through eight lane iterations."""
    _require_bf16_tensor_cores()
    max_seq_len = 131072
    query, cache, weights, lengths, page_table = _make_case(
        1,
        max_seq_len,
        [max_seq_len],
        seed=77,
    )

    actual = fp8_storage_paged_mqa_logits_triton(
        query,
        cache,
        weights,
        lengths,
        page_table,
        None,
        max_seq_len,
        clean_logits=False,
    )
    expected = _reference(
        query,
        cache,
        weights,
        lengths,
        page_table,
        max_seq_len,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_fp8_storage_indexer_writer_and_scorer_replay_in_one_cuda_graph() -> None:
    """The fused positive C4 writer feeds the scorer during graph replay."""
    _require_bf16_tensor_cores()
    device = torch.device("cuda", torch.cuda.current_device())
    max_seq_len = 128
    live_tokens = 65
    locations = torch.arange(live_tokens, dtype=torch.int32, device=device)
    key_values = torch.randn(
        (live_tokens, _HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    query = torch.randn(
        (1, 1, _NUM_HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((1, _NUM_HEADS), dtype=torch.float32, device=device)
    lengths = torch.tensor([live_tokens], dtype=torch.int32, device=device)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    cache = torch.zeros(
        (2, DSV4_INDEXER_PAGE_BYTES), dtype=torch.uint8, device=device
    )
    output = torch.empty((1, max_seq_len), dtype=torch.float32, device=device)

    def run() -> torch.Tensor:
        fused_store_cache(
            key_values,
            cache,
            locations,
            page_size=_PAGE_SIZE,
            type="indexer",
        )
        return fp8_storage_paged_mqa_logits_triton(
            query,
            cache,
            weights,
            lengths,
            page_table,
            None,
            max_seq_len,
            clean_logits=True,
            out=output,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    captured_output_ptr = output.data_ptr()
    key_values.copy_(torch.randn_like(key_values))
    query.copy_(torch.randn_like(query.to(torch.bfloat16)).to(query.dtype))
    weights.copy_(torch.randn_like(weights))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(
        query,
        cache,
        weights,
        lengths,
        page_table,
        max_seq_len,
    )
    assert output.data_ptr() == captured_output_ptr
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


def test_fp8_storage_indexer_persistent_cuda_graph_replay_with_length_changes() -> None:
    _require_bf16_tensor_cores()
    batch_size = 2
    max_seq_len = 131072
    initial_lengths = [65, 704]
    query, cache, weights, lengths, page_table = _make_case(
        batch_size,
        max_seq_len,
        initial_lengths,
        seed=91,
    )
    lengths_column = lengths[:, None]
    output = torch.empty(
        (batch_size, max_seq_len), dtype=torch.float32, device=query.device
    )

    def run() -> torch.Tensor:
        return fp8_storage_paged_mqa_logits_triton(
            query,
            cache,
            weights,
            lengths_column,
            page_table,
            None,
            max_seq_len,
            clean_logits=True,
            out=output,
        )

    # Prime the per-device LUT, persistent kernel, and launch machinery.  The
    # graph keeps both its grid and output address fixed while device-resident
    # runtime lengths grow enough to make each lane process multiple pages.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    captured_output_ptr = output.data_ptr()
    query.copy_(torch.randn_like(query.to(torch.bfloat16)).to(query.dtype))
    weights.copy_(torch.randn_like(weights) / _NUM_HEADS)
    replay_lengths = [8192, 4097]
    lengths.copy_(torch.tensor(replay_lengths, dtype=torch.int32, device=query.device))
    page_table.copy_(page_table.roll(shifts=1, dims=1))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(
        query,
        cache,
        weights,
        lengths,
        page_table,
        max(replay_lengths),
    )
    positions = torch.arange(max(replay_lengths), device=query.device)[None, :]
    valid = positions < lengths[:, None]
    assert output.data_ptr() == captured_output_ptr
    torch.testing.assert_close(
        output[:, : max(replay_lengths)][valid],
        expected[valid],
        rtol=2e-3,
        atol=2e-3,
    )
    assert torch.count_nonzero(output[:, max(replay_lengths) :]) == 0
