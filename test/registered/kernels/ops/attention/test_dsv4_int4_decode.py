from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.int4_decode import (
    decode_sparse_attention_int4,
)
from sglang.kernels.ops.attention.dsv4.int4_storage import (
    HEAD_DIM,
    dequantize_dsv4_int4_reference,
    int4_main_page_bytes,
    quantize_dsv4_int4_cache_paged,
    quantize_dsv4_int4_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm80_or_newer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("signed-INT4 sparse decode targets NVIDIA CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 tensor cores are required")


def _make_case(seed: int = 101) -> tuple[torch.Tensor, ...]:
    page_size = 8
    num_pages = 3
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(
        (num_pages * page_size, HEAD_DIM), generator=generator, dtype=torch.float32
    ).clamp_(-3.0, 3.0)
    keys = keys.to(device="cuda", dtype=torch.bfloat16)
    cache = torch.full(
        (num_pages, int4_main_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    locations = torch.arange(keys.shape[0], dtype=torch.int32, device="cuda")
    quantize_dsv4_int4_cache_paged(keys, cache, locations, page_size=page_size)

    query = torch.randn((2, 16, HEAD_DIM), generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.bfloat16
    )
    indices = torch.tensor(
        [[0, 3, 7, 8, 12, 17, 22, -1], [1, 2, 9, 10, 16, 18, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([7, 6], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    return query, keys, cache, indices, lengths, output


def _reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    decoded = dequantize_dsv4_int4_reference(quantize_dsv4_int4_reference(keys)).float()
    outputs = []
    for batch in range(query.shape[0]):
        active = indices[batch, : int(lengths[batch].item())].long()
        selected = decoded[active]
        logits = torch.einsum("hd,nd->hn", query[batch].float(), selected) * scale
        probability = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hn,nd->hd", probability, selected))
    return torch.stack(outputs).to(torch.bfloat16)


def test_int4_sparse_decode_matches_dequantized_reference() -> None:
    _require_sm80_or_newer()
    query, keys, cache, indices, lengths, output = _make_case()
    scale = HEAD_DIM**-0.5
    decode_sparse_attention_int4(
        q=query,
        swa_cache=cache,
        swa_indices=indices,
        swa_lens=lengths,
        scale=scale,
        attn_sink=None,
        out=output,
        swa_block_size=8,
    )
    torch.cuda.synchronize()
    expected = _reference(query, keys, indices, lengths, scale)
    torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=2.0e-2)


def test_int4_sparse_decode_replays_real_cuda_graph() -> None:
    _require_sm80_or_newer()
    query, keys, cache, indices, lengths, output = _make_case(seed=107)
    scale = HEAD_DIM**-0.5

    def run() -> None:
        decode_sparse_attention_int4(
            q=query,
            swa_cache=cache,
            swa_indices=indices,
            swa_lens=lengths,
            scale=scale,
            attn_sink=None,
            out=output,
            swa_block_size=8,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    generator = torch.Generator().manual_seed(109)
    query.copy_(
        torch.randn(query.shape, generator=generator, dtype=torch.float32).to(
            device="cuda", dtype=torch.bfloat16
        )
    )
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(query, keys, indices, lengths, scale)
    torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=2.0e-2)


def _reference_with_extra_and_sink(
    query: torch.Tensor,
    swa_keys: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    extra_keys: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lengths: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    decoded_swa = dequantize_dsv4_int4_reference(
        quantize_dsv4_int4_reference(swa_keys)
    ).float()
    decoded_extra = dequantize_dsv4_int4_reference(
        quantize_dsv4_int4_reference(extra_keys)
    ).float()
    outputs = []
    for token in range(query.shape[0]):
        active_swa = swa_indices[token, : int(swa_lengths[token].item())].long()
        active_extra = extra_indices[token, : int(extra_lengths[token].item())].long()
        selected = torch.cat(
            (decoded_extra[active_extra], decoded_swa[active_swa]), dim=0
        )
        logits = torch.einsum("hd,nd->hn", query[token].float(), selected) * scale
        scores_with_sink = torch.cat((sink.float()[:, None], logits), dim=1)
        probabilities = torch.softmax(scores_with_sink, dim=-1)[:, 1:]
        outputs.append(torch.einsum("hn,nd->hd", probabilities, selected))
    return torch.stack(outputs).to(torch.bfloat16)


def test_int4_sparse_decode_speculative_rows_extra_cache_sink_and_graph_replay() -> (
    None
):
    """Exercise the exact multi-row target-verification consumer geometry."""

    _require_sm80_or_newer()
    generator = torch.Generator().manual_seed(113)
    swa_page_size = 8
    extra_page_size = 2
    swa_keys = (
        torch.randn((24, HEAD_DIM), generator=generator, dtype=torch.float32)
        .clamp_(-3.0, 3.0)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    extra_keys = (
        torch.randn((8, HEAD_DIM), generator=generator, dtype=torch.float32)
        .clamp_(-3.0, 3.0)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    swa_cache = torch.full(
        (3, int4_main_page_bytes(swa_page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    extra_cache = torch.full(
        (4, int4_main_page_bytes(extra_page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    quantize_dsv4_int4_cache_paged(
        swa_keys,
        swa_cache,
        torch.arange(24, dtype=torch.int32, device="cuda"),
        page_size=swa_page_size,
    )
    quantize_dsv4_int4_cache_paged(
        extra_keys,
        extra_cache,
        torch.arange(8, dtype=torch.int32, device="cuda"),
        page_size=extra_page_size,
    )

    # Five query rows model one target-verification batch.  Metadata is padded
    # independently for SWA and compressed selections, as in the graph runner.
    query = torch.randn((5, 16, HEAD_DIM), generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.bfloat16
    )
    swa_indices = torch.tensor(
        [
            [0, 3, 7, 8, 12, 17, -1, -1],
            [1, 2, 9, 10, -1, -1, -1, -1],
            [4, 5, 6, 13, 18, 23, -1, -1],
            [11, 20, -1, -1, -1, -1, -1, -1],
            [0, 8, 16, 21, 22, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    swa_lengths = torch.tensor([6, 4, 6, 2, 5], dtype=torch.int32, device="cuda")
    extra_indices = torch.tensor(
        [
            [0, 7, -1, -1],
            [2, -1, -1, -1],
            [1, 3, 6, -1],
            [-1, -1, -1, -1],
            [4, 5, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    extra_lengths = torch.tensor([2, 1, 3, 0, 2], dtype=torch.int32, device="cuda")
    sink = torch.linspace(-0.5, 0.5, 16, dtype=torch.float32, device="cuda")
    output = torch.empty_like(query)
    scale = HEAD_DIM**-0.5

    def run() -> None:
        decode_sparse_attention_int4(
            q=query,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lengths,
            scale=scale,
            attn_sink=sink,
            out=output,
            swa_block_size=swa_page_size,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lengths,
            extra_block_size=extra_page_size,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    query.copy_(
        torch.randn(query.shape, generator=generator, dtype=torch.float32).to(
            device="cuda", dtype=torch.bfloat16
        )
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference_with_extra_and_sink(
        query,
        swa_keys,
        swa_indices,
        swa_lengths,
        extra_keys,
        extra_indices,
        extra_lengths,
        sink,
        scale,
    )
    torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=2.0e-2)
