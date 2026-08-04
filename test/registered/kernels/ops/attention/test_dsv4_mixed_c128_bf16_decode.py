from __future__ import annotations

import math

import pytest
import torch
from sglang.kernels.ops.attention.dsv4 import fused_store_cache
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    prime_e4m3fn_decode_lut,
)
from sglang.srt.layers.attention.nsa.v4_mixed_c128_bf16_kernel import (
    decode_sparse_attention_fp8_swa_bf16_extra,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
NOPE_DIM = 448
SWA_PAGE_SIZE = 128
C128_PAGE_SIZE = 2


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("selective C128 storage targets NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("selective C128 storage is exact-SM86-only")


def _fp8_page_bytes(page_size: int) -> int:
    return math.ceil(page_size * 584 / 576) * 576


def _make_caches(seed: int = 211) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    swa_values = torch.randn(
        (2 * SWA_PAGE_SIZE, HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device="cuda", dtype=torch.bfloat16)
    swa_cache = torch.full(
        (2, _fp8_page_bytes(SWA_PAGE_SIZE)),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    swa_locations = torch.arange(
        swa_values.shape[0], device="cuda", dtype=torch.int32
    )
    fused_store_cache(
        swa_values,
        swa_cache,
        swa_locations,
        page_size=SWA_PAGE_SIZE,
        type="flashmla",
    )
    swa_decoded = dequantize_k_cache_paged(
        swa_cache, swa_locations, SWA_PAGE_SIZE
    )[:, 0]

    extra_values = torch.randn(
        (8, HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device="cuda", dtype=torch.bfloat16)
    extra_cache = torch.full(
        (4, C128_PAGE_SIZE * HEAD_DIM * 2),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    extra_cache.view(torch.bfloat16).reshape(4, C128_PAGE_SIZE, HEAD_DIM).copy_(
        extra_values.view(4, C128_PAGE_SIZE, HEAD_DIM)
    )
    return swa_cache, swa_decoded, extra_cache, extra_values


def _reference(
    q: torch.Tensor,
    swa_keys: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_keys: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lens: torch.Tensor,
    sink: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    output = []
    for row in range(q.shape[0]):
        selected_extra = extra_keys[
            extra_indices[row, : int(extra_lens[row].item())].long()
        ]
        selected_swa = swa_keys[
            swa_indices[row, : int(swa_lens[row].item())].long()
        ]
        selected = torch.cat((selected_extra, selected_swa), dim=0).float()
        logits = torch.einsum("hd,nd->hn", q[row].float(), selected) * scale
        if sink is None:
            probabilities = logits.softmax(dim=-1)
        else:
            logits = torch.cat((sink.float()[:, None], logits), dim=-1)
            probabilities = logits.softmax(dim=-1)[:, 1:]
        output.append(torch.einsum("hn,nd->hd", probabilities, selected))
    return torch.stack(output).to(torch.bfloat16)


def test_sm86_mixed_c128_speculative_rows_and_graph_replay_use_live_lengths() -> None:
    _require_sm86()
    device = torch.device("cuda", torch.cuda.current_device())
    prime_e4m3fn_decode_lut(device)
    swa_cache, swa_keys, extra_cache, extra_keys = _make_caches()
    generator = torch.Generator().manual_seed(223)
    q = torch.randn((5, 16, HEAD_DIM), generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    swa_indices = torch.tensor(
        [
            [0, 3, 127, 128, 193, 255, -1, -1],
            [1, 17, 129, 201, -1, -1, -1, -1],
            [2, 66, 130, 188, 220, -1, -1, -1],
            [8, 136, -1, -1, -1, -1, -1, -1],
            [7, 77, 177, 207, -1, -1, -1, -1],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([6, 4, 5, 2, 4], device=device, dtype=torch.int32)
    extra_indices = torch.tensor(
        [
            [0, 1, 7, -1, -1, -1, -1, -1],
            [2, 6, -1, -1, -1, -1, -1, -1],
            [1, 3, 4, 5, -1, -1, -1, -1],
            [7, -1, -1, -1, -1, -1, -1, -1],
            [0, 2, 4, 6, 7, -1, -1, -1],
        ],
        device=device,
        dtype=torch.int32,
    )
    extra_lens = torch.tensor([3, 2, 4, 1, 5], device=device, dtype=torch.int32)
    sink = torch.linspace(-0.75, 0.75, 16, device=device, dtype=torch.float32)
    out = torch.empty_like(q)
    output_pointer = out.data_ptr()
    scale = HEAD_DIM**-0.5

    def run() -> None:
        decode_sparse_attention_fp8_swa_bf16_extra(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            scale=scale,
            attn_sink=sink,
            out=out,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            swa_block_size=SWA_PAGE_SIZE,
            extra_block_size=C128_PAGE_SIZE,
        )

    run()
    torch.cuda.synchronize()
    expected = _reference(
        q,
        swa_keys,
        swa_indices,
        swa_lens,
        extra_keys,
        extra_indices,
        extra_lens,
        sink,
        scale,
    )
    torch.testing.assert_close(out, expected, rtol=3.0e-2, atol=3.0e-2)
    assert out.data_ptr() == output_pointer

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    q.copy_(
        torch.randn(q.shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
    )
    # Change C128 lengths in-place without recapture, including an empty row
    # and a row that consumes every C128 index slot.
    extra_lens.copy_(torch.tensor([0, 5, 1, 8, 2], device=device, dtype=torch.int32))
    extra_indices.copy_(
        torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 2, 3, 7, -1, -1, -1],
                [6, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [5, 7, -1, -1, -1, -1, -1, -1],
            ],
            device=device,
            dtype=torch.int32,
        )
    )
    out.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        q,
        swa_keys,
        swa_indices,
        swa_lens,
        extra_keys,
        extra_indices,
        extra_lens,
        sink,
        scale,
    )
    torch.testing.assert_close(out, expected, rtol=3.0e-2, atol=3.0e-2)
    assert out.data_ptr() == output_pointer


def test_selective_c128_decode_rejects_non_sm86(monkeypatch) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA is required")
    q = torch.empty((1, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty((1, 1024), device="cuda", dtype=torch.uint8)
    indices = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    lengths = torch.ones((1,), device="cuda", dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    with pytest.raises(RuntimeError, match="only for exact SM86"):
        decode_sparse_attention_fp8_swa_bf16_extra(
            q=q,
            swa_cache=cache,
            swa_indices=indices,
            swa_lens=lengths,
            scale=1.0,
            attn_sink=None,
            out=torch.empty_like(q),
            extra_cache=cache,
            extra_indices=indices,
            extra_lens=lengths,
            swa_block_size=1,
            extra_block_size=1,
        )
