from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.oscar_int2_decode import (
    BF16_BYTES_PER_TOKEN,
    SPLIT_HISTORY_MAX_PARTIAL_ROWS,
    SPLIT_HISTORY_SPLIT_MAP,
    SPLIT_HISTORY_WORKSPACE_BYTES,
    OscarInt2SplitHistoryWorkspace,
    decode_sparse_attention_oscar_int2,
    oscar_int2_split_history_count,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    CLIP_PER_ROW_QUANTILE,
    GROUP_SIZE,
    HEAD_DIM,
    NOPE_DIM,
    NUM_GROUPS,
    STORAGE_BYTES_PER_TOKEN,
    OscarInt2Calibration,
    quantize_dsv4_oscar_int2_cache_paged,
    validate_oscar_int2_calibration,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("OSCAR-INT2 decode targets NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("OSCAR-INT2 decode is fail-closed to SM86")


def _calibration() -> OscarInt2Calibration:
    generator = torch.Generator().manual_seed(20260803)
    permutation = torch.randperm(NOPE_DIM, generator=generator)
    signs = torch.where(
        torch.arange(NOPE_DIM) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    rotation = torch.zeros((NOPE_DIM, NOPE_DIM), dtype=torch.bfloat16)
    rotation[torch.arange(NOPE_DIM), permutation] = signs.to(torch.bfloat16)
    ratio = 0.95
    clip_ratios = torch.full((NUM_GROUPS,), ratio, dtype=torch.float32, device="cuda")
    clip_indices = torch.full(
        (NUM_GROUPS,),
        min(int(ratio * GROUP_SIZE), GROUP_SIZE - 1),
        dtype=torch.int16,
        device="cuda",
    )
    return validate_oscar_int2_calibration(
        rotation.cuda(),
        torch.full((NUM_GROUPS,), 2.5, dtype=torch.float32, device="cuda"),
        clip_ratios,
        clip_indices,
        layer_id=2,
        clip_mode=CLIP_PER_ROW_QUANTILE,
        clip_provenance="decode-unit-test-calibration",
    )


def _values(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn((rows, HEAD_DIM), generator=generator)
        .clamp(-2.25, 2.25)
        .to(torch.bfloat16)
        .cuda()
    )


def _rotate_full(
    values: torch.Tensor, calibration: OscarInt2Calibration
) -> torch.Tensor:
    output = torch.empty_like(values)
    output[:, :NOPE_DIM] = (
        values[:, :NOPE_DIM].float() @ calibration.rotation.float()
    ).to(torch.bfloat16)
    output[:, NOPE_DIM:] = values[:, NOPE_DIM:]
    return output


def _reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    *,
    scale: float,
    sink: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("thd,tkd->thk", q.float(), keys.float()) * scale
    sink_scores = sink.float().view(1, -1, 1).expand(q.shape[0], -1, 1)
    probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)
    return torch.einsum("thk,tkd->thd", probabilities[..., :-1], keys.float()).to(
        torch.bfloat16
    )


def _split_workspace(device: str = "cuda") -> OscarInt2SplitHistoryWorkspace:
    storage = torch.empty(
        SPLIT_HISTORY_WORKSPACE_BYTES // torch.float32.itemsize,
        dtype=torch.float32,
        device=device,
    )
    return OscarInt2SplitHistoryWorkspace.from_backend_storage(storage)


def test_split_history_selector_covers_all_capture_small_batches() -> None:
    assert SPLIT_HISTORY_SPLIT_MAP == {
        1: 16,
        2: 16,
        3: 8,
        4: 4,
        5: 4,
        6: 4,
        7: 4,
        8: 2,
    }
    for num_tokens, split_count in SPLIT_HISTORY_SPLIT_MAP.items():
        assert oscar_int2_split_history_count(num_tokens) == split_count
        assert num_tokens * split_count <= SPLIT_HISTORY_MAX_PARTIAL_ROWS
    assert oscar_int2_split_history_count(0) is None
    assert oscar_int2_split_history_count(9) is None
    with pytest.raises(TypeError, match="integer"):
        oscar_int2_split_history_count(1.0)  # type: ignore[arg-type]


def test_split_history_workspace_is_exact_and_address_stable_on_cpu() -> None:
    workspace = _split_workspace(device="cpu")

    assert workspace.storage.numel() * workspace.storage.element_size() == 4_210_688
    assert workspace.accumulator.shape == (32, 64, HEAD_DIM)
    assert workspace.maxima.shape == workspace.sums.shape == (32, 64)
    assert workspace.fixed_data_ptr == workspace.storage.data_ptr()
    workspace.validate(device=torch.device("cpu"))

    wrong_storage = torch.empty(
        SPLIT_HISTORY_WORKSPACE_BYTES // torch.float32.itemsize - 1,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="exactly 4210688 bytes"):
        OscarInt2SplitHistoryWorkspace.from_backend_storage(wrong_storage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_history_workspace_accepts_rank_local_unindexed_cuda_alias() -> None:
    with torch.cuda.device(0):
        workspace = _split_workspace("cuda:0")
        workspace.validate(device=torch.device("cuda"))

        if torch.cuda.device_count() > 1:
            with pytest.raises(ValueError, match="wrong device"):
                workspace.validate(device=torch.device("cuda:1"))


def test_sm86_mixed_oscar_decode_matches_explicit_two_tier_reference() -> None:
    _require_sm86()
    calibration = _calibration()
    page_size = 2
    extra_values = _values(4, 11)
    extra_storage = torch.zeros(
        (2, page_size * STORAGE_BYTES_PER_TOKEN), dtype=torch.uint8, device="cuda"
    )
    locations = torch.arange(4, dtype=torch.int32, device="cuda")
    quantize_dsv4_oscar_int2_cache_paged(
        extra_values,
        calibration,
        extra_storage,
        locations,
        page_size=page_size,
    )

    from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
        dequantize_dsv4_oscar_int2_cache_paged,
    )

    extra_decoded = torch.empty_like(extra_values)
    dequantize_dsv4_oscar_int2_cache_paged(
        extra_storage, locations, extra_decoded, page_size=page_size
    )

    swa_values = _rotate_full(_values(4, 17), calibration)
    swa_storage = swa_values.reshape(2, page_size * HEAD_DIM).view(torch.uint8)
    assert swa_storage.shape[1] == page_size * BF16_BYTES_PER_TOKEN

    tokens, heads = 2, 8
    q_original = _values(tokens * heads, 23).reshape(tokens, heads, HEAD_DIM)
    q = _rotate_full(q_original.reshape(-1, HEAD_DIM), calibration).reshape_as(
        q_original
    )
    out = torch.empty_like(q)
    extra_indices = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32, device="cuda")
    swa_indices = torch.tensor([[1, 2], [0, 3]], dtype=torch.int32, device="cuda")
    extra_lens = torch.full((tokens,), 2, dtype=torch.int32, device="cuda")
    swa_lens = torch.full((tokens,), 2, dtype=torch.int32, device="cuda")
    sink = torch.linspace(-1.2, -0.4, heads, dtype=torch.float32, device="cuda")
    softmax_scale = HEAD_DIM**-0.5

    decode_sparse_attention_oscar_int2(
        q,
        swa_storage,
        swa_indices,
        swa_lens,
        softmax_scale,
        sink,
        out,
        page_size,
        extra_cache=extra_storage,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_block_size=page_size,
    )
    torch.cuda.synchronize()

    selected = torch.stack(
        [
            torch.cat((extra_decoded[extra_indices[row]], swa_values[swa_indices[row]]))
            for row in range(tokens)
        ]
    )
    expected = _reference(q, selected, scale=softmax_scale, sink=sink)
    torch.testing.assert_close(out, expected, rtol=2.0e-2, atol=2.0e-2)


def test_sm86_mixed_oscar_decode_replays_in_cuda_graph() -> None:
    _require_sm86()
    calibration = _calibration()
    page_size = 2
    extra_values = _values(4, 31)
    extra_storage = torch.zeros(
        (2, page_size * STORAGE_BYTES_PER_TOKEN), dtype=torch.uint8, device="cuda"
    )
    locations = torch.arange(4, dtype=torch.int32, device="cuda")
    quantize_dsv4_oscar_int2_cache_paged(
        extra_values,
        calibration,
        extra_storage,
        locations,
        page_size=page_size,
    )
    swa_values = _rotate_full(_values(4, 37), calibration)
    swa_storage = swa_values.reshape(2, page_size * HEAD_DIM).view(torch.uint8)
    q = _rotate_full(_values(64, 41), calibration).reshape(1, 64, HEAD_DIM)
    out = torch.empty_like(q)
    indices = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
    sink = torch.zeros(64, dtype=torch.float32, device="cuda")
    workspace = _split_workspace()

    def run() -> None:
        decode_sparse_attention_oscar_int2(
            q,
            swa_storage,
            indices,
            lengths,
            HEAD_DIM**-0.5,
            sink,
            out,
            page_size,
            extra_cache=extra_storage,
            extra_indices=indices,
            extra_lens=lengths,
            extra_block_size=page_size,
            split_workspace=workspace,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.copy_(_rotate_full(_values(64, 43), calibration).reshape_as(q))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("tokens", [1, 4, 5, 6, 7, 8])
def test_sm86_split_history_matches_explicit_two_tier_reference(tokens: int) -> None:
    _require_sm86()
    calibration = _calibration()
    extra_count = 32
    extra_page_size = 16
    extra_values = _values(extra_count, 101 + tokens)
    extra_storage = torch.zeros(
        (
            extra_count // extra_page_size,
            extra_page_size * STORAGE_BYTES_PER_TOKEN,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    extra_locations = torch.arange(extra_count, dtype=torch.int32, device="cuda")
    quantize_dsv4_oscar_int2_cache_paged(
        extra_values,
        calibration,
        extra_storage,
        extra_locations,
        page_size=extra_page_size,
    )
    from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
        dequantize_dsv4_oscar_int2_cache_paged,
    )

    extra_decoded = torch.empty_like(extra_values)
    dequantize_dsv4_oscar_int2_cache_paged(
        extra_storage,
        extra_locations,
        extra_decoded,
        page_size=extra_page_size,
    )

    swa_count = 16
    swa_page_size = 16
    swa_values = _rotate_full(_values(swa_count, 151 + tokens), calibration)
    swa_storage = swa_values.reshape(1, swa_page_size * HEAD_DIM).view(torch.uint8)
    heads = 64
    q = _rotate_full(
        _values(tokens * heads, 201 + tokens), calibration
    ).reshape(tokens, heads, HEAD_DIM)
    out = torch.empty_like(q)
    extra_indices = extra_locations.unsqueeze(0).expand(tokens, -1).contiguous()
    swa_indices = (
        torch.arange(swa_count, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(tokens, -1)
        .contiguous()
    )
    extra_lens = torch.full(
        (tokens,), extra_count, dtype=torch.int32, device="cuda"
    )
    swa_lens = torch.full((tokens,), swa_count, dtype=torch.int32, device="cuda")
    sink = torch.linspace(-1.0, 0.0, heads, dtype=torch.float32, device="cuda")
    scale = HEAD_DIM**-0.5
    workspace = _split_workspace()

    decode_sparse_attention_oscar_int2(
        q,
        swa_storage,
        swa_indices,
        swa_lens,
        scale,
        sink,
        out,
        swa_page_size,
        extra_cache=extra_storage,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_block_size=extra_page_size,
        split_workspace=workspace,
    )
    torch.cuda.synchronize()

    selected = torch.cat((extra_decoded, swa_values)).unsqueeze(0).expand(
        tokens, -1, -1
    )
    expected = _reference(q, selected, scale=scale, sink=sink)
    torch.testing.assert_close(out, expected, rtol=2.0e-2, atol=2.0e-2)


def test_sm86_mixed_oscar_decode_matches_c4_production_shape() -> None:
    """Exercise H64, C4 top-512, and the full protected SWA-128 window."""

    _require_sm86()
    calibration = _calibration()
    extra_count = 512
    extra_page_size = 64
    extra_values = _values(extra_count, 47)
    extra_storage = torch.zeros(
        (
            extra_count // extra_page_size,
            extra_page_size * STORAGE_BYTES_PER_TOKEN,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    extra_locations = torch.arange(extra_count, dtype=torch.int32, device="cuda")
    quantize_dsv4_oscar_int2_cache_paged(
        extra_values,
        calibration,
        extra_storage,
        extra_locations,
        page_size=extra_page_size,
    )
    from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
        dequantize_dsv4_oscar_int2_cache_paged,
    )

    extra_decoded = torch.empty_like(extra_values)
    dequantize_dsv4_oscar_int2_cache_paged(
        extra_storage,
        extra_locations,
        extra_decoded,
        page_size=extra_page_size,
    )

    swa_count = 128
    swa_page_size = 256
    swa_values = _rotate_full(_values(swa_count, 53), calibration)
    swa_storage = torch.zeros(
        (1, swa_page_size * BF16_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device="cuda",
    )
    swa_storage[:, : swa_count * BF16_BYTES_PER_TOKEN].copy_(
        swa_values.reshape(1, -1).view(torch.uint8)
    )
    heads = 64
    q = _rotate_full(_values(heads, 59), calibration).reshape(1, heads, HEAD_DIM)
    out = torch.empty_like(q)
    extra_indices = extra_locations.unsqueeze(0)
    swa_indices = torch.arange(swa_count, dtype=torch.int32, device="cuda").unsqueeze(0)
    extra_lens = torch.tensor([extra_count], dtype=torch.int32, device="cuda")
    swa_lens = torch.tensor([swa_count], dtype=torch.int32, device="cuda")
    sink = torch.linspace(-1.0, 0.0, heads, dtype=torch.float32, device="cuda")
    scale = HEAD_DIM**-0.5

    decode_sparse_attention_oscar_int2(
        q,
        swa_storage,
        swa_indices,
        swa_lens,
        scale,
        sink,
        out,
        swa_page_size,
        extra_cache=extra_storage,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        extra_block_size=extra_page_size,
    )
    torch.cuda.synchronize()

    selected = torch.cat((extra_decoded, swa_values)).unsqueeze(0)
    expected = _reference(q, selected, scale=scale, sink=sink)
    torch.testing.assert_close(out, expected, rtol=2.0e-2, atol=2.0e-2)
