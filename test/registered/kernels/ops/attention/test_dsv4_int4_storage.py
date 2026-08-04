from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    compress_norm_rope_store,
)
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_k_norm_rope_flashmla,
)
from sglang.kernels.ops.attention.dsv4.int4_storage import (
    GROUP_SIZE,
    HEAD_DIM,
    LOGICAL_BYTES_PER_TOKEN,
    NOPE_DIM,
    NUM_GROUPS,
    PACKED_NOPE_BYTES,
    PADDING_BYTES_PER_TOKEN,
    ROPE_DIM,
    ROPE_OFFSET_BYTES,
    SCALE_OFFSET_BYTES,
    STORAGE_BYTES_PER_TOKEN,
    apply_block_hadamard_nope_reference,
    dequantize_dsv4_int4_reference,
    dequantize_dsv4_int4_storage,
    int4_main_page_bytes,
    quantize_dsv4_int4_cache_paged,
    quantize_dsv4_int4_reference,
    quantize_dsv4_int4_storage,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("the experimental path currently targets NVIDIA CUDA")


def _make_values(num_tokens: int, *, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(
        (num_tokens, HEAD_DIM),
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    values[:, :NOPE_DIM].clamp_(-3.0, 3.0)
    return values.to(device)


def _identity_freqs(num_positions: int, *, device: str) -> torch.Tensor:
    interleaved = torch.zeros((num_positions, ROPE_DIM), device=device)
    interleaved[:, 0::2] = 1.0
    return torch.view_as_complex(interleaved.view(num_positions, ROPE_DIM // 2, 2))


def test_int4_layout_has_precise_logical_size_and_alignment() -> None:
    assert PACKED_NOPE_BYTES == 224
    assert ROPE_OFFSET_BYTES == 224
    assert SCALE_OFFSET_BYTES == 352
    assert LOGICAL_BYTES_PER_TOKEN == 366
    assert STORAGE_BYTES_PER_TOKEN == 368
    assert PADDING_BYTES_PER_TOKEN == 2
    assert STORAGE_BYTES_PER_TOKEN % 16 == 0
    assert NUM_GROUPS == 7
    assert GROUP_SIZE == 64
    assert ROPE_DIM == 64
    assert int4_main_page_bytes(128) == 128 * 368


def test_reference_known_signed_nibbles_and_exact_rope() -> None:
    values = torch.zeros((1, HEAD_DIM), dtype=torch.bfloat16)
    values[0, 0] = 7.0
    values[0, 1] = -7.0
    values[0, 2] = -1.0
    values[0, 3] = 1.0
    values[0, NOPE_DIM:] = torch.linspace(-2.0, 2.0, ROPE_DIM, dtype=torch.bfloat16)

    storage = quantize_dsv4_int4_reference(values)
    decoded = dequantize_dsv4_int4_reference(storage)

    # Low nibble is +7; high nibble is two's-complement -7 (0b1001).
    assert storage[0, 0].item() == 0x97
    # Low nibble is -1 (0b1111); high nibble is +1.
    assert storage[0, 1].item() == 0x1F
    torch.testing.assert_close(decoded[0, :4], values[0, :4], rtol=0, atol=0)
    assert torch.equal(decoded[0, NOPE_DIM:], values[0, NOPE_DIM:])
    assert torch.count_nonzero(storage[0, LOGICAL_BYTES_PER_TOKEN:]) == 0


def test_reference_error_is_bounded_by_half_a_quantization_step() -> None:
    values = _make_values(11, seed=11, device="cpu")
    storage = quantize_dsv4_int4_reference(values)
    decoded = dequantize_dsv4_int4_reference(storage)
    scales = storage[:, SCALE_OFFSET_BYTES:LOGICAL_BYTES_PER_TOKEN].view(torch.bfloat16)
    error = (decoded[:, :NOPE_DIM].float() - values[:, :NOPE_DIM].float()).abs()
    error = error.reshape(values.shape[0], NUM_GROUPS, GROUP_SIZE)

    # BF16 scale and output rounding add less than 2% of a quantization step
    # for this representative range; nearest INT4 rounding still dominates.
    assert torch.all(error <= scales.float().unsqueeze(-1) * 0.52 + 1.0e-3)
    assert torch.equal(decoded[:, NOPE_DIM:], values[:, NOPE_DIM:])


def test_normalized_block_hadamard_preserves_nope_dot_products() -> None:
    generator = torch.Generator().manual_seed(21)
    queries = torch.randn((3, NOPE_DIM), generator=generator)
    keys = torch.randn((5, NOPE_DIM), generator=generator)

    expected = queries @ keys.T
    transformed_queries = apply_block_hadamard_nope_reference(queries)
    transformed_keys = apply_block_hadamard_nope_reference(keys)
    actual = transformed_queries @ transformed_keys.T

    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_full_geometry_hadamard_leaves_rope_unchanged() -> None:
    values = _make_values(3, seed=29, device="cpu")
    transformed = apply_block_hadamard_nope_reference(values)
    assert torch.equal(transformed[:, NOPE_DIM:], values[:, NOPE_DIM:].float())


def test_cuda_int4_writer_and_decoder_match_references_with_scatter() -> None:
    _require_cuda()
    values = _make_values(5, seed=31, device="cuda")
    locations = torch.tensor([7, 0, -1, 4, 2], dtype=torch.int32, device="cuda")
    storage = torch.full(
        (8, STORAGE_BYTES_PER_TOKEN), 0xA5, dtype=torch.uint8, device="cuda"
    )
    untouched_before = storage[6].clone()

    quantize_dsv4_int4_storage(values, storage=storage, locations=locations)
    valid = locations >= 0
    gathered_locations = locations[valid]
    gathered = dequantize_dsv4_int4_storage(
        storage,
        locations=gathered_locations,
        output=torch.empty(
            (valid.sum().item(), HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        ),
    )
    torch.cuda.synchronize()

    reference_storage = quantize_dsv4_int4_reference(values[valid])
    for reference_row, cache_row in enumerate(gathered_locations.cpu().tolist()):
        assert torch.equal(
            storage[cache_row, :LOGICAL_BYTES_PER_TOKEN],
            reference_storage[reference_row, :LOGICAL_BYTES_PER_TOKEN],
        )
        assert torch.all(storage[cache_row, LOGICAL_BYTES_PER_TOKEN:] == 0xA5)
    reference_decoded = dequantize_dsv4_int4_reference(reference_storage)
    torch.testing.assert_close(gathered, reference_decoded, rtol=0, atol=0)
    assert torch.equal(storage[6], untouched_before)


def test_cuda_contiguous_int4_pack_and_decode_match_reference() -> None:
    _require_cuda()
    values = _make_values(9, seed=37, device="cuda")
    storage = torch.full(
        (values.shape[0], STORAGE_BYTES_PER_TOKEN),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    output = torch.empty_like(values)

    quantize_dsv4_int4_storage(values, storage=storage)
    dequantize_dsv4_int4_storage(storage, output=output)
    torch.cuda.synchronize()

    reference_storage = quantize_dsv4_int4_reference(values)
    reference_output = dequantize_dsv4_int4_reference(reference_storage)
    assert torch.equal(
        storage[:, :LOGICAL_BYTES_PER_TOKEN],
        reference_storage[:, :LOGICAL_BYTES_PER_TOKEN],
    )
    assert torch.all(storage[:, LOGICAL_BYTES_PER_TOKEN:] == 0xA5)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)


def test_cuda_int4_pack_and_decode_replay_in_real_cuda_graph() -> None:
    _require_cuda()
    values = _make_values(4, seed=41, device="cuda")
    locations = torch.tensor([0, 2, 5, 7], dtype=torch.int32, device="cuda")
    storage = torch.full(
        (8, STORAGE_BYTES_PER_TOKEN), 0xA5, dtype=torch.uint8, device="cuda"
    )
    output = torch.empty((4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    # Compile Triton and initialize its launch machinery before capture.
    quantize_dsv4_int4_storage(values, storage=storage, locations=locations)
    dequantize_dsv4_int4_storage(storage, locations=locations, output=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_dsv4_int4_storage(values, storage=storage, locations=locations)
        dequantize_dsv4_int4_storage(storage, locations=locations, output=output)

    next_values = _make_values(4, seed=43, device="cuda")
    values.copy_(next_values)
    locations.copy_(torch.tensor([1, 3, 4, 6], dtype=torch.int32, device="cuda"))
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()

    reference_storage = quantize_dsv4_int4_reference(next_values)
    reference_output = dequantize_dsv4_int4_reference(reference_storage)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    for cache_row in locations.cpu().tolist():
        assert torch.all(storage[cache_row, LOGICAL_BYTES_PER_TOKEN:] == 0xA5)


@pytest.mark.parametrize("writer", ["triton", "cuda_jit"])
def test_paged_int4_writers_match_reference(writer: str) -> None:
    _require_cuda()
    page_size = 8
    values = _make_values(6, seed=47, device="cuda").contiguous()
    locations = torch.tensor([0, 3, 8, 11, -1, 15], dtype=torch.int32, device="cuda")
    cache = torch.full(
        (2, int4_main_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    if writer == "triton":
        quantize_dsv4_int4_cache_paged(values, cache, locations, page_size=page_size)
    else:
        fused_store_cache(
            values,
            cache,
            locations,
            page_size=page_size,
            type="flashmla",
            int4_store=True,
        )
    torch.cuda.synchronize()

    valid = locations >= 0
    expected = quantize_dsv4_int4_reference(values[valid])
    for expected_row, location in enumerate(locations[valid].cpu().tolist()):
        page, in_page = divmod(location, page_size)
        start = in_page * STORAGE_BYTES_PER_TOKEN
        actual = cache[page, start : start + STORAGE_BYTES_PER_TOKEN]
        assert torch.equal(
            actual[:LOGICAL_BYTES_PER_TOKEN],
            expected[expected_row, :LOGICAL_BYTES_PER_TOKEN],
        )
        assert torch.all(actual[LOGICAL_BYTES_PER_TOKEN:] == 0xA5)


def test_paged_int4_writer_and_gather_decoder_replay_in_cuda_graph() -> None:
    _require_cuda()
    page_size = 8
    values = _make_values(4, seed=53, device="cuda")
    locations = torch.tensor([1, 6, 9, 14], dtype=torch.int32, device="cuda")
    cache = torch.full(
        (2, int4_main_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    output = torch.empty((4, 1, HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    def run() -> None:
        quantize_dsv4_int4_cache_paged(values, cache, locations, page_size=page_size)
        dequantize_k_cache_paged(
            cache,
            locations,
            page_size,
            out=output,
            is_int4=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    next_values = _make_values(4, seed=59, device="cuda")
    values.copy_(next_values)
    graph.replay()
    torch.cuda.synchronize()
    expected = dequantize_dsv4_int4_reference(quantize_dsv4_int4_reference(next_values))
    torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)


def test_paged_int4_writer_reuses_boundary_slot_without_stale_payload() -> None:
    _require_cuda()
    page_size = 8
    location = torch.tensor([page_size], dtype=torch.int32, device="cuda")
    first = _make_values(1, seed=61, device="cuda")
    second = _make_values(1, seed=67, device="cuda")
    cache = torch.full(
        (2, int4_main_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    quantize_dsv4_int4_cache_paged(first, cache, location, page_size=page_size)
    quantize_dsv4_int4_cache_paged(second, cache, location, page_size=page_size)
    output = torch.empty((1, 1, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    dequantize_k_cache_paged(
        cache,
        location,
        page_size,
        out=output,
        is_int4=True,
    )
    torch.cuda.synchronize()

    expected = dequantize_dsv4_int4_reference(quantize_dsv4_int4_reference(second))
    torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)


def test_int4_main_norm_rope_production_writer_matches_reference() -> None:
    _require_cuda()
    page_size = 8
    eps = 1.0e-6
    values = _make_values(4, seed=71, device="cuda")
    weight = _make_values(1, seed=73, device="cuda")[0]
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    locations = torch.tensor([0, 7, 8, 15], dtype=torch.int32, device="cuda")
    freqs_cis = _identity_freqs(4, device="cuda")
    cache = torch.full(
        (2, int4_main_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    fused_k_norm_rope_flashmla(
        kv=values,
        kv_weight=weight,
        eps=eps,
        freqs_cis=freqs_cis,
        positions=positions,
        out_loc=locations,
        kvcache=cache,
        page_size=page_size,
        bf16_store=False,
        int4_store=True,
    )
    output = torch.empty((4, 1, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    dequantize_k_cache_paged(
        cache,
        locations,
        page_size,
        out=output,
        is_int4=True,
    )
    expected = values.float()
    expected *= torch.rsqrt(expected.square().mean(dim=-1, keepdim=True) + eps)
    expected *= weight.float()
    expected_bf16 = expected.to(torch.bfloat16).float()
    torch.cuda.synchronize()

    scales = []
    for location in locations.cpu().tolist():
        page, in_page = divmod(location, page_size)
        token_start = in_page * STORAGE_BYTES_PER_TOKEN
        scales.append(
            cache[
                page,
                token_start + SCALE_OFFSET_BYTES : token_start
                + LOGICAL_BYTES_PER_TOKEN,
            ].view(torch.bfloat16)
        )
    scale_matrix = torch.stack(scales).float().repeat_interleave(GROUP_SIZE, dim=1)
    quantization_error = (
        output[:, 0, :NOPE_DIM].float() - expected_bf16[:, :NOPE_DIM]
    ).abs()
    assert torch.all(quantization_error <= scale_matrix * 0.52 + 1.0e-3)
    torch.testing.assert_close(
        output[:, 0, NOPE_DIM:].float(),
        expected_bf16[:, NOPE_DIM:],
        rtol=0.01,
        atol=0.02,
    )


def test_int4_compressor_writer_matches_bf16_and_replays_cuda_graph() -> None:
    """Qualify the C4/C128 compressor store used by decode/verification."""

    _require_cuda()
    page_size = 8
    compress_ratio = 4
    num_tokens = 4
    eps = 1.0e-6
    values = _make_values(num_tokens, seed=79, device="cuda")
    weight = _make_values(1, seed=83, device="cuda")[0]
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
    locations = torch.tensor([0, 7, 8, 15], dtype=torch.int64, device="cuda")
    freqs_cis = _identity_freqs(int(seq_lens.max().item()), device="cuda")
    int4_cache = torch.full(
        (2, int4_main_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    int4_output = torch.empty(
        (num_tokens, 1, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )

    def run_int4() -> None:
        compress_norm_rope_store(
            values,
            plan,
            norm_weight=weight,
            norm_eps=eps,
            freq_cis=freqs_cis,
            out_loc=locations,
            kvcache=int4_cache,
            page_size=page_size,
            int4_store=True,
        )
        dequantize_k_cache_paged(
            int4_cache,
            locations,
            page_size,
            out=int4_output,
            is_int4=True,
        )

    run_int4()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_int4()

    values.copy_(_make_values(num_tokens, seed=89, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    # BF16 pages are token-major and unpadded.  This deliberately spans two
    # pages so a stale FP8-style 576-byte rounding stride cannot pass.
    bf16_page_bytes = page_size * HEAD_DIM * 2
    bf16_cache = torch.zeros((2, bf16_page_bytes), dtype=torch.uint8, device="cuda")
    bf16_output = torch.empty_like(int4_output)
    compress_norm_rope_store(
        values,
        plan,
        norm_weight=weight,
        norm_eps=eps,
        freq_cis=freqs_cis,
        out_loc=locations,
        kvcache=bf16_cache,
        page_size=page_size,
        bf16_store=True,
    )
    dequantize_k_cache_paged(
        bf16_cache,
        locations,
        page_size,
        out=bf16_output,
        is_bf16=True,
    )
    torch.cuda.synchronize()

    scales = []
    for location in locations.cpu().tolist():
        page, in_page = divmod(location, page_size)
        token_start = in_page * STORAGE_BYTES_PER_TOKEN
        scales.append(
            int4_cache[
                page,
                token_start + SCALE_OFFSET_BYTES : token_start
                + LOGICAL_BYTES_PER_TOKEN,
            ].view(torch.bfloat16)
        )
    scale_matrix = torch.stack(scales).float().repeat_interleave(GROUP_SIZE, dim=1)
    quantization_error = (
        int4_output[:, 0, :NOPE_DIM].float() - bf16_output[:, 0, :NOPE_DIM].float()
    ).abs()
    assert torch.all(quantization_error <= scale_matrix * 0.52 + 1.0e-3)
    torch.testing.assert_close(
        int4_output[:, :, NOPE_DIM:],
        bf16_output[:, :, NOPE_DIM:],
        rtol=0,
        atol=0,
    )
