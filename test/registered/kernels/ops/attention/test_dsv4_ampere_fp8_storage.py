from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from sglang.kernels.ops.attention.dsv4 import fused_store_cache
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
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    e4m3fn_decode_values,
    prime_e4m3fn_decode_lut,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_NOPE_DIM = 448
_ROPE_DIM = 64
_TOKEN_BYTES = 584
_VALUE_BYTES = 576
_SCALE_BYTES = 8


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("the software FP8 storage path is specific to SM86")


def _page_bytes(page_size: int) -> int:
    return math.ceil(page_size * _TOKEN_BYTES / _VALUE_BYTES) * _VALUE_BYTES


def test_e4m3fn_decode_table_matches_torch_for_every_byte() -> None:
    raw = torch.tensor(tuple(range(256)), dtype=torch.uint8)
    expected = raw.view(torch.float8_e4m3fn).float()
    actual = torch.tensor(e4m3fn_decode_values(), dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("page_size", [2, 64, 128])
def test_sm86_flashmla_store_and_byte_dequant(page_size: int) -> None:
    _require_sm86()
    torch.manual_seed(0)
    locations = torch.tensor(
        [0, page_size - 1, page_size, 2 * page_size - 1],
        dtype=torch.int32,
        device="cuda",
    )
    values = torch.randn((locations.numel(), 512), dtype=torch.bfloat16, device="cuda")
    cache = torch.full(
        (3, _page_bytes(page_size)), 0xA5, dtype=torch.uint8, device="cuda"
    )

    fused_store_cache(values, cache, locations, page_size=page_size, type="flashmla")
    decoded = dequantize_k_cache_paged(cache, locations, page_size)[:, 0]
    torch.cuda.synchronize()

    torch.testing.assert_close(
        decoded[:, _NOPE_DIM:],
        values[:, _NOPE_DIM:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        decoded[:, :_NOPE_DIM].float(),
        values[:, :_NOPE_DIM].float(),
        rtol=0.16,
        atol=0.04,
    )

    flat = cache.flatten()
    for location in locations.cpu().tolist():
        page, offset = divmod(location, page_size)
        scale_padding = (
            page * cache.stride(0)
            + page_size * _VALUE_BYTES
            + offset * _SCALE_BYTES
            + 7
        )
        assert flat[scale_padding].item() == 0xA5


@pytest.mark.parametrize("cache_type,head_dim", [("flashmla", 512), ("indexer", 128)])
def test_sm86_fused_store_skips_negative_locations(
    cache_type: str, head_dim: int
) -> None:
    _require_sm86()
    page_size = 64
    page_bytes = (
        _page_bytes(page_size)
        if cache_type == "flashmla"
        else page_size * (head_dim + 4)
    )
    cache = torch.full((1, page_bytes), 0xA5, dtype=torch.uint8, device="cuda")
    before = cache.clone()
    values = torch.randn((1, head_dim), dtype=torch.bfloat16, device="cuda")
    locations = torch.tensor([-1], dtype=torch.int32, device="cuda")

    fused_store_cache(values, cache, locations, page_size=page_size, type=cache_type)
    torch.cuda.synchronize()
    assert torch.equal(cache, before)


def test_sm86_norm_rope_writer_uses_packed_layout() -> None:
    _require_sm86()
    torch.manual_seed(1)
    page_size = 128
    eps = 1.0e-6
    values = torch.randn((3, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((512,), dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    locations = torch.tensor([0, 127, 128], dtype=torch.int32, device="cuda")
    # Interleaved real/imag identity rotation.
    freqs = torch.zeros((3, 64), dtype=torch.float32, device="cuda")
    freqs[:, 0::2] = 1.0
    freqs_cis = torch.view_as_complex(freqs.view(3, 32, 2))
    cache = torch.zeros((2, _page_bytes(page_size)), dtype=torch.uint8, device="cuda")

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
    )
    decoded = dequantize_k_cache_paged(cache, locations, page_size)[:, 0].float()
    expected = values.float()
    expected *= torch.rsqrt(expected.square().mean(dim=-1, keepdim=True) + eps)
    expected *= weight.float()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        decoded[:, _NOPE_DIM:], expected[:, _NOPE_DIM:], rtol=0.01, atol=0.02
    )
    torch.testing.assert_close(
        decoded[:, :_NOPE_DIM], expected[:, :_NOPE_DIM], rtol=0.17, atol=0.06
    )


def test_sm86_store_and_dequant_replay_in_real_cuda_graph() -> None:
    _require_sm86()
    page_size = 64
    prime_e4m3fn_decode_lut("cuda")
    values = torch.randn((2, 512), dtype=torch.bfloat16, device="cuda")
    locations = torch.tensor([0, 65], dtype=torch.int32, device="cuda")
    cache = torch.zeros((3, _page_bytes(page_size)), dtype=torch.uint8, device="cuda")
    output = torch.empty((2, 1, 512), dtype=torch.bfloat16, device="cuda")

    # Compile both kernels before capture.
    fused_store_cache(values, cache, locations, page_size=page_size, type="flashmla")
    dequantize_k_cache_paged(cache, locations, page_size, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_store_cache(
            values, cache, locations, page_size=page_size, type="flashmla"
        )
        dequantize_k_cache_paged(cache, locations, page_size, out=output)

    values.copy_(torch.randn_like(values))
    locations.copy_(torch.tensor([64, 130], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output[:, 0, _NOPE_DIM:].float(),
        values[:, _NOPE_DIM:].float(),
        rtol=0,
        atol=0,
    )


def test_sm86_selective_c128_pool_is_unpadded_bf16_and_graph_safe() -> None:
    _require_sm86()
    page_size = 2
    pool = DeepSeekV4SingleKVPool(
        size=8,
        page_size=page_size,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=_NOPE_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_bf16_cache=True,
    )
    assert pool.use_bf16_cache
    assert not pool.use_ampere_fp8_storage
    assert pool.get_bytes_per_token() == 1024
    assert pool.bytes_per_page_padded == page_size * 1024
    assert pool.kv_buffer[0].shape[1] == page_size * 1024

    values = torch.randn((4, 512), dtype=torch.bfloat16, device="cuda")
    locations = torch.tensor([0, 1, 2, 7], dtype=torch.int32, device="cuda")
    output = torch.empty((4, 1, 512), dtype=torch.bfloat16, device="cuda")

    def run() -> None:
        # The selective pool must never reach the E4M3 fused writer even when
        # legacy compressor dispatch calls the generic "fused" method.
        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.fused_store_cache",
            side_effect=AssertionError("FP8 writer reached for BF16 C128 pool"),
        ):
            pool.set_key_buffer_fused(0, locations, values)
        dequantize_k_cache_paged(
            pool.kv_buffer[0],
            locations,
            page_size,
            out=output,
            is_bf16=True,
        )

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:, 0], values, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    values.copy_(torch.randn_like(values))
    locations.copy_(torch.tensor([1, 3, 4, 6], dtype=torch.int32, device="cuda"))
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:, 0], values, rtol=0, atol=0)


def test_sm86_c128_compressor_writes_unpadded_bf16_pages_in_cuda_graph() -> None:
    """Exercise the actual fused C128 writer across physical page boundaries."""

    _require_sm86()
    torch.manual_seed(11)
    page_size = 2
    compress_ratio = 128
    num_tokens = 4
    eps = 1.0e-6
    values = torch.randn((num_tokens, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((512,), dtype=torch.bfloat16, device="cuda")
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
    locations = torch.tensor([0, 1, 2, 7], dtype=torch.int64, device="cuda")
    interleaved_freqs = torch.zeros(
        (int(seq_lens.max().item()), 64), dtype=torch.float32, device="cuda"
    )
    interleaved_freqs[:, 0::2] = 1.0
    freqs_cis = torch.view_as_complex(interleaved_freqs.view(-1, 32, 2))
    # Four exact, unpadded 2048-byte pages.  Locations touch pages 0, 1, and 3;
    # any legacy 2304-byte padded stride would write out of bounds here.
    cache = torch.full(
        (4, page_size * 512 * 2), 0xA5, dtype=torch.uint8, device="cuda"
    )
    output = torch.empty(
        (num_tokens, 1, 512), dtype=torch.bfloat16, device="cuda"
    )

    def run() -> None:
        compress_norm_rope_store(
            values,
            plan,
            norm_weight=weight,
            norm_eps=eps,
            freq_cis=freqs_cis,
            out_loc=locations,
            kvcache=cache,
            page_size=page_size,
            bf16_store=True,
        )
        dequantize_k_cache_paged(
            cache,
            locations,
            page_size,
            out=output,
            is_bf16=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    values.copy_(torch.randn_like(values))
    locations.copy_(torch.tensor([1, 2, 4, 6], dtype=torch.int64, device="cuda"))
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()

    expected = values.float()
    expected *= torch.rsqrt(expected.square().mean(dim=-1, keepdim=True) + eps)
    expected *= weight.float()
    torch.testing.assert_close(output[:, 0].float(), expected, rtol=0.01, atol=0.02)
