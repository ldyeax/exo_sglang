import math

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    prime_e4m3fn_decode_lut,
)
from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
    decode_sparse_attention_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA CUDA test"),
]

_HEAD_DIM = 512
_NOPE_DIM = 448
_VALUE_BYTES = 576
_SCALE_BYTES = 8


def _padded_page_bytes(page_size: int) -> int:
    return math.ceil(page_size * 584 / _VALUE_BYTES) * _VALUE_BYTES


def _make_packed_cache(
    page_size: int,
    num_pages: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    num_tokens = page_size * num_pages
    source = torch.randn(
        (num_tokens, _HEAD_DIM),
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    source[:, :_NOPE_DIM].clamp_(-1.75, 1.75)

    page_bytes = _padded_page_bytes(page_size)
    cache = torch.full((num_pages, page_bytes), 0xA5, dtype=torch.uint8)
    decoded = torch.empty_like(source)
    scale_exponents = (-2, -1, 0, 1, 2, -3, 3)
    for token_id in range(num_tokens):
        page = token_id // page_size
        position = token_id % page_size
        value_base = position * _VALUE_BYTES
        scale_base = page_size * _VALUE_BYTES + position * _SCALE_BYTES

        encoded_nope = torch.empty(_NOPE_DIM, dtype=torch.uint8)
        decoded_nope = torch.empty(_NOPE_DIM, dtype=torch.bfloat16)
        for group, exponent in enumerate(scale_exponents):
            group_slice = slice(group * 64, (group + 1) * 64)
            group_scale = float(2**exponent)
            encoded_group = (
                (source[token_id, group_slice] / group_scale)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            encoded_nope[group_slice] = encoded_group
            decoded_nope[group_slice] = (
                encoded_group.view(torch.float8_e4m3fn).to(torch.float32) * group_scale
            ).to(torch.bfloat16)
        cache[page, value_base : value_base + _NOPE_DIM].copy_(encoded_nope)
        cache[
            page,
            value_base + _NOPE_DIM : value_base + _VALUE_BYTES,
        ].view(torch.bfloat16).copy_(source[token_id, _NOPE_DIM:])
        cache[page, scale_base : scale_base + 7] = torch.tensor(
            [127 + exponent for exponent in scale_exponents], dtype=torch.uint8
        )
        # The eighth byte is layout padding. Keep the canary there so a bad
        # group index cannot silently read a plausible scale.

        decoded[token_id, :_NOPE_DIM] = decoded_nope
        decoded[token_id, _NOPE_DIM:] = source[token_id, _NOPE_DIM:]
    return cache.cuda(), decoded.cuda()


def _reference_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    sink: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.einsum("bhd,bkd->bhk", q.float(), keys.float()) * scale
    if sink is not None:
        scores = torch.cat(
            (sink.float().view(1, -1, 1).expand(q.shape[0], -1, -1), scores),
            dim=-1,
        )
        keys = torch.cat(
            (
                torch.zeros(
                    (keys.shape[0], 1, keys.shape[-1]),
                    dtype=keys.dtype,
                    device=keys.device,
                ),
                keys,
            ),
            dim=1,
        )
    return torch.einsum("bhk,bkd->bhd", scores.softmax(dim=-1), keys.float()).to(
        torch.bfloat16
    )


@pytest.mark.parametrize("extra_page_size", [2, 64])
def test_byte_storage_attention_reads_swa_and_compressed_padded_pages(
    extra_page_size,
):
    device = torch.device("cuda", torch.cuda.current_device())
    prime_e4m3fn_decode_lut(device)
    swa_cache, swa_keys = _make_packed_cache(128, 2, seed=1)
    extra_cache, extra_keys = _make_packed_cache(extra_page_size, 2, seed=2)
    swa_before = swa_cache.clone()
    extra_before = extra_cache.clone()

    swa_token_ids = torch.tensor([[0, 127, 128, 193]], dtype=torch.int32, device=device)
    extra_token_ids = torch.tensor(
        [[0, extra_page_size - 1, extra_page_size]],
        dtype=torch.int32,
        device=device,
    )
    swa_lens = torch.tensor([swa_token_ids.shape[1]], dtype=torch.int32, device=device)
    extra_lens = torch.tensor(
        [extra_token_ids.shape[1]], dtype=torch.int32, device=device
    )
    q = torch.randn((1, 8, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    sink = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32, device=device)
    out = torch.empty_like(q)
    scale = _HEAD_DIM**-0.5

    decode_sparse_attention_triton(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_token_ids,
        swa_lens=swa_lens,
        scale=scale,
        attn_sink=sink,
        out=out,
        extra_cache=extra_cache,
        extra_indices=extra_token_ids,
        extra_lens=extra_lens,
        swa_block_size=128,
        extra_block_size=extra_page_size,
    )

    selected_swa = swa_keys[swa_token_ids.long()]
    selected_extra = extra_keys[extra_token_ids.long()]
    reference = _reference_attention(
        q,
        torch.cat((selected_extra, selected_swa), dim=1),
        scale,
        sink,
    )
    torch.testing.assert_close(out, reference, rtol=3e-2, atol=3e-2)
    assert torch.equal(swa_cache, swa_before)
    assert torch.equal(extra_cache, extra_before)


@pytest.mark.parametrize("num_heads", [8, 64])
def test_byte_storage_attention_cuda_graph_replay_uses_primed_lut(
    num_heads: int,
):
    device = torch.device("cuda", torch.cuda.current_device())
    prime_e4m3fn_decode_lut(device)
    cache_storage, decoded_keys = _make_packed_cache(128, 2, seed=3)
    # Exercise the legacy FlashMLA-shaped native-float8 view as well as the
    # raw 2-D uint8 buffers used by the tests above. The sliced view retains
    # the allocator's padded page stride.
    cache = cache_storage[:, : 128 * 584].view(2, 128, 1, 584)
    cache = cache.view(torch.float8_e4m3fn)
    token_ids = torch.tensor([[1, 63, 129, 255]], dtype=torch.int32, device=device)
    lengths = torch.tensor([token_ids.shape[1]], dtype=torch.int32, device=device)
    static_q = torch.randn(
        (1, num_heads, _HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    out = torch.empty_like(static_q)
    scale = _HEAD_DIM**-0.5

    def run() -> None:
        decode_sparse_attention_triton(
            q=static_q,
            swa_cache=cache,
            swa_indices=token_ids,
            swa_lens=lengths,
            scale=scale,
            attn_sink=None,
            out=out,
        )

    # Compile and initialize Triton's launch machinery before capture.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    next_q = torch.randn_like(static_q)
    static_q.copy_(next_q)
    out.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize()

    reference = _reference_attention(
        next_q,
        decoded_keys[token_ids.long()],
        scale,
        None,
    )
    torch.testing.assert_close(out, reference, rtol=3e-2, atol=3e-2)
