import pytest
import torch

from sglang.srt.layers.quantization.mla_kv_b_triton import (
    _select_launch_config,
    mla_kv_b_gptq_w8a16_bmm,
    mla_kv_b_w8a16_bmm,
)


def _quantize_uint8b128(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Test-only offline quantization of logical ``[H, K, N]`` weights."""

    maximum = weight.float().amax(dim=1, keepdim=True)
    minimum = weight.float().amin(dim=1, keepdim=True)
    scales = torch.maximum(maximum.abs() / 127.0, minimum.abs() / 128.0)
    scales = scales.clamp_min(torch.finfo(torch.bfloat16).tiny).to(torch.bfloat16)
    signed = torch.round(weight.float() / scales.float()).clamp(-128, 127)
    qweight = (signed.to(torch.int16) + 128).to(torch.uint8)
    reference = ((qweight.to(torch.int16) - 128).float() * scales.float()).to(
        torch.bfloat16
    )
    return qweight, scales, reference


def _pack_gptq_rows(qweight: torch.Tensor) -> torch.Tensor:
    """Pack four uint8b128 K rows per INT32, matching GPTQ ``pack_rows``."""

    heads, input_features, output_features = qweight.shape
    assert input_features % 4 == 0
    packed = torch.zeros(
        heads,
        input_features // 4,
        output_features,
        dtype=torch.int64,
        device=qweight.device,
    )
    for byte_index in range(4):
        packed |= qweight[:, byte_index::4, :].to(torch.int64) << (8 * byte_index)
    return packed.to(torch.int32)


@pytest.mark.parametrize(
    ("tokens", "input_features", "output_features"),
    [(1, 192, 512), (3, 512, 256), (128, 192, 512)],
)
def test_compact_triton_matches_quantized_bmm(
    tokens: int,
    input_features: int,
    output_features: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 tensor cores require Ampere or newer")

    torch.manual_seed(17)
    heads = 2
    source_weight = torch.randn(
        heads,
        input_features,
        output_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    raw_qweight, scales, reference_weight = _quantize_uint8b128(source_weight)
    qweight = _pack_gptq_rows(raw_qweight)
    # Deliberately pass a non-contiguous [M, H, K] activation to exercise the
    # explicit input strides used by the three-dimensional launch.
    x = torch.randn(
        heads,
        tokens,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)

    expected = torch.bmm(x.transpose(0, 1), reference_weight)
    actual = mla_kv_b_gptq_w8a16_bmm(x, qweight, scales)

    assert actual.shape == (heads, tokens, output_features)
    assert actual.dtype == torch.bfloat16
    assert qweight.shape == (heads, input_features // 4, output_features)
    assert qweight.element_size() * qweight.numel() == raw_qweight.numel()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=8e-2)


def test_launch_config_covers_decode_mtp_and_prefill_shapes() -> None:
    assert _select_launch_config(1).block_m == 16
    assert _select_launch_config(3).block_m == 16
    assert _select_launch_config(32).block_m == 32
    assert _select_launch_config(128).block_m == 64


def test_rejects_non_uint8b128_weights_before_launch() -> None:
    with pytest.raises(TypeError, match="qweight must be torch.uint8"):
        mla_kv_b_w8a16_bmm(
            torch.empty(1, 2, 192, dtype=torch.bfloat16),
            torch.empty(2, 192, 512, dtype=torch.int8),
            torch.empty(2, 1, 512, dtype=torch.bfloat16),
        )


def test_gptq_pack_rows_byte_order() -> None:
    raw = torch.tensor(
        [[[0, 1], [2, 3], [254, 255], [128, 129]]],
        dtype=torch.uint8,
    )
    packed = _pack_gptq_rows(raw)
    expected = (
        raw[:, 0, :].to(torch.int64)
        | (raw[:, 1, :].to(torch.int64) << 8)
        | (raw[:, 2, :].to(torch.int64) << 16)
        | (raw[:, 3, :].to(torch.int64) << 24)
    ).to(torch.int32)
    torch.testing.assert_close(packed[:, 0, :], expected)


def test_zero_tokens_returns_empty_output() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 tensor cores require Ampere or newer")

    heads = 2
    input_features = 192
    output_features = 512
    output = mla_kv_b_gptq_w8a16_bmm(
        torch.empty(
            0,
            heads,
            input_features,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        torch.empty(
            heads,
            input_features // 4,
            output_features,
            dtype=torch.int32,
            device="cuda",
        ),
        torch.empty(
            heads,
            1,
            output_features,
            dtype=torch.bfloat16,
            device="cuda",
        ),
    )

    assert output.shape == (heads, 0, output_features)
    assert output.dtype == torch.bfloat16
    assert output.numel() == 0


def test_cuda_graph_capture_and_replay() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 tensor cores require Ampere or newer")

    torch.manual_seed(23)
    heads = 2
    tokens = 3
    input_features = 512
    output_features = 256
    source_weight = torch.randn(
        heads,
        input_features,
        output_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_qweight, scales, reference_weight = _quantize_uint8b128(source_weight)
    qweight = _pack_gptq_rows(raw_qweight)
    static_x = torch.randn(
        tokens,
        heads,
        input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Triton must compile before stream capture.
    mla_kv_b_gptq_w8a16_bmm(static_x, qweight, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = mla_kv_b_gptq_w8a16_bmm(static_x, qweight, scales)

    replay_x = torch.randn_like(static_x)
    static_x.copy_(replay_x)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.bmm(replay_x.transpose(0, 1), reference_weight)
    torch.testing.assert_close(graph_output, expected, rtol=2e-2, atol=8e-2)


def test_two_streams_do_not_share_mutable_state() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 tensor cores require Ampere or newer")

    torch.manual_seed(31)
    heads = 2
    input_features = 192
    output_features = 512
    source_weight = torch.randn(
        heads,
        input_features,
        output_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_qweight, scales, reference_weight = _quantize_uint8b128(source_weight)
    qweight = _pack_gptq_rows(raw_qweight)
    x_a = torch.randn(
        1,
        heads,
        input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x_b = torch.randn(
        3,
        heads,
        input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Compile both launch shapes before dispatching on independent streams.
    mla_kv_b_gptq_w8a16_bmm(x_a, qweight, scales)
    mla_kv_b_gptq_w8a16_bmm(x_b, qweight, scales)
    current_stream = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(current_stream)
    stream_b.wait_stream(current_stream)
    with torch.cuda.stream(stream_a):
        output_a = mla_kv_b_gptq_w8a16_bmm(x_a, qweight, scales)
    with torch.cuda.stream(stream_b):
        output_b = mla_kv_b_gptq_w8a16_bmm(x_b, qweight, scales)
    current_stream.wait_stream(stream_a)
    current_stream.wait_stream(stream_b)

    expected_a = torch.bmm(x_a.transpose(0, 1), reference_weight)
    expected_b = torch.bmm(x_b.transpose(0, 1), reference_weight)
    torch.testing.assert_close(output_a, expected_a, rtol=2e-2, atol=8e-2)
    torch.testing.assert_close(output_b, expected_b, rtol=2e-2, atol=8e-2)


def test_custom_op_fake_dispatch_infers_output() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    heads = 2
    tokens = 3
    input_features = 192
    output_features = 512
    with FakeTensorMode():
        output = mla_kv_b_gptq_w8a16_bmm(
            torch.empty(
                tokens,
                heads,
                input_features,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty(
                heads,
                input_features // 4,
                output_features,
                dtype=torch.int32,
                device="cuda",
            ),
            torch.empty(
                heads,
                1,
                output_features,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        )

    assert output.shape == (heads, tokens, output_features)
    assert output.dtype == torch.bfloat16
    assert output.device.type == "cuda"
