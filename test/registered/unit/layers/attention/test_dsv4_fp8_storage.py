import math

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    decode_e4m3fn_byte,
    e4m3fn_decode_values,
    get_e4m3fn_decode_lut,
)
from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
    _prepare_packed_page_buffer,
    _resolve_packed_page_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_e4m3fn_decode_table_matches_pytorch_for_every_bit_pattern():
    expected = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    )
    actual = torch.tensor(e4m3fn_decode_values(), dtype=torch.float32)

    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    assert torch.equal(actual[finite], expected[finite])
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_e4m3fn_extended_exponent_and_nan_codes_are_exact():
    assert decode_e4m3fn_byte(0x7E) == 448.0
    assert decode_e4m3fn_byte(0xFE) == -448.0
    assert math.isnan(decode_e4m3fn_byte(0x7F))
    assert math.isnan(decode_e4m3fn_byte(0xFF))
    assert math.copysign(1.0, decode_e4m3fn_byte(0x80)) == -1.0


def test_decode_lut_has_a_stable_address_and_bf16_storage():
    first = get_e4m3fn_decode_lut("cpu")
    second = get_e4m3fn_decode_lut(torch.device("cpu"))

    assert first is second
    assert first.data_ptr() == second.data_ptr()
    assert first.dtype == torch.bfloat16
    assert first.shape == (256,)


def test_missing_cuda_lut_is_rejected_during_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="not primed before CUDA graph capture"):
        # The capture check precedes allocation, so this deliberately invalid
        # device index is safe in a CPU-only test process.
        get_e4m3fn_decode_lut("cuda:31415")


@pytest.mark.parametrize("page_size", [2, 64, 128])
def test_raw_page_preparation_retains_allocator_padding(page_size):
    logical_page_bytes = page_size * 584
    padded_page_bytes = math.ceil(logical_page_bytes / 576) * 576
    raw = torch.empty((2, padded_page_bytes), dtype=torch.uint8)

    raw_u8, raw_bf16, page_stride_bytes = _prepare_packed_page_buffer(
        raw,
        page_size,
        name="cache",
    )

    assert raw_u8.data_ptr() == raw.data_ptr()
    assert raw_bf16.data_ptr() == raw.data_ptr()
    assert page_stride_bytes == padded_page_bytes

    flashmla_view = raw[:, :logical_page_bytes].view(2, page_size, 1, 584)
    assert _resolve_packed_page_size(flashmla_view, None, name="cache") == page_size
    assert (
        _prepare_packed_page_buffer(
            flashmla_view,
            page_size,
            name="cache",
        )[2]
        == padded_page_bytes
    )

    native_fp8_view = flashmla_view.view(torch.float8_e4m3fn)
    prepared_u8, _, prepared_stride = _prepare_packed_page_buffer(
        native_fp8_view,
        page_size,
        name="cache",
    )
    assert prepared_u8.dtype == torch.uint8
    assert prepared_u8.data_ptr() == raw.data_ptr()
    assert prepared_stride == padded_page_bytes


def test_native_two_dimensional_page_buffer_requires_explicit_page_size():
    raw = torch.empty((2, 74880), dtype=torch.uint8)

    with pytest.raises(ValueError, match="required for a 2-D packed cache"):
        _resolve_packed_page_size(raw, None, name="cache")
