from unittest.mock import patch

import pytest
import torch
from sglang.srt.mem_cache.dsv4_kv_cache_dtype import (
    dsv4_kv_cache_dtype_name,
    dsv4_supports_fp8_kv_storage,
    dsv4_supports_int4_kv_storage,
    dsv4_supports_native_fp8_compute,
    dsv4_supports_oscar_int2_kv_storage,
    dsv4_supports_selective_c128_bf16_storage,
    dsv4_uses_ampere_fp8_kv_storage,
    format_dsv4_device_capability,
    get_dsv4_device_capability,
    normalize_dsv4_kv_cache_dtype_name,
    resolve_dsv4_kv_cache_dtype,
    resolve_dsv4_kv_cache_dtype_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("device_capability", [(8, 0), (8, 5), (8, 7), (8, 8)])
def test_non_sm86_pre_sm89_fp8_request_resolves_to_bfloat16(device_capability):
    assert (
        resolve_dsv4_kv_cache_dtype_name(
            "fp8_e4m3", device_capability=device_capability
        )
        == "bfloat16"
    )
    assert (
        resolve_dsv4_kv_cache_dtype(
            torch.float8_e4m3fn, device_capability=device_capability
        )
        == torch.bfloat16
    )


def test_sm86_keeps_fp8_as_byte_packed_storage():
    assert (
        resolve_dsv4_kv_cache_dtype_name("fp8_e4m3", device_capability=(8, 6))
        == "fp8_e4m3"
    )


@pytest.mark.parametrize(
    ("device_capability", "supported"),
    [
        ((8, 6), True),
        ((8, 0), False),
        ((8, 9), False),
        ((9, 0), False),
        ((12, 0), False),
        ((None, None), False),
    ],
)
def test_int4_storage_capability_is_exact_sm86_and_unknown_fails_closed(
    device_capability,
    supported,
):
    assert dsv4_supports_int4_kv_storage(device_capability) is supported
    assert (
        resolve_dsv4_kv_cache_dtype(torch.float8_e4m3fn, device_capability=(8, 6))
        == torch.float8_e4m3fn
    )


@pytest.mark.parametrize(
    ("device_capability", "supported"),
    [
        ((8, 6), True),
        ((8, 0), False),
        ((8, 9), False),
        ((9, 0), False),
        ((12, 0), False),
        ((None, None), False),
    ],
)
def test_oscar_int2_storage_capability_is_exact_sm86_and_fails_closed(
    device_capability,
    supported,
):
    assert dsv4_supports_oscar_int2_kv_storage(device_capability) is supported


@pytest.mark.parametrize(
    ("device_capability", "supported"),
    [
        ((8, 6), True),
        ((8, 0), False),
        ((8, 9), False),
        ((9, 0), False),
        ((12, 0), False),
        ((None, None), False),
    ],
)
def test_selective_c128_bf16_capability_is_exact_sm86_and_fails_closed(
    device_capability,
    supported,
):
    assert dsv4_supports_selective_c128_bf16_storage(device_capability) is supported


@pytest.mark.parametrize("device_capability", [(8, 9), (9, 0), (12, 0)])
def test_native_fp8_devices_keep_fp8_storage(device_capability):
    assert (
        resolve_dsv4_kv_cache_dtype_name(
            "fp8_e4m3", device_capability=device_capability
        )
        == "fp8_e4m3"
    )
    assert (
        resolve_dsv4_kv_cache_dtype(
            torch.float8_e4m3fn, device_capability=device_capability
        )
        == torch.float8_e4m3fn
    )


@pytest.mark.parametrize(
    ("device_capability", "supports_storage", "native_compute", "ampere_storage"),
    [
        ((8, 0), False, False, False),
        ((8, 6), True, False, True),
        ((8, 7), False, False, False),
        ((8, 9), True, True, False),
        ((9, 0), True, True, False),
        ((12, 0), True, True, False),
        ((None, None), True, False, False),
    ],
)
def test_fp8_storage_and_native_compute_capabilities_are_independent(
    device_capability,
    supports_storage,
    native_compute,
    ampere_storage,
):
    assert dsv4_supports_fp8_kv_storage(device_capability) is supports_storage
    assert dsv4_supports_native_fp8_compute(device_capability) is native_compute
    assert dsv4_uses_ampere_fp8_kv_storage(device_capability) is ampere_storage


@pytest.mark.parametrize("device_capability", [(8, 6), (12, 0)])
def test_explicit_bfloat16_is_respected_on_every_device(device_capability):
    assert (
        resolve_dsv4_kv_cache_dtype_name(
            "bfloat16", device_capability=device_capability
        )
        == "bfloat16"
    )
    assert (
        resolve_dsv4_kv_cache_dtype(torch.bfloat16, device_capability=device_capability)
        == torch.bfloat16
    )


@pytest.mark.parametrize("device_capability", [(None, None), (None, 0), (8, None)])
def test_unknown_capability_defers_fp8_fallback_to_cuda_worker(device_capability):
    assert (
        resolve_dsv4_kv_cache_dtype_name(
            "fp8_e4m3", device_capability=device_capability
        )
        == "fp8_e4m3"
    )


def test_non_cuda_device_is_not_probed_for_cuda_capability():
    with patch(
        "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability"
    ) as get_device_capability:
        assert get_dsv4_device_capability("npu") == (None, None)
        get_device_capability.assert_not_called()


def test_dtype_names_are_canonical_and_validated():
    assert normalize_dsv4_kv_cache_dtype_name("bf16") == "bfloat16"
    assert dsv4_kv_cache_dtype_name(torch.bfloat16) == "bfloat16"
    assert format_dsv4_device_capability((8, 6)) == "SM86"
    assert format_dsv4_device_capability((12, 0)) == "SM120"
    with pytest.raises(ValueError, match="only supports"):
        normalize_dsv4_kv_cache_dtype_name("fp8_e5m2")
    with pytest.raises(ValueError, match="only supports"):
        dsv4_kv_cache_dtype_name(torch.float16)
