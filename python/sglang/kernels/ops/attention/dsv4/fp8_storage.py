"""Architecture-neutral E4M3FN byte-storage helpers for DeepSeek V4.

Ampere cannot execute native FP8 MMA instructions, but it can store the exact
E4M3FN bit pattern in a ``torch.uint8`` tensor.  Consumers decode those bytes
through a tiny BF16 lookup table before using Ampere's BF16 tensor cores.  This
is the cache analogue of the FP8-as-storage technique described at
https://amohan.dev/blog/2026/fp8-as-storage-imma-ampere/.

The table is deliberately materialized before CUDA-graph capture.  A missing
table during capture is an error rather than an implicit allocation whose
address/lifetime would be unsafe to replay.
"""

from __future__ import annotations

import math
import threading

import torch

_E4M3FN_DECODE_LUTS: dict[torch.device, torch.Tensor] = {}
_E4M3FN_DECODE_LUT_LOCK = threading.Lock()


def decode_e4m3fn_byte(value: int) -> float:
    """Decode one raw PyTorch/NVIDIA E4M3FN byte into a Python float."""
    if not 0 <= value <= 0xFF:
        raise ValueError(f"E4M3FN byte must be in [0, 255], got {value}.")

    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 3) & 0x0F
    mantissa = value & 0x07

    # E4M3FN extends exponent 15 with six finite mantissas.  Only mantissa 7
    # is NaN, making +/-448 (0x7e/0xfe) the largest finite magnitude.
    if exponent == 0x0F and mantissa == 0x07:
        return math.copysign(math.nan, sign)
    if exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return math.copysign(magnitude, sign)


def e4m3fn_decode_values() -> tuple[float, ...]:
    """Return the complete 256-entry E4M3FN decode table."""
    return tuple(decode_e4m3fn_byte(value) for value in range(256))


def _normalize_device(device: torch.device | str) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


def get_e4m3fn_decode_lut(device: torch.device | str) -> torch.Tensor:
    """Return a stable per-device BF16 LUT, allocating only outside capture."""
    normalized = _normalize_device(device)
    cached = _E4M3FN_DECODE_LUTS.get(normalized)
    if cached is not None:
        return cached

    if normalized.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DeepSeek V4 Ampere FP8 decode LUT was not primed before CUDA graph "
            "capture. Construct the KV pool before capture begins."
        )

    with _E4M3FN_DECODE_LUT_LOCK:
        cached = _E4M3FN_DECODE_LUTS.get(normalized)
        if cached is None:
            # Every finite E4M3FN value is exactly representable in BF16.  Keep
            # the table in BF16 so the hot kernels can feed HMMA directly.
            cached = torch.tensor(
                e4m3fn_decode_values(),
                dtype=torch.bfloat16,
                device=normalized,
            )
            _E4M3FN_DECODE_LUTS[normalized] = cached
    return cached


def prime_e4m3fn_decode_lut(device: torch.device | str) -> None:
    """Materialize the stable LUT during pool initialization/warmup."""
    get_e4m3fn_decode_lut(device)


def clear_e4m3fn_decode_lut_cache() -> None:
    """Drop Python references to cached LUTs (tests only; never during serving)."""
    with _E4M3FN_DECODE_LUT_LOCK:
        _E4M3FN_DECODE_LUTS.clear()
