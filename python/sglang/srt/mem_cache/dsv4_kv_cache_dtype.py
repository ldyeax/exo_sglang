"""Capability-aware DeepSeek V4 KV-cache storage dtype resolution.

DeepSeek V4 can use E4M3 as a byte-packed *storage* format on exact SM86 even
though Ampere cannot execute native FP8 tensor-core instructions.  SM86 cache
consumers decode those bytes and use BF16 tensor-core instructions instead.
SM89 and newer retain the native FP8 path, while other pre-SM89 devices use
the BF16 layout.  Keep these distinct capabilities in one place so argument
resolution, memory planning, and pool allocation cannot disagree about the
bytes stored per token.
"""

from __future__ import annotations

import torch

# Keep the implicit alias form so this module remains importable on SGLang's
# supported Python 3.10/3.11 runtimes; the ``type`` statement requires 3.12.
DeviceCapability = tuple[int | None, int | None]

DSV4_NATIVE_FP8_MIN_DEVICE_CAPABILITY = (8, 9)
DSV4_AMPERE_FP8_STORAGE_DEVICE_CAPABILITIES = frozenset({(8, 6)})
DSV4_INT4_STORAGE_DEVICE_CAPABILITIES = frozenset({(8, 6)})
DSV4_OSCAR_INT2_STORAGE_DEVICE_CAPABILITIES = frozenset({(8, 6)})
DSV4_SELECTIVE_C128_BF16_DEVICE_CAPABILITIES = frozenset({(8, 6)})


def get_dsv4_device_capability(
    device: object | None = None,
    *,
    device_index: int | None = None,
) -> DeviceCapability:
    """Return a CUDA capability without probing non-CUDA devices."""
    if isinstance(device, (str, torch.device)):
        device_text = str(device)
        if device_text.split(":", 1)[0] != "cuda":
            return (None, None)
    capability_device = device_index if device_index is not None else device
    return torch.cuda.get_device_capability(capability_device)


def dsv4_kv_cache_dtype_name(dtype: torch.dtype) -> str:
    """Return the canonical DeepSeek V4 cache dtype name for ``dtype``."""
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
        return "fp8_e4m3"
    raise ValueError(
        f"DeepSeek V4 only supports bfloat16 or fp8_e4m3 KV cache storage, got {dtype}."
    )


def normalize_dsv4_kv_cache_dtype_name(dtype_name: str) -> str:
    """Normalize and validate a requested DeepSeek V4 cache dtype name."""
    if dtype_name == "bf16":
        return "bfloat16"
    if dtype_name in {"bfloat16", "fp8_e4m3"}:
        return dtype_name
    raise ValueError(
        "DeepSeek V4 only supports bfloat16 or fp8_e4m3 KV cache storage, "
        f"got {dtype_name!r}."
    )


def dsv4_supports_native_fp8_compute(
    device_capability: DeviceCapability,
) -> bool:
    """Return whether DSV4 may execute its native FP8 CUDA/Triton kernels."""
    major, minor = device_capability
    if major is None or minor is None:
        return False
    return (major, minor) >= DSV4_NATIVE_FP8_MIN_DEVICE_CAPABILITY


def dsv4_uses_ampere_fp8_kv_storage(
    device_capability: DeviceCapability,
) -> bool:
    """Return whether DSV4 uses byte-packed FP8 with software decode."""
    major, minor = device_capability
    if major is None or minor is None:
        return False
    return (major, minor) in DSV4_AMPERE_FP8_STORAGE_DEVICE_CAPABILITIES


def dsv4_supports_int4_kv_storage(
    device_capability: DeviceCapability,
) -> bool:
    """Return whether the experimental signed-INT4 storage kernels are valid.

    Unlike the FP8 request resolver, this intentionally rejects an unknown
    capability.  INT4 storage is an opt-in physical layout and must never be
    admitted by a controller before the CUDA worker proves it is exact SM86.
    """
    major, minor = device_capability
    if major is None or minor is None:
        return False
    return (major, minor) in DSV4_INT4_STORAGE_DEVICE_CAPABILITIES


def dsv4_supports_oscar_int2_kv_storage(
    device_capability: DeviceCapability,
) -> bool:
    """Return whether the model-specific OSCAR-INT2 kernels are valid.

    OSCAR is not a spelling for the generic INT2/INT4 layouts.  The SM86
    implementation consumes a calibrated DeepSeek-V4 shared-latent rotation
    and has a distinct physical format, so an unknown or merely newer
    architecture must fail closed until that kernel is qualified there.
    """
    major, minor = device_capability
    if major is None or minor is None:
        return False
    return (major, minor) in DSV4_OSCAR_INT2_STORAGE_DEVICE_CAPABILITIES


def dsv4_supports_selective_c128_bf16_storage(
    device_capability: DeviceCapability,
) -> bool:
    """Return whether the mixed FP8-SWA/BF16-C128 kernels are valid.

    This is an optimization for Ampere's software FP8 decoder, not a generic
    cache dtype.  Unknown capabilities therefore fail closed just like the
    experimental INT4 storage path.
    """
    major, minor = device_capability
    if major is None or minor is None:
        return False
    return (major, minor) in DSV4_SELECTIVE_C128_BF16_DEVICE_CAPABILITIES


def dsv4_supports_fp8_kv_storage(
    device_capability: DeviceCapability,
) -> bool:
    """Return whether an FP8 cache request should retain packed storage.

    An unknown capability returns ``True`` so controller-only configuration
    preserves the request.  A CUDA worker resolves again with its concrete
    capability before memory planning and allocation.
    """
    major, minor = device_capability
    if major is None or minor is None:
        return True
    return dsv4_uses_ampere_fp8_kv_storage(
        device_capability
    ) or dsv4_supports_native_fp8_compute(device_capability)


def resolve_dsv4_kv_cache_dtype_name(
    requested_dtype_name: str,
    *,
    device_capability: DeviceCapability,
) -> str:
    """Resolve the physical DSV4 cache dtype for a CUDA capability.

    An unknown capability preserves the request.  This lets configuration be
    constructed in a CPU-only controller process; CUDA workers resolve again
    with their concrete device before planning and allocation.
    """
    requested_dtype_name = normalize_dsv4_kv_cache_dtype_name(requested_dtype_name)
    if requested_dtype_name == "bfloat16":
        return requested_dtype_name

    if not dsv4_supports_fp8_kv_storage(device_capability):
        return "bfloat16"
    return requested_dtype_name


def resolve_dsv4_kv_cache_dtype(
    requested_dtype: torch.dtype,
    *,
    device_capability: DeviceCapability,
) -> torch.dtype:
    """Torch-dtype form of :func:`resolve_dsv4_kv_cache_dtype_name`."""
    effective_name = resolve_dsv4_kv_cache_dtype_name(
        dsv4_kv_cache_dtype_name(requested_dtype),
        device_capability=device_capability,
    )
    return torch.bfloat16 if effective_name == "bfloat16" else requested_dtype


def format_dsv4_device_capability(
    device_capability: DeviceCapability,
) -> str:
    """Format a CUDA capability for startup diagnostics."""
    major, minor = device_capability
    if major is None or minor is None:
        return "unknown device capability"
    return f"SM{major}{minor}"
