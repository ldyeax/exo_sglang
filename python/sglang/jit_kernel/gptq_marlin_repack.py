from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

try:
    from sgl_kernel import gptq_marlin_repack as _aot_gptq_marlin_repack
except ImportError:
    _aot_gptq_marlin_repack = None

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_TILE_SIZE = 16


@cache_once
def _jit_gptq_marlin_repack_module() -> Module:
    return load_jit(
        "gptq_marlin_repack",
        cuda_files=["gemm/marlin/gptq_marlin_repack.cuh"],
        cuda_wrappers=[("gptq_marlin_repack", "gptq_marlin_repack")],
    )


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    # Prefer the prebuilt SGL-Kernel implementation when the installed wheel
    # exports it. Besides avoiding redundant startup compilation, this keeps
    # Ampere deployments independent of TileLang's CUDA-runtime stub loader,
    # which cannot discover libcudart when its own interposed symbols precede
    # PyTorch's locally loaded CUDA runtime in the dynamic-linker scope.
    if _aot_gptq_marlin_repack is not None:
        return _aot_gptq_marlin_repack(
            b_q_weight, perm, size_k, size_n, num_bits
        )

    pack_factor = 32 // num_bits

    # Allocate output tensor
    out = torch.empty(
        (size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )

    module = _jit_gptq_marlin_repack_module()
    module.gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)
    return out
