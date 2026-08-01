from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    expected_shape = (x.size(0), y.size(0))
    if output is not None:
        if (
            output.shape != expected_shape
            or output.dtype != torch.float32
            or output.device != x.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "caller-owned BF16xBF16->FP32 output must be contiguous FP32 "
                f"on {x.device} with shape {expected_shape}, got "
                f"shape={tuple(output.shape)} dtype={output.dtype} "
                f"device={output.device} contiguous={output.is_contiguous()}"
            )
    if _use_aiter:
        result = tgemm.mm(x, y, otype=x.dtype).float()
        if output is not None:
            output.copy_(result)
            return output
        return result
    elif _linear_bf16_fp32_algo == "deep_gemm":
        z = (
            output
            if output is not None
            else torch.empty(
                x.size(0), y.size(0), dtype=torch.float32, device=x.device
            )
        )
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    else:
        if output is not None:
            return torch.mm(x, y.t(), out=output, out_dtype=torch.float32)
        return torch.mm(x, y.t(), out_dtype=torch.float32)
