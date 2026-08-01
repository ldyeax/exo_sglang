# SPDX-License-Identifier: Apache-2.0
"""Portable native-MXFP4 MoE execution for DeepSeek V4.

This adapter keeps checkpoint expert weights in packed E2M1 with UE8M0
scales and delegates the actual gather/scatter GEMMs to the validated
``triton_kernels.matmul_ogs`` implementation carried by KTransformers.
It is the SM86 alternative to FlashInfer's SM100 kernel and Marlin's unsafe
Ampere MXFP4 path.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.utils import log_info_on_rank0, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

_KERNEL_RELATIVE_PATH = Path(
    "third_party/sglang/python/sglang/srt/layers/quantization/"
    "v4_triton_kernels_moe.py"
)


def _resolve_portable_kernel_path() -> Path:
    """Find the KTransformers kernel from the active Python source roots."""
    configured_path = os.environ.get("SGLANG_V4_TRITON_KERNEL_PATH")
    if configured_path:
        return Path(configured_path)

    searched_paths = []
    for source_root_text in sys.path:
        if not source_root_text:
            continue
        candidate_path = Path(source_root_text) / _KERNEL_RELATIVE_PATH
        searched_paths.append(candidate_path)
        if candidate_path.is_file():
            return candidate_path

    searched = ", ".join(str(path) for path in searched_paths)
    raise FileNotFoundError(
        "DeepSeek V4 portable MXFP4 kernel was not found below any active "
        f"Python source root; searched: {searched}"
    )


@lru_cache(maxsize=1)
def _portable_kernel_module() -> ModuleType:
    """Load the proven KTransformers portable kernel against this SGLang API."""
    kernel_path = _resolve_portable_kernel_path()
    if not kernel_path.is_file():
        raise FileNotFoundError(
            f"DeepSeek V4 portable MXFP4 kernel is missing: {kernel_path}"
        )
    module_name = "_sglang_dsv4_portable_mxfp4_kernel"
    cached_module = sys.modules.get(module_name)
    if cached_module is not None:
        return cached_module
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DeepSeek V4 MXFP4 kernel: {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Mxfp4TritonKernelsMoEMethod:
    """Native E2M1/UE8M0 MoE method using portable Triton gather/scatter GEMMs."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        # This method owns the complete MoE pipeline, but the layer still
        # publishes its routing/clamp contract through MoeRunnerConfig.
        self.moe_runner_config = moe_runner_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        fp4_block_k = 32
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            return

        kernel = _portable_kernel_module()
        w13_raw = layer.w13_weight.data
        w2_raw = layer.w2_weight.data
        w13_scale_raw = layer.w13_weight_scale_inv.data
        w2_scale_raw = layer.w2_weight_scale_inv.data
        if w13_scale_raw.dtype == torch.float32:
            w13_scale_raw = w13_scale_raw.to(torch.float8_e8m0fnu)
            w2_scale_raw = w2_scale_raw.to(torch.float8_e8m0fnu)

        hidden_size = w13_raw.shape[2] * 2
        intermediate_size = w2_raw.shape[2] * 2
        log_info_on_rank0(
            logger,
            "Preparing native MXFP4 experts for portable Triton backend "
            f"(layer: {self.prefix}, hidden={hidden_size}, "
            f"intermediate={intermediate_size})...",
        )
        w13, w13_precision, w2, w2_precision = (
            kernel.convert_v4_weights_to_triton_kernels(
                w13_raw,
                w13_scale_raw,
                w2_raw,
                w2_scale_raw,
            )
        )
        del layer.w13_weight
        del layer.w2_weight
        del layer.w13_weight_scale_inv
        del layer.w2_weight_scale_inv
        layer._dsv4_tk_w13 = w13
        layer._dsv4_tk_w13_precision = w13_precision
        layer._dsv4_tk_w2 = w2
        layer._dsv4_tk_w2_precision = w2_precision
        layer._dsv4_tk_intermediate_size = intermediate_size
        layer._dsv4_tk_num_experts = w13_raw.shape[0]
        layer._dsv4_mxfp4_backend = "triton_kernels"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")
        kernel = _portable_kernel_module()
        output = kernel.apply_v4_triton_kernels_moe(
            hidden_states=dispatch_output.hidden_states,
            w13_swiz=layer._dsv4_tk_w13,
            w13_pcg=layer._dsv4_tk_w13_precision,
            w2_swiz=layer._dsv4_tk_w2,
            w2_pcg=layer._dsv4_tk_w2_precision,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            intermediate_size=layer._dsv4_tk_intermediate_size,
            num_experts=layer._dsv4_tk_num_experts,
            routed_scaling_factor=1.0,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        # DeepseekV2MoE applies routed_scaling_factor after KTEPWrapperMethod,
        # so it must not be folded into this partial GPU contribution.
        return StandardCombineInput(hidden_states=output)
