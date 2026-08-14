# SPDX-License-Identifier: Apache-2.0
"""
V4-Flash MXFP4 GPU MoE via OpenAI's `triton_kernels` package.

Default GPU MoE path for V4-Flash on every capability outside the trtllm
binary whitelist (`_TRTLLM_FP4_CAPS`, currently {(10,0)} = SM_100 datacenter
Blackwell only). Used on consumer Blackwell (SM_120, e.g. RTX 5090), Ada
(SM_89, e.g. L40S, RTX 4090), Hopper (SM_90), and Ampere — anywhere
flashinfer's `trtllm_fp4_block_scale_routed_moe` lacks a CUDA binary.

The OAI `triton_kernels` package's `matmul_ogs` (gather-or-scatter matmul)
provides a clean Triton MXFP4 path that:
- accepts FP4 packed weights + ue8m0 scales via either an upstream Hopper /
  Blackwell-DC swizzle (`_swizzle_mxfp4` from `mxfp4.py`) or a portable
  StridedLayout (`_swizzle_mxfp4_strided` here, used everywhere outside the
  trtllm whitelist),
- composes naturally with sglang's standard topk (we convert topk_ids /
  topk_weights → bitmatrix → routing_from_bitmatrix the same way the
  OAI vLLM port (PR #18595) does),
- runs the same Triton kernel that sglang already uses for unquantized
  bf16 MoE, so the kernel is on a tested code path; the only new
  ingredient is the FP4 weight + ue8m0 scale wiring.

Origin: sglang 本身.

Selection (default, capability-driven; see `mxfp4_deepseek.py`):
  cap == (10,0)             -> trtllm  (sm100f binary)
  cap not in whitelist      -> this module (StridedLayout + simulated MXFP)

Force-override env (diagnostic only):
  SGLANG_V4_USE_TRITON_KERNELS=1 -> force this module even on (10,0)
  SGLANG_V4_USE_TRITON_KERNELS=0 -> force trtllm even off-whitelist (fail loud)
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from functools import wraps
from inspect import Parameter, signature
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SMALL_ROW_ROUTING_ENV = "SGLANG_V4_MXFP4_SMALL_ROW_ROUTING"
# The variable-width residency campaign admits at most twenty-two local GPU
# experts (g14 plus eight selectively promoted experts).
# Keep the tiny router valid for every layer width under that ceiling; width is
# a compile-time specialization, while route values remain graph-replay dynamic.
_SMALL_ROW_ROUTING_MAX_EXPERTS = 22
_SMALL_ROW_ROUTING_TOP_K = 6
_SMALL_ROW_ROUTING_PADDED_TOP_K = 8
_SMALL_ROW_ROUTING_MAX_ROWS = 6
_SMALL_ROW_ROUTING_BLOCK_M_VALUES = (16, 32, 64, 128)
_SM86_SMALL_BATCH_GEMM_ENV = "SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM"
_SM86_FUSED_T5_MOE_ENV = "SGLANG_V4_MXFP4_FUSED_T5_MOE"
_SM86_SMALL_BATCH_GEMM_BLOCK_N = 128
_SM86_SMALL_BATCH_GEMM_SPLIT_K = 2
_SM86_SMALL_BATCH_GEMM_NUM_STAGES = 4
_SM86_SMALL_BATCH_GEMM_NUM_WARPS = 4
_SM86_SMALL_BATCH_GEMM_EXPECTED_PARAMETERS = (
    "out_dtype",
    "lhs_dtype",
    "rhs_dtype",
    "precision_config",
    "m",
    "n",
    "k",
    "routing_data",
    "can_use_persistent_tma",
    "can_use_fused_scatter",
    "enforce_bitwise_invariance",
    "epilogue_effective_itemsize",
    "constraints",
)

# These counters live in each scheduler process.  They record Python-side
# specialization selection, which happens while the CUDA-graph kernels are
# compiled/captured; graph replay intentionally does not re-enter Python.
_sm86_small_batch_gemm_patch_state = "not_attempted"
_sm86_small_batch_gemm_patch_error: Optional[str] = None
_sm86_small_batch_gemm_selection_counts: dict[tuple[int, int, int, int], int] = {}
_fused_t5_moe_conversion_count = 0
_fused_t5_moe_apply_count = 0
_fused_t5_kt_routing_apply_count = 0


def _sm86_small_batch_gemm_enabled() -> bool:
    return os.environ.get(_SM86_SMALL_BATCH_GEMM_ENV) == "1"


def fused_t5_moe_enabled() -> bool:
    """Return whether W13 uses the interleaved fused-SiLU layout."""

    return os.environ.get(_SM86_FUSED_T5_MOE_ENV) == "1"


def _interleave_gate_up_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``[gate..., up...]`` rows to ``[gate0, up0, ...]``."""

    if tensor.ndim != 3 or tensor.shape[1] % 2 != 0:
        raise ValueError("V4 W13 tensor must be rank 3 with an even row count")
    # CUDA has no cat/stack implementation for the native E8M0 scale dtype.
    # Interleaving is a pure byte permutation, so use its exact one-byte
    # carrier and restore the dtype without numerical conversion.
    original_dtype = tensor.dtype
    values = (
        tensor.view(torch.uint8) if original_dtype == torch.float8_e8m0fnu else tensor
    )
    intermediate_size = tensor.shape[1] // 2
    interleaved = (
        torch.stack(
            (
                values[:, :intermediate_size],
                values[:, intermediate_size:],
            ),
            dim=2,
        )
        .flatten(1, 2)
        .contiguous()
    )
    return (
        interleaved.view(original_dtype)
        if original_dtype == torch.float8_e8m0fnu
        else interleaved
    )


@triton.jit
def _dsv4_fused_silu_mul(input_values, limit):
    """DeepSeek SiLU(gate) * up epilogue for interleaved W13 columns."""

    gate, up = tl.split(
        tl.reshape(
            input_values,
            (input_values.shape[0], input_values.shape[1] // 2, 2),
        )
    )
    # The unfused path stores W13 to BF16 before sgl_kernel.silu_and_mul.
    # Preserve that rounding boundary so enabling fusion does not silently
    # change routing/acceptance through a higher-precision activation input.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    if limit is not None:
        gate = tl.minimum(gate, limit)
        up = tl.maximum(tl.minimum(up, limit), -limit)
    return (gate / (1.0 + tl.exp(-gate))) * up


def _make_dsv4_fused_activation(swiglu_limit: Optional[float]):
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation

    return FusedActivation(
        specs=FnSpecs(
            "dsv4_silu_mul",
            _dsv4_fused_silu_mul,
            ("limit",),
        ),
        fn_args=(swiglu_limit,),
        reduction_n=2,
    )


def _set_sm86_small_batch_gemm_patch_failure(
    state: str, error: BaseException | str
) -> None:
    global _sm86_small_batch_gemm_patch_error
    global _sm86_small_batch_gemm_patch_state

    _sm86_small_batch_gemm_patch_state = state
    if isinstance(error, BaseException):
        _sm86_small_batch_gemm_patch_error = f"{type(error).__name__}: {error}"
    else:
        _sm86_small_batch_gemm_patch_error = error


def get_sm86_small_batch_gemm_telemetry() -> dict[str, object]:
    """Return JSON/msgpack-safe proof of rank-local specialization selection."""

    observed_signatures = []
    for (m, n, k, local_experts), count in sorted(
        _sm86_small_batch_gemm_selection_counts.items()
    ):
        observed_signatures.append(
            {
                "m": m,
                "logical_rows": m // _SMALL_ROW_ROUTING_PADDED_TOP_K,
                "n": n,
                "k": k,
                "local_experts": local_experts,
                "selection_count": count,
            }
        )
    return {
        "configured": _sm86_small_batch_gemm_enabled(),
        "patch_state": _sm86_small_batch_gemm_patch_state,
        "patch_installed": _sm86_small_batch_gemm_patch_state == "installed",
        "patch_error": _sm86_small_batch_gemm_patch_error,
        "selection_count": sum(_sm86_small_batch_gemm_selection_counts.values()),
        "fused_t5_moe_configured": fused_t5_moe_enabled(),
        "fused_t5_moe_conversion_count": _fused_t5_moe_conversion_count,
        "fused_t5_moe_apply_count": _fused_t5_moe_apply_count,
        "fused_t5_kt_routing_apply_count": _fused_t5_kt_routing_apply_count,
        "observed_signatures": observed_signatures,
        "selected_config": {
            "block_n": _SM86_SMALL_BATCH_GEMM_BLOCK_N,
            "split_k": _SM86_SMALL_BATCH_GEMM_SPLIT_K,
            "num_stages": _SM86_SMALL_BATCH_GEMM_NUM_STAGES,
            "num_warps": _SM86_SMALL_BATCH_GEMM_NUM_WARPS,
        },
        "expected_call_parameters": list(_SM86_SMALL_BATCH_GEMM_EXPECTED_PARAMETERS),
    }


def _sm86_small_batch_gemm_matches_dispatch(
    *,
    rhs_dtype,
    precision_config,
    m: int,
    n: int,
    k: int,
    routing_data,
) -> bool:
    """Match the measured model/architecture dispatch, excluding opt-in state."""

    from triton_kernels.tensor import FP4

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 6)
        and rhs_dtype == FP4
        and getattr(precision_config, "weight_scale", None) is not None
        and n == 4096
        and k in (2048, 4096)
        and m in (8, 16, 24, 32, 40, 48)
        and routing_data is not None
        and 6 <= getattr(routing_data, "n_expts_tot", 0) <= 22
        and getattr(routing_data, "n_expts_act", None) == 8
        and getattr(routing_data, "expt_data", None) is not None
    )


def _sm86_small_batch_gemm_is_eligible(
    *,
    rhs_dtype,
    precision_config,
    m: int,
    n: int,
    k: int,
    routing_data,
    constraints: dict,
) -> bool:
    """Admit only the measured V4-Flash target/draft decode GEMMs.

    ``matmul_ogs`` sees the padded top-8 gather dimension as M.  Consequently
    M=8..48 corresponds exactly to one through six verification rows.  The
    N/K signatures below are the checkpoint's W13 and W2 matrices; constraining
    every dimension prevents this package-level flag patch from changing
    unrelated MoE, prefill, or dense matmuls.
    """
    return (
        _sm86_small_batch_gemm_enabled()
        and _sm86_small_batch_gemm_matches_dispatch(
            rhs_dtype=rhs_dtype,
            precision_config=precision_config,
            m=m,
            n=n,
            k=k,
            routing_data=routing_data,
        )
        # Respect diagnostic package constraints instead of silently
        # overriding them.  The V4 strided-layout patch contributes only the
        # required non-persistent constraint in production.
        and constraints == {"is_persistent": False}
    )


def _install_sm86_small_batch_gemm_patch(opt_flags_module) -> bool:
    """Install the exact pinned opt-flags wrapper, or fail closed when enabled.

    This is deliberately a separately testable boundary.  The upstream helper
    is package-private and has changed signatures between triton_kernels
    releases; silently retaining its heuristic after an incompatible upgrade
    makes the launcher's opt-in flag and benchmark provenance false.
    """

    global _sm86_small_batch_gemm_patch_error
    global _sm86_small_batch_gemm_patch_state

    if getattr(opt_flags_module, "_v4_sm86_small_batch_gemm_patched", False):
        _sm86_small_batch_gemm_patch_state = "installed"
        _sm86_small_batch_gemm_patch_error = None
        return True

    original = getattr(opt_flags_module, "make_default_opt_flags_nvidia", None)
    if original is None:
        message = "triton_kernels is missing make_default_opt_flags_nvidia"
        state = (
            "incompatible" if _sm86_small_batch_gemm_enabled() else "disabled_fallback"
        )
        _set_sm86_small_batch_gemm_patch_failure(state, message)
        if _sm86_small_batch_gemm_enabled():
            raise RuntimeError(message)
        return False

    parameters = tuple(signature(original).parameters.values())
    parameter_names = tuple(parameter.name for parameter in parameters)
    positional_only_or_keyword = all(
        parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    if (
        parameter_names != _SM86_SMALL_BATCH_GEMM_EXPECTED_PARAMETERS
        or not positional_only_or_keyword
    ):
        message = (
            "incompatible triton_kernels make_default_opt_flags_nvidia "
            f"signature: expected {_SM86_SMALL_BATCH_GEMM_EXPECTED_PARAMETERS}, "
            f"got {parameter_names}"
        )
        state = (
            "incompatible" if _sm86_small_batch_gemm_enabled() else "disabled_fallback"
        )
        _set_sm86_small_batch_gemm_patch_failure(state, message)
        if _sm86_small_batch_gemm_enabled():
            raise RuntimeError(message)
        return False

    @wraps(original)
    def make_default_opt_flags_nvidia_v4(*args, **kwargs):
        if len(args) != len(_SM86_SMALL_BATCH_GEMM_EXPECTED_PARAMETERS) or kwargs:
            if _sm86_small_batch_gemm_enabled():
                message = (
                    "SM86 V4 MXFP4 small-batch specialization received an "
                    "incompatible opt-flags call; expected thirteen positional "
                    f"arguments, got {len(args)} positional and "
                    f"{tuple(kwargs)} keyword arguments"
                )
                _set_sm86_small_batch_gemm_patch_failure(
                    "dispatch_incompatible", message
                )
                raise RuntimeError(message)
            return original(*args, **kwargs)

        dispatch_matches = _sm86_small_batch_gemm_matches_dispatch(
            rhs_dtype=args[2],
            precision_config=args[3],
            m=args[4],
            n=args[5],
            k=args[6],
            routing_data=args[7],
        )
        specialization_enabled = _sm86_small_batch_gemm_enabled()
        if (
            dispatch_matches
            and specialization_enabled
            and args[12] != {"is_persistent": False}
        ):
            message = (
                "SM86 V4 MXFP4 small-batch dispatch constraints drifted: "
                f"expected {{'is_persistent': False}}, got {args[12]!r}"
            )
            _set_sm86_small_batch_gemm_patch_failure("dispatch_incompatible", message)
            raise RuntimeError(message)

        selected = original(*args)
        if not dispatch_matches or not specialization_enabled:
            return selected

        specialized = replace(
            selected,
            block_n=_SM86_SMALL_BATCH_GEMM_BLOCK_N,
            split_k=_SM86_SMALL_BATCH_GEMM_SPLIT_K,
            num_stages=_SM86_SMALL_BATCH_GEMM_NUM_STAGES,
            num_warps=_SM86_SMALL_BATCH_GEMM_NUM_WARPS,
        )
        local_experts = int(args[7].n_expts_tot)
        dispatch_signature = (int(args[4]), int(args[5]), int(args[6]), local_experts)
        _sm86_small_batch_gemm_selection_counts[dispatch_signature] = (
            _sm86_small_batch_gemm_selection_counts.get(dispatch_signature, 0) + 1
        )
        if not getattr(opt_flags_module, "_v4_sm86_small_batch_gemm_logged", False):
            logger.info(
                "V4 MXFP4 SM86 small-batch GEMM enabled: "
                "block_n=%d split_k=%d stages=%d",
                _SM86_SMALL_BATCH_GEMM_BLOCK_N,
                _SM86_SMALL_BATCH_GEMM_SPLIT_K,
                _SM86_SMALL_BATCH_GEMM_NUM_STAGES,
            )
            opt_flags_module._v4_sm86_small_batch_gemm_logged = True
        return specialized

    opt_flags_module.make_default_opt_flags_nvidia = make_default_opt_flags_nvidia_v4
    opt_flags_module._v4_sm86_small_batch_gemm_patched = True
    _sm86_small_batch_gemm_patch_state = "installed"
    _sm86_small_batch_gemm_patch_error = None
    return True


def use_v4_triton_kernels() -> bool:
    """Force-override gate for the V4 triton_kernels path.

    Default behavior (env unset): the dispatcher in
    `mxfp4_deepseek.process_weights_after_loading` chooses this path for any
    capability outside `_TRTLLM_FP4_CAPS`, so this returning False does not
    mean the path is disabled — only that no override was requested.

    SGLANG_V4_USE_TRITON_KERNELS=1: force this path even on whitelisted
    capabilities (numerical comparison / debugging).

    SGLANG_V4_USE_TRITON_KERNELS=0: force trtllm even off-whitelist (will
    fail loud with "Unsupported architecture" — kept as a diagnostic exit).

    Origin: sglang 本身."""
    return os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") == "1"


def force_disable_v4_triton_kernels() -> bool:
    """True when the user explicitly set SGLANG_V4_USE_TRITON_KERNELS=0 to
    force trtllm even off-whitelist (diagnostic only). Origin: sglang 本身."""
    return os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") == "0"


# -----------------------------------------------------------------------------
# Bitmatrix construction (port of vLLM gpt_oss_triton_kernels_moe.pack_bitmatrix)
# -----------------------------------------------------------------------------


@triton.jit
def _pack_bitmatrix_v4(
    bitmatrix_ptr,  # uint32 [n_rows, bm_cols]
    topk_ids_ptr,  # int16 [n_rows, n_expts_act]
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # 32 (bits per uint32)
):
    """For each token row, set the bits of expert ids selected in topk_ids.

    bitmatrix[r, e // 32] |= (1 << (e % 32)) for each e in topk_ids[r, :].
    Uses tl.atomic_or to handle multiple bits in the same uint32 word.
    """
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n_rows

    # Iterate through topk slots (small, ≤ 16 typically)
    for k in range(n_expts_act):
        ids = tl.load(
            topk_ids_ptr + rows * n_expts_act + k,
            mask=row_mask,
            other=-1,
        ).to(tl.int32)
        valid = (ids >= 0) & row_mask
        col = ids // BLOCK_SIZE_K
        bit = ids - col * BLOCK_SIZE_K  # ids % 32
        ptrs = bitmatrix_ptr + rows * bm_cols + col
        tl.atomic_or(ptrs, (1 << bit).to(tl.uint32), mask=valid)


@triton.jit
def _pack_small_row_routing_v4(
    topk_ids_ptr,
    topk_weights_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    gate_scal_ptr,
    expert_hist_ptr,
    token_offsets_raw_ptr,
    token_offsets_pad_ptr,
    block_pid_map_ptr,
    gpu_experts_mask_ptr,
    logical_to_gpu_index_ptr,
    stride_ids_m,
    stride_ids_k,
    stride_weights_m,
    stride_weights_k,
    NUM_EXPERTS: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    INPUT_TOP_K: tl.constexpr,
    ROUTING_TOP_K: tl.constexpr,
    NUM_GATES: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_KT_REMAP: tl.constexpr,
):
    """Build every tiny grouped-MoE routing tensor in one CUDA program."""
    gate_positions = tl.arange(0, BLOCK_G)
    gate_mask = gate_positions < NUM_GATES
    token_rows = gate_positions // ROUTING_TOP_K
    topk_columns = gate_positions - token_rows * ROUTING_TOP_K
    input_mask = gate_mask & (topk_columns < INPUT_TOP_K)
    route_ids = tl.load(
        topk_ids_ptr + token_rows * stride_ids_m + topk_columns * stride_ids_k,
        mask=input_mask,
        other=-1,
    ).to(tl.int32)
    if HAS_KT_REMAP:
        global_valid = input_mask & (route_ids >= 0) & (route_ids < NUM_GLOBAL_EXPERTS)
        safe_global_ids = tl.where(global_valid, route_ids, 0)
        is_gpu_expert = tl.load(
            gpu_experts_mask_ptr + safe_global_ids,
            mask=global_valid,
            other=0,
        ).to(tl.int1)
        route_ids = tl.load(
            logical_to_gpu_index_ptr + safe_global_ids,
            mask=global_valid & is_gpu_expert,
            other=-1,
        ).to(tl.int32)
    valid_routes = gate_mask & (route_ids >= 0) & (route_ids < NUM_EXPERTS)

    # Sorting the combined (expert, original-position) key is stable by
    # construction.  It exactly matches routing_from_bitmatrix's expert-major
    # order while writing its logical top-8 padded index geometry directly.
    sentinel_expert = 0x7FFF
    sortable_experts = tl.where(valid_routes, route_ids, sentinel_expert)
    sort_keys = (sortable_experts.to(tl.uint32) << 16) | gate_positions.to(tl.uint32)
    sorted_keys = tl.sort(sort_keys)
    sorted_experts = (sorted_keys >> 16).to(tl.int32)
    sorted_positions = (sorted_keys & 0xFFFF).to(tl.int32)
    sorted_valid = (gate_positions < NUM_GATES) & (sorted_experts < NUM_EXPERTS)

    tl.store(
        gather_indices_ptr + gate_positions,
        tl.where(sorted_valid, sorted_positions, -1),
        mask=gate_mask,
    )
    tl.store(scatter_indices_ptr + gate_positions, -1, mask=gate_mask)
    tl.store(
        scatter_indices_ptr + sorted_positions,
        gate_positions,
        mask=sorted_valid,
    )
    sorted_rows = sorted_positions // ROUTING_TOP_K
    sorted_columns = sorted_positions - sorted_rows * ROUTING_TOP_K
    sorted_weights = tl.load(
        topk_weights_ptr
        + sorted_rows * stride_weights_m
        + sorted_columns * stride_weights_k,
        mask=sorted_valid,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        gate_scal_ptr + gate_positions,
        tl.where(sorted_valid, sorted_weights, 0.0),
        mask=gate_mask,
    )

    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    route_matches = (sorted_experts[None, :] == experts[:, None]) & sorted_valid[
        None, :
    ]
    expert_hist = tl.sum(route_matches.to(tl.int32), axis=1)
    expert_offsets = tl.cumsum(expert_hist, axis=0) - expert_hist
    valid_gate_count = tl.sum(expert_hist, axis=0)
    tl.store(expert_hist_ptr + experts, expert_hist, mask=expert_mask)
    tl.store(
        token_offsets_raw_ptr + experts,
        expert_offsets,
        mask=expert_mask,
    )
    tl.store(token_offsets_raw_ptr + NUM_EXPERTS, valid_gate_count)

    tile_positions = tl.arange(0, BLOCK_T)
    for block_index in tl.static_range(0, 4):
        expert_tiles = (expert_hist + (16 << block_index) - 1) // (16 << block_index)
        expert_tile_offsets = tl.cumsum(expert_tiles, axis=0) - expert_tiles
        total_tiles = tl.sum(expert_tiles, axis=0)
        token_offsets_row = token_offsets_pad_ptr + block_index * (NUM_EXPERTS + 1)
        tl.store(
            token_offsets_row + experts,
            expert_tile_offsets,
            mask=expert_mask,
        )
        tl.store(token_offsets_row + NUM_EXPERTS, total_tiles)

        owns_tile = (
            (tile_positions[:, None] >= expert_tile_offsets[None, :])
            & (tile_positions[:, None] < (expert_tile_offsets + expert_tiles)[None, :])
            & expert_mask[None, :]
        )
        tile_expert = tl.sum(
            tl.where(owns_tile, experts[None, :], 0),
            axis=1,
        )
        tile_block = tile_positions - tl.sum(
            tl.where(owns_tile, expert_tile_offsets[None, :], 0),
            axis=1,
        )
        encoded_tile = (tile_block << 16) | tile_expert
        map_mask = tile_positions < MAX_TILES
        block_map_row = block_pid_map_ptr + block_index * MAX_TILES
        tl.store(
            block_map_row + tile_positions,
            tl.where(tile_positions < total_tiles, encoded_tile, -1),
            mask=map_mask,
        )


def _small_row_routing_is_eligible(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
    gpu_experts_mask: torch.Tensor | None = None,
    logical_to_gpu_index: torch.Tensor | None = None,
) -> bool:
    """Return whether the opt-in fixed SM86 target-verify router applies."""
    base_eligible = (
        os.environ.get(_SMALL_ROW_ROUTING_ENV) == "1"
        and 1 <= num_local_experts <= _SMALL_ROW_ROUTING_MAX_EXPERTS
        and topk_ids.ndim == 2
        and topk_weights.ndim == 2
        and topk_ids.shape == topk_weights.shape
        and 1 <= topk_ids.shape[0] <= _SMALL_ROW_ROUTING_MAX_ROWS
        and topk_ids.shape[1] == _SMALL_ROW_ROUTING_TOP_K
        and topk_ids.device.type == "cuda"
        and topk_weights.device == topk_ids.device
        and torch.cuda.get_device_capability(topk_ids.device) == (8, 6)
        # StandardTopKOutput is int32/FP32.  Keep the production specialization
        # to that single compiled signature; other dtypes retain the baseline.
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
    )
    if not base_eligible:
        return False
    if (gpu_experts_mask is None) != (logical_to_gpu_index is None):
        return False
    if gpu_experts_mask is None:
        return True
    return (
        gpu_experts_mask.device == topk_ids.device
        and logical_to_gpu_index.device == topk_ids.device
        and gpu_experts_mask.dtype == torch.bool
        and logical_to_gpu_index.dtype == torch.int32
        and gpu_experts_mask.ndim == 1
        and logical_to_gpu_index.shape == gpu_experts_mask.shape
        and gpu_experts_mask.is_contiguous()
        and logical_to_gpu_index.is_contiguous()
    )


def _make_small_row_routing_data_v4(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
    *,
    gpu_experts_mask: torch.Tensor | None = None,
    logical_to_gpu_index: torch.Tensor | None = None,
):
    """Build exact top-6 routing in the baseline's padded top-8 geometry."""
    from triton_kernels.routing import (
        ExptData,
        GatherIndx,
        RoutingData,
        ScatterIndx,
    )

    num_rows, input_top_k = topk_ids.shape
    routing_top_k = _SMALL_ROW_ROUTING_PADDED_TOP_K
    # Preserve the baseline router's logical top-8 geometry.  In particular,
    # ScatterIndx uses row * 8 + column and GEMM2 reduces eight slots per row.
    # Compacting this to six changes BF16 reduction association at M >= 5.
    num_gates = num_rows * routing_top_k
    if num_gates <= num_local_experts:
        max_tiles = num_gates
    else:
        max_tiles = num_local_experts - 1 - ((num_local_experts - num_gates - 1) // 16)
    device = topk_ids.device
    gather_indices = torch.empty(num_gates, dtype=torch.int32, device=device)
    scatter_indices = torch.empty_like(gather_indices)
    gate_scal = torch.empty(num_gates, dtype=torch.bfloat16, device=device)
    expert_hist = torch.empty(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    token_offsets_raw = torch.empty(
        num_local_experts + 1,
        dtype=torch.int32,
        device=device,
    )
    token_offsets_pad_storage = torch.empty(
        (len(_SMALL_ROW_ROUTING_BLOCK_M_VALUES), num_local_experts + 1),
        dtype=torch.int32,
        device=device,
    )
    block_pid_map_storage = torch.empty(
        (len(_SMALL_ROW_ROUTING_BLOCK_M_VALUES), max_tiles),
        dtype=torch.int32,
        device=device,
    )
    _pack_small_row_routing_v4[(1,)](
        topk_ids,
        topk_weights,
        gather_indices,
        scatter_indices,
        gate_scal,
        expert_hist,
        token_offsets_raw,
        token_offsets_pad_storage,
        block_pid_map_storage,
        topk_ids if gpu_experts_mask is None else gpu_experts_mask,
        topk_ids if logical_to_gpu_index is None else logical_to_gpu_index,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        NUM_EXPERTS=num_local_experts,
        NUM_GLOBAL_EXPERTS=(
            0 if gpu_experts_mask is None else gpu_experts_mask.shape[0]
        ),
        INPUT_TOP_K=input_top_k,
        ROUTING_TOP_K=routing_top_k,
        NUM_GATES=num_gates,
        MAX_TILES=max_tiles,
        BLOCK_G=triton.next_power_of_2(num_gates),
        BLOCK_E=triton.next_power_of_2(num_local_experts),
        BLOCK_T=triton.next_power_of_2(max_tiles),
        HAS_KT_REMAP=gpu_experts_mask is not None,
        num_warps=1,
    )
    token_offsets_pad = {
        block_m: token_offsets_pad_storage[index]
        for index, block_m in enumerate(_SMALL_ROW_ROUTING_BLOCK_M_VALUES)
    }
    block_pid_map = {
        block_m: block_pid_map_storage[index]
        for index, block_m in enumerate(_SMALL_ROW_ROUTING_BLOCK_M_VALUES)
    }
    expert_data = ExptData(
        expert_hist,
        token_offsets_raw,
        token_offsets_pad,
        block_pid_map,
    )
    return (
        RoutingData(
            gate_scal,
            expert_hist,
            num_local_experts,
            routing_top_k,
            expert_data,
        ),
        GatherIndx(gather_indices, scatter_indices),
        ScatterIndx(scatter_indices, gather_indices),
    )


def _make_routing_data_v4(
    topk_ids: torch.Tensor,  # [M, n_topk] int (any int dtype)
    topk_weights: torch.Tensor,  # [M, n_topk] float
    num_local_experts: int,
    *,
    gpu_experts_mask: torch.Tensor | None = None,
    logical_to_gpu_index: torch.Tensor | None = None,
):
    """Convert sglang's standard (topk_ids, topk_weights) to triton_kernels'
    (RoutingData, GatherIndx, ScatterIndx) via the bitmatrix path.

    Mirrors vLLM `make_routing_data` (gpt_oss_triton_kernels_moe.py).

    Note: triton_kernels' internal routing kernel
    (routing_details/_routing_compute._routing_compute_indx) does
    `tl.arange(0, N_EXPTS_ACT * BLOCK_M)` which requires the product to be
    a power of 2; equivalently `n_topk` must itself be a power of 2.
    V4-Flash uses top-6, which is not — so we pad to the next power of 2
    by appending invalid (-1) slots, which the kernel masks out via the
    `expt_indx == -1` check (also matches matmul_ogs's gammas == -1
    convention). Pad cost is ~33% extra slot bookkeeping; gemm work is
    unchanged because invalid slots are not routed.
    """
    if _small_row_routing_is_eligible(
        topk_ids,
        topk_weights,
        num_local_experts,
        gpu_experts_mask,
        logical_to_gpu_index,
    ):
        return _make_small_row_routing_data_v4(
            topk_ids,
            topk_weights,
            num_local_experts,
            gpu_experts_mask=gpu_experts_mask,
            logical_to_gpu_index=logical_to_gpu_index,
        )

    if gpu_experts_mask is not None or logical_to_gpu_index is not None:
        raise RuntimeError(
            "KT fused routing was requested outside the exact small-row specialization"
        )

    try:
        from triton_kernels.routing import routing_from_bitmatrix
    except ModuleNotFoundError:
        routing_from_bitmatrix = None
    from triton_kernels.tensor import BIT, Bitmatrix

    if routing_from_bitmatrix is None:
        from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
        from triton_kernels.tensor import SparseMatrix, make_ragged_tensor_metadata

        topk_ids_i16 = topk_ids.to(torch.int16).contiguous()
        topk_weights_bf = topk_weights.to(torch.bfloat16).contiguous()
        n_rows, n_topk_raw = topk_ids_i16.shape
        n_topk = triton.next_power_of_2(n_topk_raw)
        if n_topk != n_topk_raw:
            pad_len = n_topk - n_topk_raw
            topk_ids_i16 = torch.cat(
                [
                    topk_ids_i16,
                    torch.full(
                        (n_rows, pad_len),
                        -1,
                        dtype=torch.int16,
                        device=topk_ids_i16.device,
                    ),
                ],
                dim=1,
            ).contiguous()
            topk_weights_bf = torch.cat(
                [
                    topk_weights_bf,
                    torch.full(
                        (n_rows, pad_len),
                        -1.0,
                        dtype=torch.bfloat16,
                        device=topk_weights_bf.device,
                    ),
                ],
                dim=1,
            ).contiguous()
        block_size_m = 512
        block_size_k = 32
        bm_cols = triton.cdiv(num_local_experts, block_size_k)
        bitmatrix_data = torch.zeros(
            (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids_i16.device
        )
        _pack_bitmatrix_v4[(triton.cdiv(n_rows, block_size_m),)](
            bitmatrix_data,
            topk_ids_i16,
            n_rows,
            bm_cols,
            n_topk,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_K=block_size_k,
        )
        bitmatrix = Bitmatrix(
            bitmatrix_data,
            dtype=BIT,
            shape=[n_rows, num_local_experts],
            shape_max=[n_rows, None],
        )
        sparse_topk = SparseMatrix(
            vals=topk_weights_bf, indx=topk_ids_i16, mask=bitmatrix
        )
        dispatch_indx = sparse_topk.mask_metadata.row_sorted_indx
        combine_indx = sparse_topk.mask_metadata.col_sorted_indx
        ragged_metadata = make_ragged_tensor_metadata(
            sparse_topk.mask_metadata.col_sum, dispatch_indx.shape[0]
        )
        gate_scal = sparse_topk.vals.flatten()[combine_indx]
        routing_data = RoutingData(
            gate_scal,
            ragged_metadata.slice_sizes,
            num_local_experts,
            n_topk,
            ragged_metadata,
        )
        return (
            routing_data,
            GatherIndx(combine_indx, dispatch_indx),
            ScatterIndx(dispatch_indx, combine_indx),
        )

    topk_ids_i16 = topk_ids.to(torch.int16).contiguous()
    topk_weights_bf = topk_weights.to(torch.bfloat16).contiguous()

    n_rows, n_topk_raw = topk_ids_i16.shape

    # Pad n_topk to next power of 2 (V4: 6 -> 8) for triton_kernels routing.
    n_topk = 1
    while n_topk < n_topk_raw:
        n_topk *= 2
    if n_topk != n_topk_raw:
        pad_len = n_topk - n_topk_raw
        pad_ids = torch.full(
            (n_rows, pad_len), -1, dtype=torch.int16, device=topk_ids_i16.device
        )
        pad_w = torch.full(
            (n_rows, pad_len),
            -1.0,
            dtype=torch.bfloat16,
            device=topk_ids_i16.device,
        )
        topk_ids_i16 = torch.cat([topk_ids_i16, pad_ids], dim=1).contiguous()
        topk_weights_bf = torch.cat([topk_weights_bf, pad_w], dim=1).contiguous()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    bitmatrix_data = torch.zeros(
        (n_rows, bm_cols),
        dtype=torch.uint32,
        device=topk_ids_i16.device,
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    _pack_bitmatrix_v4[grid](
        bitmatrix_data,
        topk_ids_i16,
        n_rows,
        bm_cols,
        n_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix = Bitmatrix(
        bitmatrix_data,
        shape=[n_rows, bm_cols * 32],
        shape_max=[n_rows, None],
        scratchpad=None,
    )

    # matmul_ogs convention: invalid topk weights are -1.0 (not 0).
    # Use masked_fill (Python-scalar) instead of torch.where(...,
    # torch.tensor(-1.0, ...), ...) so this is CUDA-graph-capture safe
    # (no H2D copy during capture).
    topk_weights_bf = topk_weights_bf.masked_fill(topk_ids_i16 == -1, -1.0)

    return routing_from_bitmatrix(
        bitmatrix, topk_weights_bf, topk_ids_i16, num_local_experts, n_topk
    )


# -----------------------------------------------------------------------------
# Weight conversion: raw V4 MXFP4 → triton_kernels-format
# -----------------------------------------------------------------------------


# Capabilities for which flashinfer's `trtllm_fp4_block_scale_routed_moe`
# ships a working binary. Verified on flashinfer 0.6.8: only sm100f.
# Outside this set, we use the StridedLayout + simulated-MXFP non-persistent
# matmul_ogs path here. Keep this in sync with the same constant in
# `mxfp4_deepseek.py`.
_TRTLLM_FP4_CAPS = {(10, 0)}


def _use_strided_layout() -> bool:
    """True when matmul_ogs must use StridedLayout + simulated MXFP rather
    than the upstream Hopper-TMA / Blackwell-DC swizzle. Origin: sglang 本身.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() not in _TRTLLM_FP4_CAPS


def _patch_strided_mxfp():
    """Strided-layout enablement for triton_kernels MXFP4 path. Three patches:

    (1) `target_info.has_native_mxfp()` must return False on non-trtllm-
        whitelist capabilities so matmul_ogs takes the simulated MXFP non-
        persistent code path instead of the native (TMA + cluster shared
        mem + tile::gather4) path that needs Hopper / DC-Blackwell features
        consumer / Ada / Ampere don't have. (On SM_89/SM_80 the upstream
        `has_native_mxfp` already returns False; force-False is a no-op
        but cheap. On SM_120 the upstream returns True since cap[0]==12
        is treated as Blackwell, hence the override.)

    (2) opt_flags must force `is_persistent=False`. The auto-selection in
        make_opt_flags chooses persistent for larger M (= prefill batches),
        which then collides with simulated-MXFP and raises 'Must use non-
        persistent kernel for simulated MXFP' at runtime. CG decode (small
        M) auto-selects non-persistent so it works without this; prefill
        needs the explicit constraint.

    (3) An opt-in exact-SM86 decode specialization replaces the package's
        dense-grid split-K estimate with the real-weight-qualified block-N
        128, split-K 2, four-stage point.  Its guards exclude prefill and all
        non-V4 matrix shapes.

    Origin: sglang 本身 (triton_kernels package's Blackwell layout assumes
    SM_100 features; everything outside the trtllm whitelist falls in a
    similar gap)."""
    if not _use_strided_layout():
        if _sm86_small_batch_gemm_enabled():
            message = (
                "SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM=1 requires the "
                "SM86 strided-MXFP path"
            )
            _set_sm86_small_batch_gemm_patch_failure("incompatible", message)
            raise RuntimeError(message)
        return
    import triton_kernels.target_info as target_info

    if not getattr(target_info, "_v4_strided_patched", False):
        original = target_info.has_native_mxfp

        def has_native_mxfp_strided():
            if _use_strided_layout():
                return False
            return original()

        target_info.has_native_mxfp = has_native_mxfp_strided
        target_info._v4_strided_patched = True
    # opt_flags imports has_native_mxfp into its namespace at import time;
    # refresh the binding there too. Also force is_persistent=False so the
    # auto-selector in make_opt_flags can't pick the native path.
    #
    # Source-level patch make_default_opt_flags_nvidia: replace the bare
    # `assert num_stages >= 1` with `num_stages = max(num_stages, 1)`. The
    # assertion fires for capabilities outside the tested matrix (observed
    # on certain Hopper / SM_120 configs under MXFP4 strided layout). The
    # update_opt_flags_constraints API does NOT override num_stages before
    # the heuristic's local computation, so a constraint-only fix would be
    # honored too late. Patching the function source at import time is the
    # only place this can be neutralized without forking triton_kernels.
    try:
        import triton_kernels.matmul_ogs_details.opt_flags as _of

        if hasattr(_of, "has_native_mxfp"):
            _of.has_native_mxfp = target_info.has_native_mxfp
        _of.update_opt_flags_constraints({"is_persistent": False})

        if hasattr(_of, "make_default_opt_flags_nvidia") and not getattr(
            _of, "_v4_assert_patched", False
        ):
            import inspect as _inspect
            import textwrap as _textwrap

            _src = _textwrap.dedent(
                _inspect.getsource(_of.make_default_opt_flags_nvidia)
            )
            _patched_src = _src.replace(
                "assert num_stages >= 1",
                "num_stages = max(num_stages, 1)  # v4-flash patch",
            )
            if _patched_src != _src:
                # rename to avoid recursive shadow when exec'd
                _patched_src = _patched_src.replace(
                    "def make_default_opt_flags_nvidia",
                    "def _v4_make_default_opt_flags_nvidia",
                    1,
                )
                exec(_patched_src, _of.__dict__)
                _of.make_default_opt_flags_nvidia = (
                    _of._v4_make_default_opt_flags_nvidia
                )
                _of._v4_assert_patched = True

        _install_sm86_small_batch_gemm_patch(_of)
    except Exception as error:
        if _sm86_small_batch_gemm_patch_state not in {
            "incompatible",
            "dispatch_incompatible",
        }:
            _set_sm86_small_batch_gemm_patch_failure(
                (
                    "patch_error"
                    if _sm86_small_batch_gemm_enabled()
                    else "disabled_fallback"
                ),
                error,
            )
        if _sm86_small_batch_gemm_enabled():
            raise RuntimeError(
                "SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM=1 but the "
                "triton_kernels specialization could not be installed"
            ) from error


def _swizzle_mxfp4_strided(quant_tensor: torch.Tensor, scale: torch.Tensor):
    """Wrap raw MXFP4 weight + ue8m0 scale into triton_kernels Tensor objects
    using StridedLayout (no swizzle), suitable for the simulated-MXFP non-
    persistent path on capabilities outside the trtllm whitelist. Origin:
    sglang 本身.

    Matches the API contract of sglang.srt.layers.quantization.mxfp4
    `_swizzle_mxfp4`: returns (Tensor[FP4], InFlexData(), Tensor[ue8m0])."""
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout

    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), StridedLayout
    )
    scale = convert_layout(wrap_torch_tensor(scale), StridedLayout)
    return quant_tensor, InFlexData(), scale


def convert_v4_weights_to_triton_kernels(
    w13: torch.Tensor,  # [E, 2*N_int, K//2] int8 packed FP4
    w13_scale: torch.Tensor,  # [E, 2*N_int, K//group] float8_e8m0fnu (or uint8)
    w2: torch.Tensor,  # [E, K, N_int//2] int8 packed FP4
    w2_scale: torch.Tensor,  # [E, K, N_int//group] float8_e8m0fnu
    *,
    num_warps: int = 4,
) -> Tuple:
    """Apply `_swizzle_mxfp4` from sglang.srt.layers.quantization.mxfp4 to
    each weight tensor and build the matching `PrecisionConfig`. The same
    swizzle is used by sglang's existing OAI MXFP4 path (see `mxfp4.py:136`).

    Returns:
        (w13_swiz, w13_pcg, w2_swiz, w2_pcg)
        where w13/w2 are `triton_kernels.Tensor` (FP4 layout) and
        pcg are `triton_kernels.matmul_ogs.PrecisionConfig`.
    """
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    _patch_strided_mxfp()

    if fused_t5_moe_enabled():
        global _fused_t5_moe_conversion_count

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(
            w13.device
        ) != (8, 6):
            raise RuntimeError(
                f"{_SM86_FUSED_T5_MOE_ENV}=1 requires exact SM86 CUDA weights"
            )
        # FusedActivation.reduce_n=2 pairs adjacent accumulator columns.
        # Interleave both packed codes and their per-output-row ue8m0 scales
        # before wrapping either tensor in the immutable strided layout.
        w13 = _interleave_gate_up_rows(w13)
        w13_scale = _interleave_gate_up_rows(w13_scale)
        _fused_t5_moe_conversion_count += 1

    # Wrap raw scale as float8_e8m0fnu if it came in as uint8/float32.
    if w13_scale.dtype != torch.float8_e8m0fnu:
        if w13_scale.dtype == torch.uint8:
            w13_scale = w13_scale.view(torch.float8_e8m0fnu)
        elif w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
    if w2_scale.dtype != torch.float8_e8m0fnu:
        if w2_scale.dtype == torch.uint8:
            w2_scale = w2_scale.view(torch.float8_e8m0fnu)
        elif w2_scale.dtype == torch.float32:
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)

    # The packed FP4 weight is stored as int8 in safetensors; the matmul_ogs
    # kernel asserts the underlying torch dtype is uint8 (or fp8). View
    # without copy.
    if w13.dtype != torch.uint8:
        w13 = w13.view(torch.uint8)
    if w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    if _use_strided_layout():
        # Non-trtllm-whitelist capability: bypass Blackwell DC swizzle (which
        # needs Hopper TMA / cluster shared mem) and use StridedLayout +
        # simulated MXFP non-persistent kernel.
        w13_swiz, w13_flex, w13_scale_swiz = _swizzle_mxfp4_strided(w13, w13_scale)
        w2_swiz, w2_flex, w2_scale_swiz = _swizzle_mxfp4_strided(w2, w2_scale)
    else:
        # SM_100 (cap=(10,0)): use the upstream swizzle from sglang.mxfp4.
        # Reached only via the SGLANG_V4_USE_TRITON_KERNELS=1 force-override;
        # the default dispatch routes this capability to the trtllm path.
        from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4

        w13_swiz, w13_flex, w13_scale_swiz = _swizzle_mxfp4(w13, w13_scale, num_warps)
        w2_swiz, w2_flex, w2_scale_swiz = _swizzle_mxfp4(w2, w2_scale, num_warps)

    w13_pcg = PrecisionConfig(
        weight_scale=w13_scale_swiz,
        flex_ctx=FlexCtx(rhs_data=w13_flex),
    )
    w2_pcg = PrecisionConfig(
        weight_scale=w2_scale_swiz,
        flex_ctx=FlexCtx(rhs_data=w2_flex),
    )

    return w13_swiz, w13_pcg, w2_swiz, w2_pcg


# -----------------------------------------------------------------------------
# Apply: V4-Flash MoE forward via matmul_ogs
# -----------------------------------------------------------------------------


def apply_v4_triton_kernels_moe(
    *,
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13_swiz,  # triton_kernels.Tensor (FP4) [E, K, 2*N]
    w13_pcg,  # PrecisionConfig
    w2_swiz,  # triton_kernels.Tensor (FP4) [E, N, K]
    w2_pcg,  # PrecisionConfig
    topk_weights: torch.Tensor,  # [M, n_topk] bf16/float
    topk_ids: torch.Tensor,  # [M, n_topk] int
    intermediate_size: int,  # per-partition N
    num_experts: int,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: Optional[float] = None,
    caller_output: Optional[torch.Tensor] = None,
    fused_t5_moe: bool = False,
    gpu_experts_mask: torch.Tensor | None = None,
    logical_to_gpu_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run V4 sparse MoE through `triton_kernels.matmul_ogs`.

    Deterministic across runs (no atomic-add reduction order issues) and
    byte-level reproducible under same input.

    Activation: silu_and_mul (V4 default), applied between the two GEMMs.

    `swiglu_limit` (DSV4 2604B): if not None, applies the same gate/up
    asymmetric clamp the trtllm path's `gemm1_clamp_limit` and the
    deep_gemm path's `_apply_swiglu_limit` use, on the gemm1 output before
    silu_and_mul:
        gate = clamp(gate, max=limit)            # one-sided
        up   = clamp(up, min=-limit, max=limit)  # symmetric
    Origin: sglang 本身 (matches `moe_runner/deep_gemm.py:_apply_swiglu_limit`).
    """
    from triton_kernels.matmul_ogs import matmul_ogs

    # Refresh strided-layout patch (cheap idempotent guard) in case apply
    # runs in a process where target_info was re-imported.
    _patch_strided_mxfp()

    M, K = hidden_states.shape
    N = intermediate_size

    gemm2_output = None
    if caller_output is not None:
        if (
            caller_output.shape != hidden_states.shape
            or caller_output.dtype != hidden_states.dtype
            or caller_output.device != hidden_states.device
            or not caller_output.is_contiguous()
        ):
            raise ValueError(
                "V4 MXFP4 caller-owned output must be a contiguous tensor "
                "matching hidden_states shape, dtype, and device"
            )
        # matmul_ogs uses a leading batch dimension for caller-owned output,
        # then returns a squeezed view.  The KT hybrid path intentionally
        # supplies hidden_states here: its CPU staging copy was enqueued first,
        # and GEMM1 consumes the input before GEMM2 overwrites it on the same
        # CUDA stream.
        gemm2_output = caller_output.unsqueeze(0)

    # Build routing data from sglang topk → triton_kernels (RoutingData,
    # GatherIndx, ScatterIndx). Note: this rebuilds per-call. Cheap
    # (O(M * n_topk)) compared to the gemms themselves.
    routing_data, gather_indx, scatter_indx = _make_routing_data_v4(
        topk_ids,
        topk_weights,
        num_experts,
        gpu_experts_mask=gpu_experts_mask,
        logical_to_gpu_index=logical_to_gpu_index,
    )

    # gemm1: hidden_states (M, K) @ w13 → (M*topk, 2*N) bf16
    if fused_t5_moe:
        global _fused_t5_kt_routing_apply_count
        global _fused_t5_moe_apply_count

        if not fused_t5_moe_enabled():
            raise RuntimeError(
                "fused T5 MoE weights were selected without the matching runtime opt-in"
            )
        _fused_t5_moe_apply_count += 1
        if gpu_experts_mask is not None:
            _fused_t5_kt_routing_apply_count += 1
        # With split-K, matmul_ogs moves this epilogue into its deterministic
        # grouped reduction.  The reduction writes only N BF16 values rather
        # than materializing 2*N BF16 W13 output plus a second activation
        # buffer/kernel.  With split-K=1 it executes in the W13 matmul epilogue.
        intermediate2 = matmul_ogs(
            hidden_states,
            w13_swiz,
            None,
            routing_data,
            gather_indx=gather_indx,
            precision_config=w13_pcg,
            fused_activation=_make_dsv4_fused_activation(swiglu_limit),
        )
    else:
        from sgl_kernel import silu_and_mul

        intermediate1 = matmul_ogs(
            hidden_states,
            w13_swiz,
            None,  # bias
            routing_data,
            gather_indx=gather_indx,
            precision_config=w13_pcg,
        )
        # intermediate1 shape: [M*topk, 2*N]; layout = [gate, up] along last dim.
        # We skipped reorder_w1w3_to_w3w1 for this path so the natural [w1, w3]
        # = [gate, up] order from the checkpoint is preserved.
        if swiglu_limit is not None:
            # 2604B asymmetric SwiGLU clamp. View slices and clamp_ in place to
            # avoid chunk+cat copy; safe because intermediate1 is a fresh
            # matmul_ogs output, not a cached buffer.
            N_int = intermediate1.shape[-1] // 2
            intermediate1[..., :N_int].clamp_(max=swiglu_limit)
            intermediate1[..., N_int:].clamp_(min=-swiglu_limit, max=swiglu_limit)
        M_topk = intermediate1.shape[0]
        intermediate2 = torch.empty(
            (M_topk, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        silu_and_mul(intermediate1.view(-1, 2 * N), intermediate2)

    # gemm2: (M*topk, N) @ w2 → (M, K), with gammas=topk_weights for combine
    output = matmul_ogs(
        intermediate2,
        w2_swiz,
        None,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_pcg,
        gammas=routing_data.gate_scal,
        y=gemm2_output,
    )
    if caller_output is not None and output.data_ptr() != caller_output.data_ptr():
        raise RuntimeError("V4 MXFP4 GEMM2 did not preserve caller-owned output")

    # routed_scaling_factor is NOT applied here; the caller
    # (mxfp4_deepseek.apply) handles it to stay consistent with the trtllm
    # path and avoid double-apply when FUSE_RSF_SHARED_ADD is enabled.

    return output
