from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization import v4_triton_kernels_moe as v4_moe
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@dataclass(frozen=True)
class _FakeOptFlags:
    block_n: int = 64
    split_k: int = 1
    num_stages: int = 2
    num_warps: int = 2


def _fake_make_default_opt_flags_nvidia(
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    precision_config,
    m,
    n,
    k,
    routing_data,
    can_use_persistent_tma,
    can_use_fused_scatter,
    enforce_bitwise_invariance,
    epilogue_effective_itemsize,
    constraints,
):
    return _FakeOptFlags()


def _reset_small_batch_gemm_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(v4_moe, "_sm86_small_batch_gemm_patch_state", "not_attempted")
    monkeypatch.setattr(v4_moe, "_sm86_small_batch_gemm_patch_error", None)
    monkeypatch.setattr(v4_moe, "_sm86_small_batch_gemm_selection_counts", {})


def _small_batch_gemm_call_args(*, rows: int, expert_count: int, k: int) -> tuple:
    from triton_kernels.tensor import FP4

    return (
        object(),
        torch.bfloat16,
        FP4,
        SimpleNamespace(weight_scale=object()),
        rows * 8,
        4096,
        k,
        SimpleNamespace(
            n_expts_tot=expert_count,
            n_expts_act=8,
            expt_data=object(),
        ),
        False,
        False,
        False,
        2,
        {"is_persistent": False},
    )


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("the fixed small-row MXFP4 router is specific to SM86")


@triton.jit
def _fused_silu_mul_test_kernel(
    input_pointer,
    output_pointer,
    pair_count,
    LIMIT: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    input_offsets = tl.arange(0, BLOCK_PAIRS * 2)
    values = tl.load(
        input_pointer + input_offsets,
        mask=input_offsets < pair_count * 2,
        other=0.0,
    )
    values = tl.reshape(values, (1, BLOCK_PAIRS * 2))
    activated = v4_moe._dsv4_fused_silu_mul(values, LIMIT)
    activated = tl.reshape(activated, (BLOCK_PAIRS,))
    output_offsets = tl.arange(0, BLOCK_PAIRS)
    tl.store(
        output_pointer + output_offsets,
        activated,
        mask=output_offsets < pair_count,
    )


def _route_inputs(
    rows: int, active_slots: int, expert_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    route_ids = torch.full((rows, 6), -1, dtype=torch.int32, device="cuda")
    route_weights = torch.empty((rows, 6), dtype=torch.float32, device="cuda")
    base_weights = torch.arange(1, 7, dtype=torch.float32, device="cuda")
    base_weights /= base_weights.sum()
    for row in range(rows):
        valid_slots = min(active_slots, expert_count)
        route_ids[row, :valid_slots] = (
            torch.arange(valid_slots, dtype=torch.int32, device="cuda") + 3 * row
        ) % expert_count
        route_weights[row] = base_weights.roll(row)
    return route_ids, route_weights


def _assert_routing_equal(baseline, candidate, expert_count: int) -> None:
    baseline_data, baseline_gather, baseline_scatter = baseline
    candidate_data, candidate_gather, candidate_scatter = candidate

    assert baseline_data.n_expts_tot == candidate_data.n_expts_tot == expert_count
    assert baseline_data.n_expts_act == candidate_data.n_expts_act == 8
    torch.testing.assert_close(
        candidate_data.expt_hist,
        baseline_data.expt_hist,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate_gather.src_indx,
        baseline_gather.src_indx,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate_gather.dst_indx,
        baseline_gather.dst_indx,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate_scatter.src_indx,
        baseline_scatter.src_indx,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate_scatter.dst_indx,
        baseline_scatter.dst_indx,
        rtol=0,
        atol=0,
    )

    # routing_from_bitmatrix leaves the unused tail uninitialized.  Every
    # valid expert-major scale, including its stable sort order, must match.
    valid_gates = int(baseline_data.expt_hist.sum().item())
    torch.testing.assert_close(
        candidate_data.gate_scal[:valid_gates],
        baseline_data.gate_scal[:valid_gates],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate_data.expt_data.token_offs_raw,
        baseline_data.expt_data.token_offs_raw,
        rtol=0,
        atol=0,
    )
    for block_m in (16, 32, 64, 128):
        torch.testing.assert_close(
            candidate_data.expt_data.token_offs_pad[block_m],
            baseline_data.expt_data.token_offs_pad[block_m],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate_data.expt_data.block_pid_map[block_m],
            baseline_data.expt_data.block_pid_map[block_m],
            rtol=0,
            atol=0,
        )


def test_interleave_gate_up_rows_preserves_expert_and_inner_axes() -> None:
    source = torch.arange(2 * 6 * 4, dtype=torch.int32).view(2, 6, 4)
    result = v4_moe._interleave_gate_up_rows(source)
    expected = source[:, [0, 3, 1, 4, 2, 5], :]

    assert result.is_contiguous()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_interleave_native_e8m0_scales_uses_exact_byte_carrier() -> None:
    source_bytes = torch.arange(2 * 6 * 4, dtype=torch.uint8).view(2, 6, 4)
    source = source_bytes.view(torch.float8_e8m0fnu)
    result = v4_moe._interleave_gate_up_rows(source)
    expected_bytes = source_bytes[:, [0, 3, 1, 4, 2, 5], :]

    assert result.dtype == torch.float8_e8m0fnu
    assert result.is_contiguous()
    torch.testing.assert_close(result.view(torch.uint8), expected_bytes, rtol=0, atol=0)


def test_adapter_execution_uses_conversion_process_predicate(monkeypatch) -> None:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_deepseek import (
        DeepSeekMxfp4MoEMethod,
    )

    captured: dict[str, object] = {}

    def fake_apply_v4_triton_kernels_moe(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["hidden_states"])

    monkeypatch.setattr(
        v4_moe,
        "apply_v4_triton_kernels_moe",
        fake_apply_v4_triton_kernels_moe,
    )
    monkeypatch.setattr(v4_moe, "fused_t5_moe_enabled", lambda: True)

    method = object.__new__(DeepSeekMxfp4MoEMethod)
    method._kt_compact_ids = True
    method._gemm1_clamp_limit_tensor = None
    hidden_states = torch.randn(1, 8, dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 6), dtype=torch.int32)
    topk_weights = torch.full((1, 6), 1.0 / 6.0, dtype=torch.float32)
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.empty((1, 1)),
        ),
    )
    layer = SimpleNamespace(
        _v4_tk_path=True,
        _v4_tk_fused_t5_moe=False,
        _v4_tk_w13=object(),
        _v4_tk_w13_pcg=object(),
        _v4_tk_w2=object(),
        _v4_tk_w2_pcg=object(),
        _v4_tk_intermediate_size=4,
        _v4_tk_num_experts=1,
        moe_runner_config=SimpleNamespace(
            routed_scaling_factor=1.0,
            swiglu_limit=None,
        ),
    )

    result = method._apply(
        layer,
        dispatch_output,
        caller_output=None,
        gpu_experts_mask=torch.ones(1, dtype=torch.bool),
        logical_to_gpu_index=torch.zeros(1, dtype=torch.int32),
    )

    assert result.hidden_states.shape == hidden_states.shape
    assert captured["fused_t5_moe"] is True


@pytest.mark.parametrize("limit", [None, 7.0])
def test_fused_silu_mul_matches_bf16_materialization_and_graph_replay(
    limit: float | None,
) -> None:
    _require_sm86()
    pair_count = 73
    block_pairs = 128
    source = torch.randn(pair_count, 2, dtype=torch.bfloat16, device="cuda") * 9
    output = torch.empty(pair_count, dtype=torch.bfloat16, device="cuda")

    def reference(values: torch.Tensor) -> torch.Tensor:
        gate = values[:, 0].float()
        up = values[:, 1].float()
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

    _fused_silu_mul_test_kernel[(1,)](
        source,
        output,
        pair_count,
        LIMIT=limit,
        BLOCK_PAIRS=block_pairs,
        num_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference(source), rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _fused_silu_mul_test_kernel[(1,)](
            source,
            output,
            pair_count,
            LIMIT=limit,
            BLOCK_PAIRS=block_pairs,
            num_warps=4,
        )
    changed = torch.randn_like(source) * 13
    source.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference(changed), rtol=0, atol=0)


def test_small_row_router_is_opt_in_and_sm86_only(monkeypatch) -> None:
    cuda_device = torch.device("cuda")
    topk_ids = SimpleNamespace(
        ndim=2,
        shape=(6, 6),
        device=cuda_device,
        dtype=torch.int32,
    )
    topk_weights = SimpleNamespace(
        ndim=2,
        shape=(6, 6),
        device=cuda_device,
        dtype=torch.float32,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 6))

    monkeypatch.delenv(v4_moe._SMALL_ROW_ROUTING_ENV, raising=False)
    assert not v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 13)

    monkeypatch.setenv(v4_moe._SMALL_ROW_ROUTING_ENV, "1")
    assert v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 1)
    assert v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 13)
    assert v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 14)
    assert v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 22)
    assert not v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 0)
    assert not v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 23)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    assert not v4_moe._small_row_routing_is_eligible(topk_ids, topk_weights, 13)


def test_small_row_router_kt_remap_contract_is_exact(monkeypatch) -> None:
    cuda_device = torch.device("cuda")
    topk_ids = SimpleNamespace(
        ndim=2,
        shape=(5, 6),
        device=cuda_device,
        dtype=torch.int32,
    )
    topk_weights = SimpleNamespace(
        ndim=2,
        shape=(5, 6),
        device=cuda_device,
        dtype=torch.float32,
    )
    gpu_mask = SimpleNamespace(
        ndim=1,
        shape=(256,),
        device=cuda_device,
        dtype=torch.bool,
        is_contiguous=lambda: True,
    )
    logical_to_local = SimpleNamespace(
        ndim=1,
        shape=(256,),
        device=cuda_device,
        dtype=torch.int32,
        is_contiguous=lambda: True,
    )
    monkeypatch.setenv(v4_moe._SMALL_ROW_ROUTING_ENV, "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 6))

    assert v4_moe._small_row_routing_is_eligible(
        topk_ids,
        topk_weights,
        14,
        gpu_mask,
        logical_to_local,
    )
    assert not v4_moe._small_row_routing_is_eligible(
        topk_ids,
        topk_weights,
        14,
        gpu_mask,
        None,
    )
    wrong_dtype = SimpleNamespace(
        **{
            **logical_to_local.__dict__,
            "dtype": torch.int64,
        }
    )
    assert not v4_moe._small_row_routing_is_eligible(
        topk_ids,
        topk_weights,
        14,
        gpu_mask,
        wrong_dtype,
    )


def test_sm86_small_batch_gemm_gate_is_exact_and_opt_in(monkeypatch) -> None:
    from triton_kernels.tensor import FP4

    precision_config = SimpleNamespace(weight_scale=object())
    routing_data = SimpleNamespace(
        n_expts_tot=14,
        n_expts_act=8,
        expt_data=object(),
    )
    common = {
        "rhs_dtype": FP4,
        "precision_config": precision_config,
        "m": 8,
        "n": 4096,
        "k": 4096,
        "routing_data": routing_data,
        "constraints": {"is_persistent": False},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))

    monkeypatch.delenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, raising=False)
    assert not v4_moe._sm86_small_batch_gemm_is_eligible(**common)

    monkeypatch.setenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, "1")
    for expert_count in range(14, 23):
        for rows in range(1, 7):
            assert v4_moe._sm86_small_batch_gemm_is_eligible(
                **{
                    **common,
                    "m": rows * 8,
                    "routing_data": SimpleNamespace(
                        n_expts_tot=expert_count,
                        n_expts_act=8,
                        expt_data=object(),
                    ),
                }
            )
    assert v4_moe._sm86_small_batch_gemm_is_eligible(**{**common, "k": 2048})

    assert not v4_moe._sm86_small_batch_gemm_is_eligible(**{**common, "m": 56})
    assert not v4_moe._sm86_small_batch_gemm_is_eligible(**{**common, "n": 2048})
    assert not v4_moe._sm86_small_batch_gemm_is_eligible(**{**common, "k": 1024})
    assert not v4_moe._sm86_small_batch_gemm_is_eligible(
        **{
            **common,
            "routing_data": SimpleNamespace(
                n_expts_tot=23,
                n_expts_act=8,
                expt_data=object(),
            ),
        }
    )
    assert not v4_moe._sm86_small_batch_gemm_is_eligible(
        **{**common, "constraints": {"is_persistent": False, "split_k": 1}}
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    assert not v4_moe._sm86_small_batch_gemm_is_eligible(**common)


def test_sm86_small_batch_patch_fails_closed_on_api_drift(monkeypatch) -> None:
    _reset_small_batch_gemm_telemetry(monkeypatch)
    monkeypatch.setenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, "1")

    def incompatible_signature(single_argument):
        return single_argument

    opt_flags_module = SimpleNamespace(
        make_default_opt_flags_nvidia=incompatible_signature
    )
    with pytest.raises(RuntimeError, match="incompatible triton_kernels"):
        v4_moe._install_sm86_small_batch_gemm_patch(opt_flags_module)

    telemetry = v4_moe.get_sm86_small_batch_gemm_telemetry()
    assert telemetry["patch_state"] == "incompatible"
    assert telemetry["patch_installed"] is False
    assert "single_argument" in telemetry["patch_error"]


def test_sm86_small_batch_patch_disabled_mode_retains_fallback(monkeypatch) -> None:
    _reset_small_batch_gemm_telemetry(monkeypatch)
    monkeypatch.delenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, raising=False)

    def incompatible_signature(single_argument):
        return single_argument

    opt_flags_module = SimpleNamespace(
        make_default_opt_flags_nvidia=incompatible_signature
    )
    assert not v4_moe._install_sm86_small_batch_gemm_patch(opt_flags_module)
    assert opt_flags_module.make_default_opt_flags_nvidia is incompatible_signature
    assert opt_flags_module.make_default_opt_flags_nvidia("fallback") == "fallback"
    assert (
        v4_moe.get_sm86_small_batch_gemm_telemetry()["patch_state"]
        == "disabled_fallback"
    )


def test_sm86_small_batch_patch_rejects_dispatch_drift(monkeypatch) -> None:
    _reset_small_batch_gemm_telemetry(monkeypatch)
    monkeypatch.setenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    opt_flags_module = SimpleNamespace(
        make_default_opt_flags_nvidia=_fake_make_default_opt_flags_nvidia
    )
    assert v4_moe._install_sm86_small_batch_gemm_patch(opt_flags_module)

    args = _small_batch_gemm_call_args(rows=4, expert_count=14, k=4096)
    with pytest.raises(RuntimeError, match="thirteen positional"):
        opt_flags_module.make_default_opt_flags_nvidia(*args[:-1], constraints=args[-1])

    # Re-install a clean wrapper after the first deliberate fatal dispatch.
    _reset_small_batch_gemm_telemetry(monkeypatch)
    opt_flags_module = SimpleNamespace(
        make_default_opt_flags_nvidia=_fake_make_default_opt_flags_nvidia
    )
    assert v4_moe._install_sm86_small_batch_gemm_patch(opt_flags_module)
    drifted_args = (*args[:-1], {"is_persistent": False, "split_k": 1})
    with pytest.raises(RuntimeError, match="constraints drifted"):
        opt_flags_module.make_default_opt_flags_nvidia(*drifted_args)
    assert (
        v4_moe.get_sm86_small_batch_gemm_telemetry()["patch_state"]
        == "dispatch_incompatible"
    )


def test_sm86_small_batch_patch_selects_every_live_signature(monkeypatch) -> None:
    _reset_small_batch_gemm_telemetry(monkeypatch)
    monkeypatch.setenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    opt_flags_module = SimpleNamespace(
        make_default_opt_flags_nvidia=_fake_make_default_opt_flags_nvidia
    )
    assert v4_moe._install_sm86_small_batch_gemm_patch(opt_flags_module)

    for expert_count in range(14, 23):
        for rows in range(1, 7):
            for k in (2048, 4096):
                selected = opt_flags_module.make_default_opt_flags_nvidia(
                    *_small_batch_gemm_call_args(
                        rows=rows,
                        expert_count=expert_count,
                        k=k,
                    )
                )
                assert selected == _FakeOptFlags(
                    block_n=128,
                    split_k=2,
                    num_stages=4,
                    num_warps=4,
                )

    # A neighboring prefill shape must keep the package default even while the
    # exact decode specialization is enabled.
    unrelated_args = list(_small_batch_gemm_call_args(rows=6, expert_count=14, k=4096))
    unrelated_args[4] = 56
    assert (
        opt_flags_module.make_default_opt_flags_nvidia(*unrelated_args)
        == _FakeOptFlags()
    )

    telemetry = v4_moe.get_sm86_small_batch_gemm_telemetry()
    assert telemetry["patch_state"] == "installed"
    assert telemetry["patch_installed"] is True
    assert telemetry["selection_count"] == 9 * 6 * 2
    assert len(telemetry["observed_signatures"]) == 9 * 6 * 2
    assert telemetry["selected_config"] == {
        "block_n": 128,
        "split_k": 2,
        "num_stages": 4,
        "num_warps": 4,
    }


@pytest.mark.parametrize("expert_count", range(14, 23))
def test_small_row_router_matches_baseline_and_replays_dynamic_graph(
    monkeypatch,
    expert_count: int,
) -> None:
    _require_sm86()
    monkeypatch.delenv(v4_moe._SMALL_ROW_ROUTING_ENV, raising=False)

    for rows in range(1, 7):
        route_ids, route_weights = _route_inputs(
            rows, active_slots=2, expert_count=expert_count
        )
        # Compile before graph capture.
        v4_moe._make_small_row_routing_data_v4(route_ids, route_weights, expert_count)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = v4_moe._make_small_row_routing_data_v4(
                route_ids,
                route_weights,
                expert_count,
            )

        # All route values stay dynamic across replay.  The 42 cases span an
        # empty local route through all six local routes for each graph tier.
        for active_slots in range(7):
            changed_ids, changed_weights = _route_inputs(
                rows, active_slots, expert_count
            )
            route_ids.copy_(changed_ids)
            route_weights.copy_(changed_weights)
            graph.replay()
            baseline = v4_moe._make_routing_data_v4(
                route_ids,
                route_weights,
                expert_count,
            )
            torch.cuda.synchronize()
            _assert_routing_equal(baseline, captured, expert_count)

        del graph
        torch.cuda.empty_cache()


@pytest.mark.parametrize("local_expert_count", [14, 22])
def test_small_row_router_fuses_dynamic_kt_mask_remap_in_graph(
    monkeypatch,
    local_expert_count: int,
) -> None:
    _require_sm86()
    monkeypatch.setenv(v4_moe._SMALL_ROW_ROUTING_ENV, "1")
    rows = 5
    global_expert_count = 256
    logical_gpu_ids = (
        torch.arange(local_expert_count, dtype=torch.int32, device="cuda") * 11 + 3
    ) % global_expert_count
    gpu_mask = torch.zeros(global_expert_count, dtype=torch.bool, device="cuda")
    gpu_mask[logical_gpu_ids.long()] = True
    logical_to_local = torch.full(
        (global_expert_count,), -1, dtype=torch.int32, device="cuda"
    )
    logical_to_local[logical_gpu_ids.long()] = torch.arange(
        local_expert_count, dtype=torch.int32, device="cuda"
    )
    route_ids = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    route_weights = torch.arange(1, 7, dtype=torch.float32, device="cuda")
    route_weights = route_weights.repeat(rows, 1)
    route_weights /= route_weights.sum(dim=1, keepdim=True)

    # Compile this exact remap specialization before capture.
    route_ids.copy_(logical_gpu_ids[:6].repeat(rows, 1))
    v4_moe._make_small_row_routing_data_v4(
        route_ids,
        route_weights,
        local_expert_count,
        gpu_experts_mask=gpu_mask,
        logical_to_gpu_index=logical_to_local,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = v4_moe._make_small_row_routing_data_v4(
            route_ids,
            route_weights,
            local_expert_count,
            gpu_experts_mask=gpu_mask,
            logical_to_gpu_index=logical_to_local,
        )

    for replay in range(7):
        logical_routes = torch.empty_like(route_ids)
        for row in range(rows):
            for slot in range(6):
                if slot < replay:
                    logical_routes[row, slot] = logical_gpu_ids[
                        (row * 3 + slot + replay) % local_expert_count
                    ]
                else:
                    # A valid global ID owned by the CPU must become -1.
                    candidate = (row * 17 + slot * 7 + 1) % global_expert_count
                    while bool(gpu_mask[candidate].item()):
                        candidate = (candidate + 1) % global_expert_count
                    logical_routes[row, slot] = candidate
        route_ids.copy_(logical_routes)
        route_weights.copy_(route_weights.roll(1, dims=1))
        graph.replay()

        safe_ids = route_ids.clamp(min=0, max=global_expert_count - 1)
        remapped = torch.where(
            gpu_mask[safe_ids.long()],
            logical_to_local[safe_ids.long()],
            torch.full_like(route_ids, -1),
        )
        baseline = v4_moe._make_routing_data_v4(
            remapped,
            route_weights,
            local_expert_count,
        )
        torch.cuda.synchronize()
        _assert_routing_equal(baseline, captured, local_expert_count)


def test_full_fused_t5_moe_matches_materialized_split_k_and_replays_graph(
    monkeypatch,
) -> None:
    """Exercise the real SM86 MXFP4 W13/activation/W2 implementation.

    The dimensions deliberately select the production small-row split-K=2
    signature; this catches errors that an isolated activation test cannot,
    including packed-row/scale permutation and grouped-reduction placement.
    """

    _require_sm86()
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

    torch.manual_seed(1234)
    rows = 5
    hidden_size = 2048
    intermediate_size = 4096
    expert_count = 1
    w13_bf16 = (
        torch.randn(
            expert_count,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.02
    )
    w2_bf16 = (
        torch.randn(
            expert_count,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.02
    )
    w13, w13_scale = downcast_to_mxfp(w13_bf16, torch.uint8, axis=-1)
    w2, w2_scale = downcast_to_mxfp(w2_bf16, torch.uint8, axis=-1)
    # Checkpoint safetensors expose E8M0 scales in their native dtype rather
    # than the uint8 carrier used by the synthetic downcast helper.
    w13_scale = w13_scale.view(torch.float8_e8m0fnu)
    w2_scale = w2_scale.view(torch.float8_e8m0fnu)
    del w13_bf16, w2_bf16

    monkeypatch.setenv(v4_moe._SMALL_ROW_ROUTING_ENV, "1")
    monkeypatch.setenv(v4_moe._SM86_SMALL_BATCH_GEMM_ENV, "1")
    monkeypatch.delenv(v4_moe._SM86_FUSED_T5_MOE_ENV, raising=False)
    baseline_weights = v4_moe.convert_v4_weights_to_triton_kernels(
        w13,
        w13_scale,
        w2,
        w2_scale,
    )
    monkeypatch.setenv(v4_moe._SM86_FUSED_T5_MOE_ENV, "1")
    fused_weights = v4_moe.convert_v4_weights_to_triton_kernels(
        w13,
        w13_scale,
        w2,
        w2_scale,
    )

    hidden_states = torch.randn(rows, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros((rows, 6), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand((rows, 6), dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    def run(weights, *, fused: bool) -> torch.Tensor:
        return v4_moe.apply_v4_triton_kernels_moe(
            hidden_states=hidden_states,
            w13_swiz=weights[0],
            w13_pcg=weights[1],
            w2_swiz=weights[2],
            w2_pcg=weights[3],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            intermediate_size=intermediate_size,
            num_experts=expert_count,
            swiglu_limit=7.0,
            fused_t5_moe=fused,
        )

    baseline = run(baseline_weights, fused=False)
    fused = run(fused_weights, fused=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(fused, baseline, rtol=2e-2, atol=3.125e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(fused_weights, fused=True)
    hidden_states.normal_()
    topk_weights.uniform_()
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    graph.replay()
    baseline = run(baseline_weights, fused=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, baseline, rtol=2e-2, atol=3.125e-2)
