from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
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
