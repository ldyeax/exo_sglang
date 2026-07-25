from __future__ import annotations

from dataclasses import replace

import pytest
from sglang.srt.disaggregation.glm52_intra_node_policy import (
    DWAGON_HARDWARE_PROFILE,
    GLM52CapacityContract,
    GLM52ExecutorCompatibility,
    GLM52IntraNodeActionKind,
    GLM52IntraNodeAdmissionController,
    GLM52IntraNodeContractError,
    GLM52IntraNodeContracts,
    GLM52IntraNodePolicyConfig,
    GLM52IntraNodeRoute,
    GLM52IntraNodeRuntimeState,
    GLM52RouteRequest,
    GLM52SharedHostWeights,
    plan_glm52_intra_node_route,
)

_GIB = 1024**3


def _config(**overrides) -> GLM52IntraNodePolicyConfig:
    values = {
        "enabled": True,
        "long_prompt_threshold": 4096,
        "chunked_prefill_size": 2048,
        "gpu_ids": (0, 1),
    }
    values.update(overrides)
    return GLM52IntraNodePolicyConfig(**values)


def _executors(**overrides) -> GLM52ExecutorCompatibility:
    values = {
        "architecture": "GlmMoeDsaForCausalLM",
        "hardware_profile": DWAGON_HARDWARE_PROFILE,
        "gpu_ids": (0, 1),
        "pipeline_parallel_size": 1,
        "chunked_prefill_ready": True,
        "chunked_prefill_tensor_parallel_size": 2,
        "distributed_slp_ready": True,
        "distributed_slp_tensor_parallel_size": 2,
        "distributed_slp_admission": "kt-stream-prefill:ring=2:chunk=4",
        # A unified server does not imply that the separate TP1 process trees
        # are live. Split tests opt in with an explicit runtime proof.
        "split_slp_ready": False,
        "split_slp_tensor_parallel_size": 1,
        "split_decode_ready": False,
        "split_decode_tensor_parallel_size": 1,
        "split_kv_transfer_ready": False,
    }
    values.update(overrides)
    return GLM52ExecutorCompatibility(**values)


def _capacity(
    gpu_ids: tuple[int, ...],
    *,
    max_concurrent_requests: int = 4,
    max_inflight_tokens: int = 32768,
    available_device_bytes: tuple[int, ...] | None = None,
    required_device_bytes: tuple[int, ...] | None = None,
) -> GLM52CapacityContract:
    return GLM52CapacityContract(
        gpu_ids=gpu_ids,
        max_concurrent_requests=max_concurrent_requests,
        max_inflight_tokens=max_inflight_tokens,
        available_device_bytes=(
            available_device_bytes
            if available_device_bytes is not None
            else tuple(2 * _GIB for _ in gpu_ids)
        ),
        required_device_bytes=(
            required_device_bytes
            if required_device_bytes is not None
            else tuple(_GIB for _ in gpu_ids)
        ),
    )


def _shared_weights(**overrides) -> GLM52SharedHostWeights:
    values = {
        "weight_identity": "glm52-amxint4-sha256:abc",
        "allocation_identity": "dwagon-host-pool:0",
        "weight_bytes": 300 * _GIB,
        "host_capacity_bytes": 512 * _GIB,
        "safety_margin_bytes": 64 * _GIB,
        "immutable": True,
        "sharing_verified": True,
        "prefill_reader_ready": True,
        "decode_reader_ready": True,
    }
    values.update(overrides)
    return GLM52SharedHostWeights(**values)


def _contracts(
    *,
    executors: GLM52ExecutorCompatibility | None = None,
    shared_host_weights: GLM52SharedHostWeights | None = None,
) -> GLM52IntraNodeContracts:
    return GLM52IntraNodeContracts(
        executors=executors or _executors(),
        chunked_prefill_capacity=_capacity((0, 1)),
        distributed_slp_capacity=_capacity((0, 1), max_concurrent_requests=1),
        split_slp_capacity=_capacity((0,), max_concurrent_requests=1),
        split_decode_capacity=_capacity((1,)),
        shared_host_weights=shared_host_weights,
    )


def test_short_prompt_routes_to_chunked_prefill() -> None:
    request = GLM52RouteRequest("short", prompt_tokens=4095, max_new_tokens=256)

    plan = plan_glm52_intra_node_route(
        _config(),
        request,
        GLM52IntraNodeRuntimeState(decode_active_requests=2),
        _contracts(),
    )

    assert plan.route == GLM52IntraNodeRoute.CHUNKED_PREFILL
    assert plan.admitted
    # The current executable path remains the unified TP2 worker, not a
    # fabricated one-GPU worker.
    assert len(plan.actions) == 1
    assert plan.actions[0].kind == GLM52IntraNodeActionKind.RUN_CHUNKED_PREFILL
    assert plan.actions[0].gpu_ids == (0, 1)


def test_threshold_token_routes_isolated_prompt_to_distributed_slp() -> None:
    request = GLM52RouteRequest("long", prompt_tokens=4096, max_new_tokens=256)

    plan = plan_glm52_intra_node_route(
        _config(),
        request,
        GLM52IntraNodeRuntimeState(),
        _contracts(),
    )

    assert plan.route == GLM52IntraNodeRoute.DISTRIBUTED_SLP
    assert plan.actions[0].kind == GLM52IntraNodeActionKind.RUN_DISTRIBUTED_SLP
    assert plan.actions[0].gpu_ids == (0, 1)
    assert plan.slp_reserved_tokens == 4096
    assert plan.decode_reserved_tokens == 0


@pytest.mark.parametrize(
    ("executor_overrides", "expected"),
    [
        ({"architecture": "Qwen3ForCausalLM"}, "not an admitted GLM-5.2"),
        ({"distributed_slp_ready": False}, "executor is not ready"),
        ({"distributed_slp_tensor_parallel_size": 1}, "requires TP=2"),
        ({"distributed_slp_admission": None}, "no KT stream-prefill admission"),
    ],
)
def test_distributed_slp_fails_closed_without_compatibility(
    executor_overrides,
    expected: str,
) -> None:
    plan = plan_glm52_intra_node_route(
        _config(),
        GLM52RouteRequest("long", prompt_tokens=8192, max_new_tokens=32),
        GLM52IntraNodeRuntimeState(),
        _contracts(executors=_executors(**executor_overrides)),
    )

    assert plan.route == GLM52IntraNodeRoute.REJECT
    assert expected in plan.reason
    assert plan.actions == ()


def test_distributed_slp_defers_until_prompt_is_isolated() -> None:
    plan = plan_glm52_intra_node_route(
        _config(),
        GLM52RouteRequest("long", prompt_tokens=8192, max_new_tokens=32),
        GLM52IntraNodeRuntimeState(other_active_requests=1),
        _contracts(),
    )

    assert plan.route == GLM52IntraNodeRoute.DEFER
    assert "isolated long prompt" in plan.reason


def test_active_decode_rejects_unwired_split_instead_of_using_tp2_slp() -> None:
    plan = plan_glm52_intra_node_route(
        _config(),
        GLM52RouteRequest("long", prompt_tokens=8192, max_new_tokens=512),
        GLM52IntraNodeRuntimeState(
            decode_active_requests=1,
            decode_tokens_in_use=4096,
        ),
        _contracts(shared_host_weights=_shared_weights()),
    )

    assert plan.route == GLM52IntraNodeRoute.REJECT
    assert "one-GPU SLP executor is not ready" in plan.reason
    assert "SLP-to-decode KV transfer is not ready" in plan.reason


def test_active_decode_routes_to_split_only_with_every_contract() -> None:
    split_executors = _executors(
        split_slp_ready=True,
        split_decode_ready=True,
        split_kv_transfer_ready=True,
    )
    request = GLM52RouteRequest("split", prompt_tokens=8192, max_new_tokens=512)

    plan = plan_glm52_intra_node_route(
        _config(),
        request,
        GLM52IntraNodeRuntimeState(
            decode_active_requests=1,
            decode_tokens_in_use=4096,
        ),
        _contracts(
            executors=split_executors,
            shared_host_weights=_shared_weights(),
        ),
    )

    assert plan.route == GLM52IntraNodeRoute.SPLIT_SLP_DECODE
    assert [action.kind for action in plan.actions] == [
        GLM52IntraNodeActionKind.RUN_SINGLE_GPU_SLP,
        GLM52IntraNodeActionKind.TRANSFER_KV_TO_DECODE,
        GLM52IntraNodeActionKind.CONTINUE_SINGLE_GPU_DECODE,
    ]
    assert [action.gpu_ids for action in plan.actions] == [(0,), (0, 1), (1,)]
    assert plan.slp_reserved_tokens == 8192
    assert plan.decode_reserved_tokens == 8704


@pytest.mark.parametrize(
    ("shared_weights", "expected"),
    [
        (None, "contract is missing"),
        (_shared_weights(sharing_verified=False), "was not verified"),
        (_shared_weights(immutable=False), "not immutable"),
        (
            _shared_weights(host_capacity_bytes=350 * _GIB),
            "host capacity is insufficient",
        ),
    ],
)
def test_split_fails_closed_without_shared_host_weights(
    shared_weights,
    expected: str,
) -> None:
    plan = plan_glm52_intra_node_route(
        _config(),
        GLM52RouteRequest("split", prompt_tokens=8192, max_new_tokens=512),
        GLM52IntraNodeRuntimeState(decode_active_requests=1),
        _contracts(
            executors=_executors(
                split_slp_ready=True,
                split_decode_ready=True,
                split_kv_transfer_ready=True,
            ),
            shared_host_weights=shared_weights,
        ),
    )

    assert plan.route == GLM52IntraNodeRoute.REJECT
    assert expected in plan.reason


def test_split_defers_when_decode_token_capacity_is_exhausted() -> None:
    contracts = replace(
        _contracts(
            executors=_executors(
                split_slp_ready=True,
                split_decode_ready=True,
                split_kv_transfer_ready=True,
            ),
            shared_host_weights=_shared_weights(),
        ),
        split_decode_capacity=_capacity((1,), max_inflight_tokens=12000),
    )

    plan = plan_glm52_intra_node_route(
        _config(),
        GLM52RouteRequest("split", prompt_tokens=8192, max_new_tokens=512),
        GLM52IntraNodeRuntimeState(
            decode_active_requests=1,
            decode_tokens_in_use=4096,
        ),
        contracts,
    )

    assert plan.route == GLM52IntraNodeRoute.DEFER
    assert "one-GPU decode token capacity exhausted" in plan.reason


def test_controller_reserves_atomically_and_releases_by_lifecycle() -> None:
    controller = GLM52IntraNodeAdmissionController(_config())
    controller.report_external_state(
        decode_active_requests=1,
        decode_tokens_in_use=1024,
    )
    contracts = _contracts(
        executors=_executors(
            split_slp_ready=True,
            split_decode_ready=True,
            split_kv_transfer_ready=True,
        ),
        shared_host_weights=_shared_weights(),
    )

    first = controller.admit(
        GLM52RouteRequest("first", prompt_tokens=8192, max_new_tokens=512),
        contracts,
    )
    second = controller.admit(
        GLM52RouteRequest("second", prompt_tokens=8192, max_new_tokens=512),
        contracts,
    )

    assert first.route == GLM52IntraNodeRoute.SPLIT_SLP_DECODE
    assert second.route == GLM52IntraNodeRoute.DEFER
    assert "one-GPU SLP request capacity exhausted" in second.reason
    assert controller.snapshot().slp_prompt_tokens_in_use == 8192
    assert controller.snapshot().split_decode_reserved_tokens == 8704

    assert controller.mark_prefill_complete("first")
    snapshot = controller.snapshot()
    assert snapshot.slp_active_requests == 0
    assert snapshot.split_decode_reserved_requests == 1
    assert snapshot.split_decode_reserved_tokens == 8704
    assert not controller.mark_prefill_complete("first")

    assert controller.release("first")
    assert controller.snapshot().split_decode_reserved_requests == 0
    assert not controller.release("first")


def test_invalid_policy_and_capacity_contracts_raise_early() -> None:
    with pytest.raises(GLM52IntraNodeContractError, match="two distinct"):
        _config(gpu_ids=(0, 0))

    with pytest.raises(GLM52IntraNodeContractError, match="one entry per GPU"):
        _capacity((0, 1), available_device_bytes=(2 * _GIB,))
