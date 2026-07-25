# SPDX-License-Identifier: Apache-2.0
"""Fail-closed GLM-5.2 routing for the two-GPU ``dwagon`` host.

This module is deliberately a pure admission and action-planning boundary.
``glm52_intra_node_runtime`` supplies the executable PP=1/TP=1 prefill,
PP=1/TP=1 decode, native KV-handoff, and model-gateway process topology.  The
bounded KTransformers stream-loading prefill (SLP) executor also retains its
unified PP=1/TP=2 path.

Callers therefore have to present explicit executor, shared-host-weight, and
capacity contracts.  A missing or stale proof never falls back to an
unverified execution mode:

* prompts below ``long_prompt_threshold`` use ordinary chunked prefill;
* an isolated long prompt may use the admitted two-GPU SLP executor;
* a long prompt while decode is active may use a one-GPU/one-GPU split only
  after every split contract is satisfied;
* all other combinations are rejected or deferred.

The controller at the bottom provides atomic reservations and state-reporting
hooks for callers that need selective admission.  The executable runtime
withholds its public gateway until its live contracts satisfy this planner.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

GLM52_ARCHITECTURES = frozenset(
    {
        "GlmMoeDsaForCausalLM",
        "GlmMoeDsaForConditionalGeneration",
    }
)
DWAGON_HARDWARE_PROFILE = "glm52-dwagon-2x24g"


class GLM52IntraNodeContractError(ValueError):
    """Raised for structurally invalid policy inputs or state transitions."""


class GLM52IntraNodeRoute(str, Enum):
    """Result of routing and admission."""

    CHUNKED_PREFILL = "chunked_prefill"
    DISTRIBUTED_SLP = "distributed_slp"
    SPLIT_SLP_DECODE = "split_slp_decode"
    DEFER = "defer"
    REJECT = "reject"


class GLM52IntraNodeActionKind(str, Enum):
    """Executor-facing primitives emitted by an admitted plan."""

    RUN_CHUNKED_PREFILL = "run_chunked_prefill"
    RUN_DISTRIBUTED_SLP = "run_distributed_slp"
    RUN_SINGLE_GPU_SLP = "run_single_gpu_slp"
    TRANSFER_KV_TO_DECODE = "transfer_kv_to_decode"
    CONTINUE_SINGLE_GPU_DECODE = "continue_single_gpu_decode"


@dataclass(frozen=True)
class GLM52IntraNodePolicyConfig:
    """Static policy settings derived from server arguments."""

    enabled: bool
    long_prompt_threshold: int
    chunked_prefill_size: int
    gpu_ids: tuple[int, int]
    hardware_profile: str = DWAGON_HARDWARE_PROFILE

    def __post_init__(self) -> None:
        if self.long_prompt_threshold <= 0:
            raise GLM52IntraNodeContractError("long_prompt_threshold must be positive")
        if self.chunked_prefill_size <= 0:
            raise GLM52IntraNodeContractError("chunked_prefill_size must be positive")
        if len(set(self.gpu_ids)) != 2 or any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise GLM52IntraNodeContractError(
                f"dwagon policy requires two distinct non-negative GPU ids, got {self.gpu_ids}"
            )
        if not self.hardware_profile:
            raise GLM52IntraNodeContractError("hardware_profile must not be empty")


@dataclass(frozen=True)
class GLM52RouteRequest:
    """Request facts used for routing and capacity reservation."""

    request_id: str
    prompt_tokens: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise GLM52IntraNodeContractError("request_id must not be empty")
        if self.prompt_tokens <= 0:
            raise GLM52IntraNodeContractError("prompt_tokens must be positive")
        if self.max_new_tokens < 0:
            raise GLM52IntraNodeContractError("max_new_tokens must not be negative")

    @property
    def decode_reservation_tokens(self) -> int:
        """KV capacity needed when this request enters the split decode worker."""

        return self.prompt_tokens + self.max_new_tokens


@dataclass(frozen=True)
class GLM52ExecutorCompatibility:
    """Runtime proof of model topology and available executor wiring.

    ``distributed_slp_admission`` must identify a successful admission of the
    existing bounded KT stream-prefill executor.  The split booleans stay
    separate because a TP1 SLP worker, TP1 decode worker, and KV handoff are
    independently attested pieces of the live runtime.
    """

    architecture: str
    hardware_profile: str
    gpu_ids: tuple[int, ...]
    pipeline_parallel_size: int
    chunked_prefill_ready: bool
    chunked_prefill_tensor_parallel_size: int
    distributed_slp_ready: bool
    distributed_slp_tensor_parallel_size: int
    distributed_slp_admission: Optional[str]
    split_slp_ready: bool
    split_slp_tensor_parallel_size: int
    split_decode_ready: bool
    split_decode_tensor_parallel_size: int
    split_kv_transfer_ready: bool

    def __post_init__(self) -> None:
        if self.pipeline_parallel_size <= 0:
            raise GLM52IntraNodeContractError("pipeline_parallel_size must be positive")
        tensor_parallel_sizes = (
            self.chunked_prefill_tensor_parallel_size,
            self.distributed_slp_tensor_parallel_size,
            self.split_slp_tensor_parallel_size,
            self.split_decode_tensor_parallel_size,
        )
        if any(size <= 0 for size in tensor_parallel_sizes):
            raise GLM52IntraNodeContractError(
                "all executor tensor-parallel sizes must be positive"
            )
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise GLM52IntraNodeContractError(
                f"executor GPU ids must be distinct, got {self.gpu_ids}"
            )


@dataclass(frozen=True)
class GLM52CapacityContract:
    """A live, route-specific device and token capacity proof."""

    gpu_ids: tuple[int, ...]
    max_concurrent_requests: int
    max_inflight_tokens: int
    available_device_bytes: tuple[int, ...]
    required_device_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise GLM52IntraNodeContractError(
                "capacity contract must name at least one GPU"
            )
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise GLM52IntraNodeContractError(
                f"capacity GPU ids must be distinct, got {self.gpu_ids}"
            )
        if len(self.available_device_bytes) != len(self.gpu_ids):
            raise GLM52IntraNodeContractError(
                "available_device_bytes must have one entry per GPU"
            )
        if len(self.required_device_bytes) != len(self.gpu_ids):
            raise GLM52IntraNodeContractError(
                "required_device_bytes must have one entry per GPU"
            )
        if self.max_concurrent_requests <= 0:
            raise GLM52IntraNodeContractError(
                "max_concurrent_requests must be positive"
            )
        if self.max_inflight_tokens <= 0:
            raise GLM52IntraNodeContractError("max_inflight_tokens must be positive")
        if any(value < 0 for value in self.available_device_bytes):
            raise GLM52IntraNodeContractError(
                "available device byte counts must not be negative"
            )
        if any(value <= 0 for value in self.required_device_bytes):
            raise GLM52IntraNodeContractError(
                "required device byte counts must be positive"
            )

    def admission_failure(
        self,
        *,
        active_requests: int,
        active_tokens: int,
        requested_tokens: int,
    ) -> Optional[str]:
        """Return a precise failure instead of overcommitting this contract."""

        for gpu_id, available_bytes, required_bytes in zip(
            self.gpu_ids,
            self.available_device_bytes,
            self.required_device_bytes,
        ):
            if available_bytes < required_bytes:
                return (
                    f"GPU {gpu_id} has {available_bytes} available bytes but "
                    f"{required_bytes} are required"
                )
        if active_requests + 1 > self.max_concurrent_requests:
            return (
                "request capacity exhausted: "
                f"{active_requests + 1}>{self.max_concurrent_requests}"
            )
        if active_tokens + requested_tokens > self.max_inflight_tokens:
            return (
                "token capacity exhausted: "
                f"{active_tokens}+{requested_tokens}>{self.max_inflight_tokens}"
            )
        return None


@dataclass(frozen=True)
class GLM52SharedHostWeights:
    """Attestation that TP1 prefill and decode reuse one immutable allocation."""

    weight_identity: str
    allocation_identity: str
    weight_bytes: int
    host_capacity_bytes: int
    safety_margin_bytes: int
    immutable: bool
    sharing_verified: bool
    prefill_reader_ready: bool
    decode_reader_ready: bool

    def failure(self) -> Optional[str]:
        if not self.weight_identity:
            return "shared host weights have no weight identity"
        if not self.allocation_identity:
            return "shared host weights have no allocation identity"
        if self.weight_bytes <= 0:
            return "shared host weight size must be positive"
        if self.host_capacity_bytes <= 0:
            return "shared host capacity must be positive"
        if self.safety_margin_bytes < 0:
            return "shared host safety margin must not be negative"
        if self.host_capacity_bytes < self.weight_bytes + self.safety_margin_bytes:
            return (
                "shared host capacity is insufficient: "
                f"{self.host_capacity_bytes}<{self.weight_bytes}+"
                f"{self.safety_margin_bytes}"
            )
        if not self.immutable:
            return "shared host weights are not immutable"
        if not self.sharing_verified:
            return "one shared host allocation was not verified"
        if not self.prefill_reader_ready:
            return "prefill reader is not attached to shared host weights"
        if not self.decode_reader_ready:
            return "decode reader is not attached to shared host weights"
        return None


@dataclass(frozen=True)
class GLM52IntraNodeContracts:
    """All proofs that may be consumed by one policy decision."""

    executors: GLM52ExecutorCompatibility
    chunked_prefill_capacity: Optional[GLM52CapacityContract] = None
    distributed_slp_capacity: Optional[GLM52CapacityContract] = None
    split_slp_capacity: Optional[GLM52CapacityContract] = None
    split_decode_capacity: Optional[GLM52CapacityContract] = None
    shared_host_weights: Optional[GLM52SharedHostWeights] = None


@dataclass(frozen=True)
class GLM52IntraNodeRuntimeState:
    """Live state plus reservations already owned by the controller.

    External counters must exclude requests represented by controller
    reservations, otherwise split decode capacity would be counted twice.
    """

    decode_active_requests: int = 0
    decode_tokens_in_use: int = 0
    other_active_requests: int = 0
    other_tokens_in_use: int = 0
    slp_active_requests: int = 0
    slp_prompt_tokens_in_use: int = 0
    split_decode_reserved_requests: int = 0
    split_decode_reserved_tokens: int = 0
    distributed_slp_active: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.decode_active_requests,
            self.decode_tokens_in_use,
            self.other_active_requests,
            self.other_tokens_in_use,
            self.slp_active_requests,
            self.slp_prompt_tokens_in_use,
            self.split_decode_reserved_requests,
            self.split_decode_reserved_tokens,
        )
        if any(value < 0 for value in counts):
            raise GLM52IntraNodeContractError(
                "runtime request and token counts must not be negative"
            )


@dataclass(frozen=True)
class GLM52IntraNodeAction:
    """One ordered action for an executor."""

    kind: GLM52IntraNodeActionKind
    gpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class GLM52IntraNodePlan:
    """Complete routing result and the capacity it reserves."""

    request_id: str
    route: GLM52IntraNodeRoute
    actions: tuple[GLM52IntraNodeAction, ...]
    reason: str
    slp_reserved_tokens: int = 0
    decode_reserved_tokens: int = 0

    @property
    def admitted(self) -> bool:
        return self.route not in {
            GLM52IntraNodeRoute.DEFER,
            GLM52IntraNodeRoute.REJECT,
        }


def _terminal_plan(
    request: GLM52RouteRequest,
    route: GLM52IntraNodeRoute,
    reason: str,
) -> GLM52IntraNodePlan:
    return GLM52IntraNodePlan(
        request_id=request.request_id,
        route=route,
        actions=(),
        reason=reason,
    )


def _target_failure(
    config: GLM52IntraNodePolicyConfig,
    executors: GLM52ExecutorCompatibility,
) -> Optional[str]:
    if not config.enabled:
        return "GLM-5.2 intra-node policy is disabled"
    if executors.architecture not in GLM52_ARCHITECTURES:
        return (
            f"architecture {executors.architecture!r} is not an admitted GLM-5.2 "
            "architecture"
        )
    if executors.hardware_profile != config.hardware_profile:
        return (
            f"hardware profile {executors.hardware_profile!r} does not match "
            f"{config.hardware_profile!r}"
        )
    if executors.gpu_ids != config.gpu_ids:
        return (
            f"executor GPU ids {executors.gpu_ids} do not match configured "
            f"{config.gpu_ids}"
        )
    if executors.pipeline_parallel_size != 1:
        return (
            "GLM-5.2 intra-node routing requires pipeline parallel size 1, got "
            f"{executors.pipeline_parallel_size}"
        )
    return None


def _capacity_shape_failure(
    capacity: GLM52CapacityContract,
    expected_gpu_ids: tuple[int, ...],
) -> Optional[str]:
    if capacity.gpu_ids != expected_gpu_ids:
        return (
            f"capacity GPU ids {capacity.gpu_ids} do not match expected "
            f"{expected_gpu_ids}"
        )
    return None


def plan_glm52_intra_node_route(
    config: GLM52IntraNodePolicyConfig,
    request: GLM52RouteRequest,
    state: GLM52IntraNodeRuntimeState,
    contracts: GLM52IntraNodeContracts,
) -> GLM52IntraNodePlan:
    """Classify and admit one request without mutating runtime state."""

    executors = contracts.executors
    if failure := _target_failure(config, executors):
        return _terminal_plan(request, GLM52IntraNodeRoute.REJECT, failure)

    is_long_prompt = request.prompt_tokens >= config.long_prompt_threshold
    if not is_long_prompt:
        if not executors.chunked_prefill_ready:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "chunked-prefill executor is not ready",
            )
        if executors.chunked_prefill_tensor_parallel_size != 2:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "chunked prefill requires the admitted TP=2 unified worker",
            )
        capacity = contracts.chunked_prefill_capacity
        if capacity is None:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "chunked-prefill capacity contract is missing",
            )
        if failure := _capacity_shape_failure(capacity, config.gpu_ids):
            return _terminal_plan(request, GLM52IntraNodeRoute.REJECT, failure)
        if state.slp_active_requests > 0:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.DEFER,
                "an SLP reservation currently owns at least one policy GPU",
            )
        if failure := capacity.admission_failure(
            active_requests=state.other_active_requests,
            active_tokens=state.other_tokens_in_use,
            requested_tokens=request.prompt_tokens,
        ):
            return _terminal_plan(request, GLM52IntraNodeRoute.DEFER, failure)
        return GLM52IntraNodePlan(
            request_id=request.request_id,
            route=GLM52IntraNodeRoute.CHUNKED_PREFILL,
            actions=(
                GLM52IntraNodeAction(
                    GLM52IntraNodeActionKind.RUN_CHUNKED_PREFILL,
                    config.gpu_ids,
                ),
            ),
            reason=(
                f"prompt has {request.prompt_tokens} tokens, below the "
                f"{config.long_prompt_threshold}-token SLP threshold"
            ),
        )

    decode_is_active = (
        state.decode_active_requests + state.split_decode_reserved_requests > 0
    )
    if not decode_is_active:
        if (
            state.other_active_requests > 0
            or state.slp_active_requests > 0
            or state.distributed_slp_active
        ):
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.DEFER,
                "two-GPU SLP requires an isolated long prompt",
            )
        if not executors.distributed_slp_ready:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "two-GPU distributed SLP executor is not ready",
            )
        if executors.distributed_slp_tensor_parallel_size != 2:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "two-GPU distributed SLP requires TP=2",
            )
        if not executors.distributed_slp_admission:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "two-GPU distributed SLP has no KT stream-prefill admission receipt",
            )
        capacity = contracts.distributed_slp_capacity
        if capacity is None:
            return _terminal_plan(
                request,
                GLM52IntraNodeRoute.REJECT,
                "two-GPU distributed SLP capacity contract is missing",
            )
        if failure := _capacity_shape_failure(capacity, config.gpu_ids):
            return _terminal_plan(request, GLM52IntraNodeRoute.REJECT, failure)
        if failure := capacity.admission_failure(
            active_requests=state.slp_active_requests,
            active_tokens=state.slp_prompt_tokens_in_use,
            requested_tokens=request.prompt_tokens,
        ):
            return _terminal_plan(request, GLM52IntraNodeRoute.DEFER, failure)
        return GLM52IntraNodePlan(
            request_id=request.request_id,
            route=GLM52IntraNodeRoute.DISTRIBUTED_SLP,
            actions=(
                GLM52IntraNodeAction(
                    GLM52IntraNodeActionKind.RUN_DISTRIBUTED_SLP,
                    config.gpu_ids,
                ),
            ),
            reason="isolated long prompt admitted to the existing TP=2 SLP executor",
            slp_reserved_tokens=request.prompt_tokens,
        )

    split_failures: list[str] = []
    if not executors.split_slp_ready:
        split_failures.append("one-GPU SLP executor is not ready")
    if executors.split_slp_tensor_parallel_size != 1:
        split_failures.append("one-GPU SLP executor must use TP=1")
    if not executors.split_decode_ready:
        split_failures.append("one-GPU decode executor is not ready")
    if executors.split_decode_tensor_parallel_size != 1:
        split_failures.append("one-GPU decode executor must use TP=1")
    if not executors.split_kv_transfer_ready:
        split_failures.append("SLP-to-decode KV transfer is not ready")
    if state.other_active_requests > 0:
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.DEFER,
            "a unified-worker request still owns both policy GPUs",
        )
    if state.distributed_slp_active:
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.DEFER,
            "the two-GPU distributed SLP executor currently owns both GPUs",
        )
    if split_failures:
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.REJECT,
            "split SLP/decode admission failed: " + "; ".join(split_failures),
        )

    slp_capacity = contracts.split_slp_capacity
    decode_capacity = contracts.split_decode_capacity
    if slp_capacity is None or decode_capacity is None:
        missing = []
        if slp_capacity is None:
            missing.append("one-GPU SLP")
        if decode_capacity is None:
            missing.append("one-GPU decode")
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.REJECT,
            f"split capacity contract is missing for {', '.join(missing)}",
        )
    if len(slp_capacity.gpu_ids) != 1 or len(decode_capacity.gpu_ids) != 1:
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.REJECT,
            "split SLP and decode capacity contracts must each name one GPU",
        )
    split_gpu_ids = slp_capacity.gpu_ids + decode_capacity.gpu_ids
    if len(set(split_gpu_ids)) != 2 or set(split_gpu_ids) != set(config.gpu_ids):
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.REJECT,
            "split SLP/decode GPU assignments must be distinct and cover both policy GPUs",
        )

    shared_weights = contracts.shared_host_weights
    if shared_weights is None:
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.REJECT,
            "shared-host-weight contract is missing",
        )
    if failure := shared_weights.failure():
        return _terminal_plan(request, GLM52IntraNodeRoute.REJECT, failure)

    if failure := slp_capacity.admission_failure(
        active_requests=state.slp_active_requests,
        active_tokens=state.slp_prompt_tokens_in_use,
        requested_tokens=request.prompt_tokens,
    ):
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.DEFER,
            f"one-GPU SLP {failure}",
        )
    if failure := decode_capacity.admission_failure(
        active_requests=(
            state.decode_active_requests + state.split_decode_reserved_requests
        ),
        active_tokens=(state.decode_tokens_in_use + state.split_decode_reserved_tokens),
        requested_tokens=request.decode_reservation_tokens,
    ):
        return _terminal_plan(
            request,
            GLM52IntraNodeRoute.DEFER,
            f"one-GPU decode {failure}",
        )

    slp_gpu_ids = slp_capacity.gpu_ids
    decode_gpu_ids = decode_capacity.gpu_ids
    return GLM52IntraNodePlan(
        request_id=request.request_id,
        route=GLM52IntraNodeRoute.SPLIT_SLP_DECODE,
        actions=(
            GLM52IntraNodeAction(
                GLM52IntraNodeActionKind.RUN_SINGLE_GPU_SLP,
                slp_gpu_ids,
            ),
            GLM52IntraNodeAction(
                GLM52IntraNodeActionKind.TRANSFER_KV_TO_DECODE,
                split_gpu_ids,
            ),
            GLM52IntraNodeAction(
                GLM52IntraNodeActionKind.CONTINUE_SINGLE_GPU_DECODE,
                decode_gpu_ids,
            ),
        ),
        reason=(
            "decode is active and TP1 executors, KV transfer, shared host "
            "weights, and both capacity contracts are satisfied"
        ),
        slp_reserved_tokens=request.prompt_tokens,
        decode_reserved_tokens=request.decode_reservation_tokens,
    )


@dataclass(frozen=True)
class _GLM52Reservation:
    route: GLM52IntraNodeRoute
    slp_tokens: int
    decode_tokens: int
    prefill_active: bool = True


class GLM52IntraNodeAdmissionController:
    """Thread-safe admission and lifecycle hooks for a request router."""

    def __init__(self, config: GLM52IntraNodePolicyConfig):
        self.config = config
        self._lock = threading.Lock()
        self._decode_active_requests = 0
        self._decode_tokens_in_use = 0
        self._other_active_requests = 0
        self._other_tokens_in_use = 0
        self._reservations: dict[str, _GLM52Reservation] = {}

    def report_external_state(
        self,
        *,
        decode_active_requests: int,
        decode_tokens_in_use: int,
        other_active_requests: int = 0,
        other_tokens_in_use: int = 0,
    ) -> None:
        """Replace externally owned counters.

        These counters must exclude requests still present in this controller's
        reservation table.
        """

        values = (
            decode_active_requests,
            decode_tokens_in_use,
            other_active_requests,
            other_tokens_in_use,
        )
        if any(value < 0 for value in values):
            raise GLM52IntraNodeContractError(
                "external runtime counters must not be negative"
            )
        with self._lock:
            self._decode_active_requests = decode_active_requests
            self._decode_tokens_in_use = decode_tokens_in_use
            self._other_active_requests = other_active_requests
            self._other_tokens_in_use = other_tokens_in_use

    def _snapshot_locked(self) -> GLM52IntraNodeRuntimeState:
        active_prefills = [
            reservation
            for reservation in self._reservations.values()
            if reservation.prefill_active
        ]
        split_reservations = [
            reservation
            for reservation in self._reservations.values()
            if reservation.route == GLM52IntraNodeRoute.SPLIT_SLP_DECODE
        ]
        return GLM52IntraNodeRuntimeState(
            decode_active_requests=self._decode_active_requests,
            decode_tokens_in_use=self._decode_tokens_in_use,
            other_active_requests=self._other_active_requests,
            other_tokens_in_use=self._other_tokens_in_use,
            slp_active_requests=len(active_prefills),
            slp_prompt_tokens_in_use=sum(
                reservation.slp_tokens for reservation in active_prefills
            ),
            split_decode_reserved_requests=len(split_reservations),
            split_decode_reserved_tokens=sum(
                reservation.decode_tokens for reservation in split_reservations
            ),
            distributed_slp_active=any(
                reservation.route == GLM52IntraNodeRoute.DISTRIBUTED_SLP
                for reservation in active_prefills
            ),
        )

    def snapshot(self) -> GLM52IntraNodeRuntimeState:
        with self._lock:
            return self._snapshot_locked()

    def plan(
        self,
        request: GLM52RouteRequest,
        contracts: GLM52IntraNodeContracts,
    ) -> GLM52IntraNodePlan:
        with self._lock:
            state = self._snapshot_locked()
        return plan_glm52_intra_node_route(
            self.config,
            request,
            state,
            contracts,
        )

    def admit(
        self,
        request: GLM52RouteRequest,
        contracts: GLM52IntraNodeContracts,
    ) -> GLM52IntraNodePlan:
        """Plan and atomically reserve SLP/decode capacity when admitted."""

        with self._lock:
            if request.request_id in self._reservations:
                return _terminal_plan(
                    request,
                    GLM52IntraNodeRoute.REJECT,
                    f"request {request.request_id!r} already owns a reservation",
                )
            plan = plan_glm52_intra_node_route(
                self.config,
                request,
                self._snapshot_locked(),
                contracts,
            )
            if plan.route in {
                GLM52IntraNodeRoute.DISTRIBUTED_SLP,
                GLM52IntraNodeRoute.SPLIT_SLP_DECODE,
            }:
                self._reservations[request.request_id] = _GLM52Reservation(
                    route=plan.route,
                    slp_tokens=plan.slp_reserved_tokens,
                    decode_tokens=plan.decode_reserved_tokens,
                )
            return plan

    def mark_prefill_complete(self, request_id: str) -> bool:
        """Release SLP capacity while retaining a split decode reservation."""

        with self._lock:
            reservation = self._reservations.get(request_id)
            if reservation is None or not reservation.prefill_active:
                return False
            if reservation.route == GLM52IntraNodeRoute.DISTRIBUTED_SLP:
                del self._reservations[request_id]
            else:
                self._reservations[request_id] = replace(
                    reservation,
                    prefill_active=False,
                    slp_tokens=0,
                )
            return True

    def release(self, request_id: str) -> bool:
        """Release every remaining reservation for a completed/aborted request."""

        with self._lock:
            return self._reservations.pop(request_id, None) is not None
