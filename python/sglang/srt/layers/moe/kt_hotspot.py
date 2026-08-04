"""Request-boundary expert-cache updates for KTransformers hybrid MoE.

The CUDA graph contract is deliberately narrow: registered layers keep the
same number of GPU expert slots and the same tensor objects for their entire
lifetime.  A plan update may only change the bytes and mapping tables inside
those existing slots, and only while the scheduler is fully idle.

This module owns plan validation and the per-process transaction.  The actual
MXFP4 byte copy lives on :class:`KTEPWrapperMethod`, beside the layout-specific
state it updates.  Keeping the controller layout-agnostic also makes the plan
and rollback logic unit-testable without CUDA or KTransformers installed.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod


_REGISTRY_LOCK = threading.Lock()
_REGISTERED_METHODS: dict[int, weakref.ReferenceType[KTEPWrapperMethod]] = {}
_LAST_COMMITTED_GENERATION = -1


@dataclass(frozen=True)
class HotspotLayerUpdate:
    """One validated rank-local layer update."""

    layer_idx: int
    selected_experts: tuple[int, ...]


@dataclass(frozen=True)
class LoadedHotspotPlan:
    """Validated rank-local view of a multi-rank hotspot plan."""

    source_path: str
    ep_rank: int
    ep_size: int
    num_layers: int
    num_experts: int
    updates: tuple[HotspotLayerUpdate, ...]


def register_hotspot_method(method: KTEPWrapperMethod) -> None:
    """Register a graph-safe target-layer cache without extending its lifetime."""

    method_id = id(method)

    def remove_dead_method(_reference: object) -> None:
        with _REGISTRY_LOCK:
            _REGISTERED_METHODS.pop(method_id, None)

    with _REGISTRY_LOCK:
        _REGISTERED_METHODS[method_id] = weakref.ref(method, remove_dead_method)


def _live_methods() -> tuple[KTEPWrapperMethod, ...]:
    with _REGISTRY_LOCK:
        methods = []
        dead_ids = []
        for method_id, reference in _REGISTERED_METHODS.items():
            method = reference()
            if method is None:
                dead_ids.append(method_id)
            else:
                methods.append(method)
        for method_id in dead_ids:
            _REGISTERED_METHODS.pop(method_id, None)
    return tuple(
        sorted(methods, key=lambda method: int(method.kt_config.layer_idx))
    )


def assign_experts_to_slots(
    current_slot_experts: Iterable[int], requested_experts: Iterable[int]
) -> tuple[int, ...]:
    """Retain cache hits in their slots and deterministically fill misses.

    CUDA graph safety only requires stable tensor addresses, but retaining
    matching logical experts in the same slots also avoids needless host to
    device promotions at every request boundary.
    """

    current = tuple(int(expert_id) for expert_id in current_slot_experts)
    requested = tuple(int(expert_id) for expert_id in requested_experts)
    if len(requested) != len(current):
        raise ValueError(
            "Hotspot placement must preserve the fixed GPU slot count: "
            f"current={len(current)} requested={len(requested)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Hotspot placement contains duplicate expert IDs")

    requested_set = set(requested)
    result: list[int | None] = [
        expert_id if expert_id in requested_set else None for expert_id in current
    ]
    retained = {expert_id for expert_id in result if expert_id is not None}
    misses = sorted(requested_set - retained)
    free_slots = [slot for slot, expert_id in enumerate(result) if expert_id is None]
    if len(misses) != len(free_slots):
        raise RuntimeError("Hotspot slot assignment did not form a bijection")
    for slot, expert_id in zip(free_slots, misses, strict=True):
        result[slot] = expert_id
    if any(expert_id is None for expert_id in result):
        raise RuntimeError("Hotspot slot assignment left an empty GPU slot")
    return tuple(int(expert_id) for expert_id in result)


def load_hotspot_plan(
    plan_path: str,
    *,
    ep_rank: int,
    ep_size: int,
    expected_num_layers: int,
    expected_num_experts: int,
) -> LoadedHotspotPlan:
    """Load and validate a fixed-width, disjoint multi-rank GPU placement."""

    real_path = os.path.realpath(plan_path)
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"Hotspot plan does not exist: {real_path}")
    loaded = torch.load(real_path, map_location="cpu", weights_only=True)
    raw_masks = (
        loaded.get("gpu_experts_mask_by_rank")
        if isinstance(loaded, dict)
        else None
    )
    if raw_masks is None:
        raise ValueError(
            "Hotspot plan must contain 'gpu_experts_mask_by_rank'"
        )
    masks = torch.as_tensor(raw_masks, dtype=torch.bool, device="cpu").contiguous()
    expected_shape = (ep_size, expected_num_layers, expected_num_experts)
    if tuple(masks.shape) != expected_shape:
        raise ValueError(
            f"Hotspot GPU masks must have shape {expected_shape}, "
            f"got {tuple(masks.shape)}"
        )
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"Hotspot EP rank {ep_rank} is outside [0, {ep_size})")

    widths = masks.sum(dim=-1)
    if widths.numel() == 0 or int(widths.min()) != int(widths.max()):
        raise ValueError("Hotspot plan must use one fixed GPU slot width")
    if int(widths.min()) <= 0:
        raise ValueError("Hotspot plan must retain at least one GPU expert per layer")

    # Expert ownership is rank-local in the production hybrid path.  A plan
    # assigning the same logical expert to two ranks would execute it twice,
    # so reject overlaps before any layer bytes are touched.
    if (masks.to(torch.int16).sum(dim=0) > 1).any().item():
        raise ValueError("Hotspot plan assigns a GPU expert to multiple EP ranks")

    updates = tuple(
        HotspotLayerUpdate(
            layer_idx=layer_idx,
            selected_experts=tuple(
                int(expert_id)
                for expert_id in torch.where(masks[ep_rank, layer_idx])[0].tolist()
            ),
        )
        for layer_idx in range(expected_num_layers)
    )
    return LoadedHotspotPlan(
        source_path=real_path,
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_layers=expected_num_layers,
        num_experts=expected_num_experts,
        updates=updates,
    )


def _parallel_identity() -> tuple[int, int]:
    # RuntimeContext is the canonical source for the composed TP/EP identity.
    # ``distributed.parallel_state`` exposes the process groups themselves but
    # does not export ``get_parallel`` in this SGLang revision.  Keep this
    # import lazy so CPU-only plan-validation tests remain lightweight.
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return int(parallel.moe_ep_rank), int(parallel.moe_ep_size)


def _synchronize_hotspot_ranks() -> None:
    if not dist.is_initialized():
        return
    from sglang.srt.distributed import get_tp_group

    dist.barrier(group=get_tp_group().device_group)


def _collect_rank_errors(local_error: str) -> list[str]:
    if not dist.is_initialized():
        return [local_error]
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group().cpu_group
    gathered: list[str | None] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local_error, group=group)
    return [error or "" for error in gathered]


def _collect_rank_values(local_value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [local_value]
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group().cpu_group
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local_value, group=group)
    return gathered


def _rank_failures(stage: str, errors: Iterable[str]) -> list[str]:
    return [
        f"{stage} rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error
    ]


def apply_hotspot_plan(
    *, plan_path: str, generation: int, dry_run: bool
) -> dict[str, Any]:
    """Validate or atomically apply a rank-local request-boundary placement.

    Every rank first validates every layer.  Mutations are then attempted
    locally, status is exchanged over the TP CPU group, and all completed
    layers are rolled back if any rank reports a failure.
    """

    global _LAST_COMMITTED_GENERATION

    methods: tuple[KTEPWrapperMethod, ...] = ()
    prepared: list[tuple[KTEPWrapperMethod, tuple[int, ...], tuple[int, ...]]] = []
    base_receipt: dict[str, Any] = {}
    validation_error = ""
    try:
        methods = _live_methods()
        if not methods:
            raise RuntimeError(
                "No hotspot-capable target layers are registered; launch with "
                "SGLANG_KT_HOTSPOT_EXPERT_CACHE=1"
            )
        layer_indices = tuple(int(method.kt_config.layer_idx) for method in methods)
        if layer_indices != tuple(range(len(methods))):
            raise RuntimeError(
                "Hotspot target-layer registry is incomplete: "
                f"registered={layer_indices}"
            )
        num_experts_values = {int(method.global_num_experts) for method in methods}
        if len(num_experts_values) != 1:
            raise RuntimeError(
                f"Hotspot layers disagree on global expert count: {num_experts_values}"
            )
        if generation < 0:
            raise ValueError("Hotspot generation must be non-negative")
        if not dry_run and generation <= _LAST_COMMITTED_GENERATION:
            raise ValueError(
                "Hotspot generation must increase monotonically: "
                f"last={_LAST_COMMITTED_GENERATION} requested={generation}"
            )

        ep_rank, ep_size = _parallel_identity()
        plan = load_hotspot_plan(
            plan_path,
            ep_rank=ep_rank,
            ep_size=ep_size,
            expected_num_layers=len(methods),
            expected_num_experts=next(iter(num_experts_values)),
        )
        for method, update in zip(methods, plan.updates, strict=True):
            old_slots = tuple(
                int(value) for value in method.gpu_index_to_logical.tolist()
            )
            new_slots = assign_experts_to_slots(old_slots, update.selected_experts)
            method.validate_hotspot_slots(new_slots)
            prepared.append((method, old_slots, new_slots))

        planned_receipts = [
            {
                "layer": int(method.kt_config.layer_idx),
                "old_slots": list(old_slots),
                "new_slots": list(new_slots),
                "swaps": sum(
                    old != new
                    for old, new in zip(old_slots, new_slots, strict=True)
                ),
            }
            for method, old_slots, new_slots in prepared
        ]
        base_receipt = {
            "generation": generation,
            "dry_run": dry_run,
            "plan_path": plan.source_path,
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "layers": planned_receipts,
            "total_swaps": sum(item["swaps"] for item in planned_receipts),
        }
    except Exception as error:  # noqa: BLE001 - publish validation on every rank
        validation_error = f"{type(error).__name__}: {error}"

    validation_failures = _rank_failures(
        "validation", _collect_rank_errors(validation_error)
    )
    if validation_failures:
        raise RuntimeError("Hotspot transaction failed: " + "; ".join(validation_failures))

    if dry_run:
        base_receipt["rank_receipts"] = _collect_rank_values(dict(base_receipt))
        return base_receipt

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _synchronize_hotspot_ranks()
    started = time.perf_counter()
    attempted: list[tuple[KTEPWrapperMethod, tuple[int, ...]]] = []
    local_error = ""
    copied_bytes = 0
    try:
        for method, old_slots, new_slots in prepared:
            if old_slots == new_slots:
                continue
            # Record the layer before mutation.  commit_hotspot_slots may fail
            # after overwriting only a subset of its fixed GPU slots.
            attempted.append((method, old_slots))
            copied_bytes += int(method.commit_hotspot_slots(new_slots))
    except Exception as error:  # noqa: BLE001 - all ranks must reach rollback
        local_error = f"{type(error).__name__}: {error}"

    rank_errors = _collect_rank_errors(local_error)
    failures = _rank_failures("commit", rank_errors)
    if failures:
        rollback_errors = []
        for method, old_slots in reversed(attempted):
            try:
                method.commit_hotspot_slots(old_slots, force_reload=True)
            except Exception as error:  # noqa: BLE001 - record rollback evidence
                rollback_errors.append(
                    f"layer {method.kt_config.layer_idx}: "
                    f"{type(error).__name__}: {error}"
                )
        all_rollback_errors = _collect_rank_values(rollback_errors)
        _synchronize_hotspot_ranks()
        flattened_rollback_errors = [
            f"rank {rank}: {error}"
            for rank, errors in enumerate(all_rollback_errors)
            for error in errors
        ]
        suffix = (
            f"; rollback failures: {flattened_rollback_errors}"
            if flattened_rollback_errors
            else ""
        )
        raise RuntimeError("Hotspot transaction failed: " + "; ".join(failures) + suffix)

    _synchronize_hotspot_ranks()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _LAST_COMMITTED_GENERATION = generation
    base_receipt.update(
        {
            "local_copied_bytes": copied_bytes,
            "local_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "last_committed_generation": _LAST_COMMITTED_GENERATION,
        }
    )
    rank_receipts = _collect_rank_values(dict(base_receipt))
    base_receipt["rank_receipts"] = rank_receipts
    base_receipt["copied_bytes"] = sum(
        int(receipt["local_copied_bytes"]) for receipt in rank_receipts
    )
    base_receipt["elapsed_ms"] = max(
        float(receipt["local_elapsed_ms"]) for receipt in rank_receipts
    )
    return base_receipt


def reset_hotspot_state_for_tests() -> None:
    """Clear process-global registry state for isolated unit tests."""

    global _LAST_COMMITTED_GENERATION
    with _REGISTRY_LOCK:
        _REGISTERED_METHODS.clear()
    _LAST_COMMITTED_GENERATION = -1
