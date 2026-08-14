"""CUDA-graph-safe, opt-in timing markers for DeepSeek V4 decode.

The ordinary :class:`torch.cuda.Event` default is an internal dependency when
recorded during CUDA graph capture.  Timing markers need to remain observable
after replay, so this module owns persistent ``external=True`` event pairs.
The pairs are created once, reused by every graph variant, and harvested only
after the complete DSpark cycle has returned to Python.  No allocation,
elapsed-time query, or synchronization occurs inside capture/replay.

This is diagnostic instrumentation.  Event nodes have a real (small) replay
cost, therefore production launchers leave it disabled unless both the
``internal_gpu_time`` DSpark dump component and
``SGLANG_DSV4_INTERNAL_TIMING=1`` are selected.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import Final

import torch

_ENABLE_ENV: Final = "SGLANG_DSV4_INTERNAL_TIMING"
_MAX_REASONABLE_RANGE_MS: Final = 10_000.0
_NULL_RANGE = nullcontext()


def dsv4_internal_timing_enabled() -> bool:
    """Return the process-static opt-in state without touching CUDA."""

    return os.environ.get(_ENABLE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


@dataclass(slots=True)
class _EventPair:
    start: torch.cuda.Event
    end: torch.cuda.Event


class _ActiveRange(AbstractContextManager[None]):
    __slots__ = ("_events", "_stream")

    def __init__(
        self,
        events: _EventPair,
        stream: torch.cuda.Stream | None,
    ) -> None:
        self._events = events
        self._stream = stream

    def __enter__(self) -> None:
        self._events.start.record(self._stream)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._events.end.record(self._stream)


class Dsv4InternalTimingRegistry:
    """Process-local arena of graph-persistent external CUDA event pairs."""

    def __init__(self) -> None:
        self._enabled = dsv4_internal_timing_enabled()
        self._events: dict[str, _EventPair] = {}
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def range(
        self,
        category: str,
        *,
        layer_id: int | None = None,
        role: str = "target",
        phase: str | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> AbstractContextManager[None]:
        """Return a range whose record operations are legal in graph capture.

        A semantic key must execute at most once in one model graph.  Callers
        that have multiple disjoint ranges in a layer provide distinct
        ``phase`` values; this prevents a later record from overwriting an
        earlier start within the same replay.
        """

        if not self._enabled:
            return _NULL_RANGE
        if not category or "." in category:
            raise ValueError("DSV4 timing category must be a non-empty atom")
        if not role or "." in role:
            raise ValueError("DSV4 timing role must be a non-empty atom")
        if phase is not None and (not phase or "." in phase):
            raise ValueError("DSV4 timing phase must be a non-empty atom")
        if layer_id is not None and layer_id < 0:
            raise ValueError("DSV4 timing layer_id must be non-negative")

        key_parts = [role, category]
        if layer_id is not None:
            key_parts.append(f"layer_{layer_id:02d}")
        if phase is not None:
            key_parts.append(phase)
        key = ".".join(key_parts)
        events = self._events.get(key)
        if events is None:
            # Graph construction is single-threaded in the serving worker, but
            # setup probes can run concurrently.  Lock only the cold path.
            with self._lock:
                events = self._events.get(key)
                if events is None:
                    events = _EventPair(
                        start=torch.cuda.Event(enable_timing=True, external=True),
                        end=torch.cuda.Event(enable_timing=True, external=True),
                    )
                    self._events[key] = events
        return _ActiveRange(events, stream)

    def snapshot(self) -> dict[str, float]:
        """Synchronize and read the latest complete replay for every range.

        This function is intentionally forbidden during capture.  The caller
        invokes it once after a DSpark cycle and before the next graph replay,
        which also prevents persistent events from being overwritten before
        their timestamps are consumed.
        """

        if not self._enabled or not self._events:
            return {}
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSV4 internal timing cannot be harvested in capture")

        result: dict[str, float] = {}
        for key, events in self._events.items():
            events.end.synchronize()
            elapsed_ms = float(events.start.elapsed_time(events.end))
            if 0.0 <= elapsed_ms <= _MAX_REASONABLE_RANGE_MS:
                result[key] = round(elapsed_ms, 4)
        return result

    def clear_for_test(self) -> None:
        """Drop cold registry state; tests only, never a serving hot-path API."""

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot clear DSV4 timing events during capture")
        self._events.clear()


_REGISTRY = Dsv4InternalTimingRegistry()


def dsv4_timing_range(
    category: str,
    *,
    layer_id: int | None = None,
    role: str = "target",
    phase: str | None = None,
    stream: torch.cuda.Stream | None = None,
) -> AbstractContextManager[None]:
    return _REGISTRY.range(
        category,
        layer_id=layer_id,
        role=role,
        phase=phase,
        stream=stream,
    )


def snapshot_dsv4_internal_timing() -> dict[str, float]:
    return _REGISTRY.snapshot()


__all__ = [
    "Dsv4InternalTimingRegistry",
    "dsv4_internal_timing_enabled",
    "dsv4_timing_range",
    "snapshot_dsv4_internal_timing",
]
