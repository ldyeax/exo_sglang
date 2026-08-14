from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from sglang.srt.observability.dsv4_internal_timing import (
    Dsv4InternalTimingRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=2, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class _FakeEvent:
    constructed: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.recorded_streams: list[object] = []
        self.synchronize_count = 0
        self.constructed.append(kwargs)

    def record(self, stream=None) -> None:
        self.recorded_streams.append(stream)

    def synchronize(self) -> None:
        self.synchronize_count += 1

    def elapsed_time(self, end: _FakeEvent) -> float:
        assert end is not self
        return 1.23456


def _enabled_registry() -> Dsv4InternalTimingRegistry:
    with patch.dict(os.environ, {"SGLANG_DSV4_INTERNAL_TIMING": "1"}):
        return Dsv4InternalTimingRegistry()


def test_disabled_registry_never_constructs_cuda_events() -> None:
    with patch.dict(os.environ, {"SGLANG_DSV4_INTERNAL_TIMING": "0"}):
        registry = Dsv4InternalTimingRegistry()
    with patch("torch.cuda.Event") as event_constructor:
        with registry.range("routed_moe", layer_id=3):
            pass
    event_constructor.assert_not_called()
    assert registry.snapshot() == {}


def test_event_pairs_are_external_persistent_and_keyed_by_semantics() -> None:
    _FakeEvent.constructed.clear()
    registry = _enabled_registry()
    stream = object()
    with (
        patch("torch.cuda.Event", _FakeEvent),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        for _ in range(2):
            with registry.range(
                "routed_moe",
                layer_id=7,
                role="target",
                phase="route_experts_merge",
                stream=stream,
            ):
                pass
        snapshot = registry.snapshot()

    assert _FakeEvent.constructed == [
        {"enable_timing": True, "external": True},
        {"enable_timing": True, "external": True},
    ]
    assert snapshot == {"target.routed_moe.layer_07.route_experts_merge": 1.2346}
    pair = registry._events["target.routed_moe.layer_07.route_experts_merge"]
    assert pair.start.recorded_streams == [stream, stream]
    assert pair.end.recorded_streams == [stream, stream]
    assert pair.end.synchronize_count == 1


def test_snapshot_fails_closed_during_graph_capture() -> None:
    registry = _enabled_registry()
    with patch("torch.cuda.Event", _FakeEvent):
        with registry.range("final_head", phase="logits"):
            pass
    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="cannot be harvested"):
            registry.snapshot()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_external_timing_events_survive_cuda_graph_replay() -> None:
    registry = _enabled_registry()
    source = torch.randn(1 << 18, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)

    # Materialize the persistent events and the pointwise specialization before
    # capture, mirroring model warmup. No event object is allocated in capture.
    with registry.range("attention_indexer", layer_id=2, phase="c4_indexer"):
        torch.mul(source, source, out=output)
    torch.cuda.synchronize()
    registry.snapshot()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with registry.range("attention_indexer", layer_id=2, phase="c4_indexer"):
            torch.mul(source, source, out=output)
    source.fill_(3.0)
    graph.replay()
    measured = registry.snapshot()

    torch.testing.assert_close(output, torch.full_like(output, 9.0))
    elapsed = measured["target.attention_indexer.layer_02.c4_indexer"]
    assert 0.0 < elapsed < 100.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"category": "bad.atom"}, "category"),
        ({"category": "ok", "role": "bad.atom"}, "role"),
        ({"category": "ok", "phase": "bad.atom"}, "phase"),
        ({"category": "ok", "layer_id": -1}, "layer_id"),
    ],
)
def test_semantic_keys_reject_ambiguous_atoms(kwargs, message: str) -> None:
    registry = _enabled_registry()
    with pytest.raises(ValueError, match=message):
        registry.range(**kwargs)
