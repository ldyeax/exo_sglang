from __future__ import annotations

import pytest
from sglang.srt.managers.scheduler import (
    _gather_dsv4_sm86_small_batch_gemm_workers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeTensorParallelGroup:
    world_size = 2

    def __init__(self, gathered: list[dict]) -> None:
        self.gathered = gathered
        self.received: dict | None = None

    def all_gather_object(self, local: dict) -> list[dict]:
        self.received = local
        return self.gathered


def _telemetry(tp_rank: int) -> dict:
    return {
        "tp_rank": tp_rank,
        "pp_rank": 0,
        "dp_rank": 0,
        "patch_state": "installed",
        "patch_installed": True,
        "selection_count": 1,
    }


def test_tp_gather_forwards_both_distinct_rank_records() -> None:
    local = _telemetry(0)
    gathered = [_telemetry(0), _telemetry(1)]
    group = _FakeTensorParallelGroup(gathered)

    assert _gather_dsv4_sm86_small_batch_gemm_workers(group, local) == gathered
    assert group.received is local


@pytest.mark.parametrize(
    "gathered,match",
    [
        ([_telemetry(0)], "expected 2"),
        ([_telemetry(0), _telemetry(0)], "rank coverage mismatch"),
    ],
)
def test_tp_gather_fails_closed_on_incomplete_rank_proof(
    gathered: list[dict], match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _gather_dsv4_sm86_small_batch_gemm_workers(
            _FakeTensorParallelGroup(gathered), _telemetry(0)
        )
