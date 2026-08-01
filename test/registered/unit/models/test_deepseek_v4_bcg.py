from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.srt.models import deepseek_v4
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_moe_bcg_reestablishes_expert_recorder_layer(monkeypatch):
    events = []

    class FakeRecorder:
        @contextmanager
        def with_current_layer_if_absent(self, layer_id):
            events.append(("enter", layer_id))
            try:
                yield
            finally:
                events.append(("exit", layer_id))

    class FakeDecoderLayer:
        layer_id = 37

        def _run_moe_ffn_dp_sync(
            self,
            hidden_states,
            forward_batch,
            *,
            input_ids,
            input_ids_global,
        ):
            events.append(("run", self.layer_id))
            return hidden_states + 1

    monkeypatch.setattr(
        deepseek_v4,
        "get_global_expert_distribution_recorder",
        lambda: FakeRecorder(),
    )
    hidden_states = torch.tensor([[2.0]])

    output = deepseek_v4.deepseek_v4_moe_ffn_bcg(
        FakeDecoderLayer(),
        hidden_states,
        SimpleNamespace(),
        torch.tensor([1]),
        torch.tensor([1]),
    )

    torch.testing.assert_close(output, torch.tensor([[3.0]]))
    assert events == [("enter", 37), ("run", 37), ("exit", 37)]


def test_bcg_recorder_layer_scope_is_same_layer_reentrant():
    recorder = object.__new__(_ExpertDistributionRecorderReal)
    recorder._current_layer_idx = Withable()

    with recorder.with_current_layer(37):
        with recorder.with_current_layer_if_absent(37):
            assert recorder._current_layer_idx.value == 37
        with pytest.raises(RuntimeError, match="layer mismatch"):
            with recorder.with_current_layer_if_absent(38):
                pass

    assert recorder._current_layer_idx.value is None
    with recorder.with_current_layer_if_absent(19):
        assert recorder._current_layer_idx.value == 19
    assert recorder._current_layer_idx.value is None
