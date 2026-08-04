from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.models import (
    deepseek_v2,
    deepseek_v4,
    deepseek_v4_dspark,
)
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeStream:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    def wait_stream(self, other: "_FakeStream") -> None:
        self.events.append(f"{self.name}.wait({other.name})")


class _FakeStreamContext:
    def __init__(self, cuda, stream: _FakeStream):
        self.cuda = cuda
        self.stream = stream
        self.previous = None

    def __enter__(self):
        self.previous = self.cuda.current
        self.cuda.current = self.stream
        self.cuda.events.append(f"enter({self.stream.name})")

    def __exit__(self, *_):
        self.cuda.events.append(f"exit({self.stream.name})")
        self.cuda.current = self.previous


class _FakeCuda:
    def __init__(self, events: list[str]):
        self.events = events
        self.current = _FakeStream("main", events)

    def current_stream(self, *_args, **_kwargs) -> _FakeStream:
        return self.current

    def stream(self, stream: _FakeStream) -> _FakeStreamContext:
        return _FakeStreamContext(self, stream)


class _FakeTensor:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self.shape = (1, 8)

    def record_stream(self, stream: _FakeStream) -> None:
        self.events.append(f"{self.name}.record({stream.name})")


def test_bcg_context_publishes_static_and_raw_runtime_batches():
    captured_batch = SimpleNamespace(name="capture")
    static_batch = SimpleNamespace(name="static-padded")
    runtime_batch = SimpleNamespace(name="runtime-raw")

    with set_tc_piecewise_forward_context(
        static_batch,
        [],
        None,
        [],
        [],
        runtime_forward_batch=runtime_batch,
    ):
        assert deepseek_v4._get_bcg_static_forward_batch(captured_batch) is static_batch
        assert (
            deepseek_v4._get_bcg_runtime_forward_batch(captured_batch)
            is runtime_batch
        )

    assert deepseek_v4._get_bcg_static_forward_batch(captured_batch) is captured_batch
    assert deepseek_v4._get_bcg_runtime_forward_batch(captured_batch) is captured_batch


def test_moe_side_stream_result_is_retained_by_consumer_stream():
    events = []
    producer = _FakeStream("producer", events)
    consumer = _FakeStream("consumer", events)
    output = _FakeTensor("output", events)

    deepseek_v2._join_cuda_side_stream_tensor(
        output,
        producer_stream=producer,
        consumer_stream=consumer,
    )

    assert events == [
        "consumer.wait(producer)",
        "output.record(consumer)",
    ]


@pytest.mark.parametrize(
    ("breakable", "expected_events"),
    [
        (
            True,
            [
                "gate",
                "alt.wait(main)",
                "enter(alt)",
                "shared",
                "exit(alt)",
                "join",
                "topk",
                "experts",
            ],
        ),
        (
            False,
            [
                "gate",
                "alt.wait(main)",
                "enter(alt)",
                "shared",
                "exit(alt)",
                "topk",
                "experts",
                "join",
            ],
        ),
    ],
)
def test_deepep_side_stream_joins_on_consumer_for_both_graph_paths(
    monkeypatch, breakable, expected_events
):
    events = []
    fake_cuda = _FakeCuda(events)
    alt_stream = _FakeStream("alt", events)
    shared_output = torch.tensor([[1.0, 2.0]])
    routed_output = torch.tensor([[3.0, 4.0]])

    class FakeGate:
        def __call__(self, hidden_states, *, forward_batch):
            events.append("gate")
            return object()

    class FakeTopK:
        def __call__(self, hidden_states, router_logits, **kwargs):
            events.append("topk")
            return object()

    class FakeExperts:
        should_fuse_routed_scaling_factor_in_topk = True

        def __call__(self, *, hidden_states, topk_output):
            events.append("experts")
            return routed_output

    def forward_shared_experts(hidden_states):
        events.append("shared")
        return shared_output

    def join_side_stream_tensor(tensor, *, producer_stream, consumer_stream) -> None:
        assert tensor is shared_output
        assert producer_stream is alt_stream
        assert consumer_stream is fake_cuda.current
        events.append("join")

    monkeypatch.setattr(torch.cuda, "current_stream", fake_cuda.current_stream)
    monkeypatch.setattr(torch.cuda, "stream", fake_cuda.stream)
    monkeypatch.setattr(deepseek_v2, "is_in_breakable_cuda_graph", lambda: breakable)
    monkeypatch.setattr(
        deepseek_v2, "_join_cuda_side_stream_tensor", join_side_stream_tensor
    )
    moe = SimpleNamespace(
        _fuse_shared_experts_inside_sbo=False,
        is_nextn=True,
        num_fused_shared_experts=0,
        gate=FakeGate(),
        topk=FakeTopK(),
        experts=FakeExperts(),
        alt_stream=alt_stream,
        _forward_shared_experts=forward_shared_experts,
        routed_scaling_factor=1.0,
    )
    hidden_states = torch.tensor([[0.5, -0.5]])
    forward_batch = SimpleNamespace(num_token_non_padded=1)

    result = deepseek_v2.DeepseekV2MoE.forward_deepep(moe, hidden_states, forward_batch)

    torch.testing.assert_close(result, torch.tensor([[4.0, 6.0]]))
    assert events == expected_events


def test_attention_bcg_carries_explicit_context_and_restores_cache_loc():
    calls = []

    class FakeAttentionBackend:
        def forward(self, **kwargs):
            calls.append(
                {
                    **kwargs,
                    "out_cache_loc": kwargs["forward_batch"].out_cache_loc.clone(),
                }
            )
            return kwargs["q"] + kwargs["k"]

    capture_num_tokens = 1024
    real_num_tokens = 646
    original_out_cache_loc = torch.arange(capture_num_tokens)
    forward_batch = SimpleNamespace(
        num_token_non_padded_cpu=real_num_tokens,
        out_cache_loc=original_out_cache_loc,
    )
    query = torch.arange(capture_num_tokens, dtype=torch.float32).unsqueeze(1)
    key_value = query + 1
    output = torch.full_like(query, -1)
    attention_layer = SimpleNamespace(layer_id=7)
    attn_sink = torch.tensor([0.25])

    result = deepseek_v4.deepseek_v4_attention_bcg(
        query,
        key_value,
        output,
        FakeAttentionBackend(),
        attention_layer,
        forward_batch,
        4,
        attn_sink,
        False,
    )

    assert result is None
    assert output.shape == (capture_num_tokens, 1)
    torch.testing.assert_close(
        output[:real_num_tokens],
        query[:real_num_tokens] + key_value[:real_num_tokens],
    )
    torch.testing.assert_close(
        output[real_num_tokens:],
        torch.full_like(output[real_num_tokens:], -1),
    )
    assert forward_batch.out_cache_loc is original_out_cache_loc
    assert len(calls) == 1
    assert calls[0]["forward_batch"] is forward_batch
    assert calls[0]["layer"] is attention_layer
    assert calls[0]["attn_sink"] is attn_sink
    torch.testing.assert_close(calls[0]["q"], query[:real_num_tokens])
    torch.testing.assert_close(calls[0]["k"], key_value[:real_num_tokens])
    torch.testing.assert_close(calls[0]["v"], key_value[:real_num_tokens])
    torch.testing.assert_close(
        calls[0]["out_cache_loc"], original_out_cache_loc[:real_num_tokens]
    )


def test_attention_bcg_uses_live_replay_batch_for_padded_bucket(monkeypatch):
    calls = []

    class FakeAttentionBackend:
        def forward(self, **kwargs):
            calls.append(
                {
                    **kwargs,
                    "out_cache_loc": kwargs["forward_batch"].out_cache_loc.clone(),
                }
            )
            return kwargs["q"] + kwargs["k"]

    capture_num_tokens = 1024
    raw_num_tokens = 515
    # The graph input registry zero-pads cache locations. Reading the captured
    # bucket length here would write all 509 padding rows into KV slot zero.
    live_out_cache_loc = torch.cat(
        [
            torch.arange(raw_num_tokens) + 100,
            torch.zeros(capture_num_tokens - raw_num_tokens, dtype=torch.int64),
        ]
    )
    # Capture and replay point at the same stable token-axis buffer; only the
    # retained capture-time Python token count is stale.
    captured_batch = SimpleNamespace(
        num_token_non_padded_cpu=capture_num_tokens,
        out_cache_loc=live_out_cache_loc,
    )
    live_batch = SimpleNamespace(
        num_token_non_padded_cpu=raw_num_tokens,
        out_cache_loc=live_out_cache_loc,
    )
    monkeypatch.setattr(
        deepseek_v4,
        "get_tc_piecewise_forward_context",
        lambda: SimpleNamespace(forward_batch=live_batch),
    )

    query = torch.arange(capture_num_tokens, dtype=torch.float32).unsqueeze(1)
    key_value = query + 1
    output = torch.full_like(query, -1)
    deepseek_v4.deepseek_v4_attention_bcg(
        query,
        key_value,
        output,
        FakeAttentionBackend(),
        SimpleNamespace(layer_id=7),
        captured_batch,
        4,
        torch.tensor([0.25]),
        True,
    )

    assert len(calls) == 1
    assert calls[0]["forward_batch"] is live_batch
    assert calls[0]["q"].shape[0] == raw_num_tokens
    assert calls[0]["k"].shape[0] == raw_num_tokens
    torch.testing.assert_close(
        calls[0]["out_cache_loc"], live_out_cache_loc[:raw_num_tokens]
    )
    torch.testing.assert_close(
        output[:raw_num_tokens],
        query[:raw_num_tokens] + key_value[:raw_num_tokens],
    )
    torch.testing.assert_close(
        output[raw_num_tokens:],
        torch.full_like(output[raw_num_tokens:], -1),
    )
    assert captured_batch.num_token_non_padded_cpu == capture_num_tokens


def test_attention_module_bcg_excludes_padded_cache_writes(monkeypatch):
    capture_num_tokens = 1024
    raw_num_tokens = 515
    captured_batch = SimpleNamespace(num_token_non_padded_cpu=capture_num_tokens)
    static_batch = SimpleNamespace(
        num_token_non_padded_cpu=raw_num_tokens,
        kind="static-padded",
    )
    live_batch = SimpleNamespace(
        num_token_non_padded_cpu=raw_num_tokens,
        kind="runtime-raw",
    )
    monkeypatch.setattr(
        deepseek_v4,
        "get_tc_piecewise_forward_context",
        lambda: SimpleNamespace(
            forward_batch=static_batch,
            runtime_forward_batch=live_batch,
        ),
    )
    calls = []

    class FakeAttentionModule:
        def __call__(self, *, x, positions, forward_batch, x_quant):
            calls.append((x, positions, forward_batch, x_quant))
            return x + 3

    x = torch.arange(capture_num_tokens, dtype=torch.float32).unsqueeze(1)
    positions = torch.arange(capture_num_tokens)
    x_quant = x + 100
    output = torch.full_like(x, -1)

    result = deepseek_v4.deepseek_v4_attention_module_bcg(
        FakeAttentionModule(),
        x,
        positions,
        output,
        captured_batch,
        x_quant,
    )

    assert result is None
    assert len(calls) == 1
    called_x, called_positions, called_batch, called_x_quant = calls[0]
    assert called_batch is live_batch
    assert called_x.shape[0] == raw_num_tokens
    assert called_positions.shape[0] == raw_num_tokens
    assert called_x_quant.shape[0] == raw_num_tokens
    torch.testing.assert_close(output[:raw_num_tokens], x[:raw_num_tokens] + 3)
    torch.testing.assert_close(
        output[raw_num_tokens:],
        torch.full_like(output[raw_num_tokens:], -1),
    )


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
            events.append(("run", self.layer_id, forward_batch.name))
            return hidden_states + 1

    monkeypatch.setattr(
        deepseek_v4,
        "get_global_expert_distribution_recorder",
        lambda: FakeRecorder(),
    )
    hidden_states = torch.tensor([[2.0]])
    captured_batch = SimpleNamespace(name="capture")
    live_batch = SimpleNamespace(name="replay")
    runtime_batch = SimpleNamespace(name="runtime-raw")
    monkeypatch.setattr(
        deepseek_v4,
        "get_tc_piecewise_forward_context",
        lambda: SimpleNamespace(
            forward_batch=live_batch,
            runtime_forward_batch=runtime_batch,
        ),
    )

    output = deepseek_v4.deepseek_v4_moe_ffn_bcg(
        FakeDecoderLayer(),
        hidden_states,
        captured_batch,
        torch.tensor([1]),
        torch.tensor([1]),
    )

    torch.testing.assert_close(output, torch.tensor([[3.0]]))
    assert events == [
        ("enter", 37),
        ("run", 37, "replay"),
        ("exit", 37),
    ]


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


@pytest.mark.parametrize("in_breakable_graph", [False, True])
def test_dspark_stage_keeps_complete_moe_inside_bcg(monkeypatch, in_breakable_graph):
    calls = []

    class FakeStage:
        dim = 4

        def _run_moe_ffn_dp_sync(
            self,
            hidden_states,
            forward_batch,
            *,
            input_ids,
            input_ids_global,
        ):
            calls.append(("direct", tuple(hidden_states.shape)))
            assert input_ids is None
            assert input_ids_global is None
            return hidden_states + 1

    def fake_bcg(stage, hidden_states, forward_batch, input_ids, input_ids_global):
        calls.append(("bcg", tuple(hidden_states.shape)))
        assert input_ids is None
        assert input_ids_global is None
        return hidden_states + 2

    monkeypatch.setattr(
        deepseek_v4_dspark,
        "is_in_breakable_cuda_graph",
        lambda: in_breakable_graph,
    )
    monkeypatch.setattr(deepseek_v4_dspark, "bcg_deepseek_v4_moe_ffn", fake_bcg)
    hidden_states = torch.zeros((2, 3, 4))

    output = deepseek_v4_dspark.DSparkV4Stage._run_ffn(
        FakeStage(), hidden_states, SimpleNamespace()
    )

    expected_path = "bcg" if in_breakable_graph else "direct"
    expected_increment = 2 if in_breakable_graph else 1
    torch.testing.assert_close(output, hidden_states + expected_increment)
    assert calls == [(expected_path, (6, 4))]
