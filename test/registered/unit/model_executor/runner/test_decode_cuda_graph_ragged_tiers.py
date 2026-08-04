from contextlib import contextmanager
from types import SimpleNamespace

import sglang.srt.model_executor.runner.decode_cuda_graph_runner as runner_module
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _single_request_dspark_runner() -> DecodeCudaGraphRunner:
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.capture_bs = [1]
    runner.captured_req_width = 6
    return runner


def _record_capture_shapes(runner, monkeypatch) -> list[tuple[int, int]]:
    runner.model_runner = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        model=object(),
        tp_group=object(),
    )
    runner.compile_bs = []
    runner.record_nolora_graph = False

    monkeypatch.setattr(
        runner_module,
        "get_available_gpu_memory",
        lambda *_args, **_kwargs: 8.0,
    )
    monkeypatch.setattr(
        runner_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=1),
    )
    monkeypatch.setattr(
        runner_module,
        "_set_capture_lora_variant",
        lambda _value: None,
    )

    patch_num_tokens: list[int] = []

    @contextmanager
    def fake_patch_model(model, should_compile, *, num_tokens, tp_group):
        del model, should_compile, tp_group
        patch_num_tokens.append(num_tokens)
        yield object()

    monkeypatch.setattr(
        runner_module.torch_compile_decoration,
        "patch_model",
        fake_patch_model,
    )

    capture_shapes: list[tuple[int, int]] = []

    def record_capture_shape(
        size,
        forward,
        stream_idx=None,
        variant_label=None,
        num_tokens=None,
    ) -> None:
        del forward, stream_idx, variant_label
        assert num_tokens is not None
        capture_shapes.append((size, num_tokens))

    runner.capture_one_shape = record_capture_shape
    runner._capture_one_stream()
    assert patch_num_tokens == [num_tokens for _, num_tokens in capture_shapes]
    return capture_shapes


def test_ragged_verify_uses_only_ordinary_bucket_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS", raising=False)

    assert _single_request_dspark_runner()._build_ragged_verify_token_buckets() == [
        6
    ]


def test_fine_ragged_verify_captures_every_sub_block_tier(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS", "1")

    assert _single_request_dspark_runner()._build_ragged_verify_token_buckets() == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_capture_stream_stores_every_fine_ragged_graph_key(monkeypatch) -> None:
    runner = _single_request_dspark_runner()
    runner.ragged_verify_mode = True
    runner.capture_num_tokens = [1, 2, 3, 4, 5, 6]
    runner.max_bs = 1

    capture_shapes = _record_capture_shapes(runner, monkeypatch)

    # Capture runs largest-first for graph-pool reuse. In ragged mode the graph
    # key is the explicit token tier, while the request-slot count remains one.
    assert capture_shapes == [(1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)]
    assert {
        runner._capture_graph_size(bs=bs, num_tokens=num_tokens)
        for bs, num_tokens in capture_shapes
    } == {1, 2, 3, 4, 5, 6}
