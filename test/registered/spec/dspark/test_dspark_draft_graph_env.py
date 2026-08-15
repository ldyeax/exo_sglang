import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_worker_v2


def test_dsv4_env_disables_only_dspark_draft_graph(
    monkeypatch,
    caplog,
) -> None:
    worker = object.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker.ps = SimpleNamespace(tp_rank=0)
    worker._draft_worker = Mock()
    worker._draft_context = nullcontext
    worker.device = "cuda"
    worker.gpu_id = 0
    monkeypatch.setattr(
        dspark_worker_v2,
        "get_exec",
        lambda: SimpleNamespace(graph=SimpleNamespace(disable_cuda_graph=False)),
    )
    monkeypatch.setattr(dspark_worker_v2, "is_cuda", lambda: True)
    unavailable_memory_probe = Mock(
        side_effect=AssertionError("disabled draft graph must not probe GPU memory")
    )
    monkeypatch.setattr(
        dspark_worker_v2,
        "get_available_gpu_memory",
        unavailable_memory_probe,
    )

    with (
        envs.SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH.override(True),
        caplog.at_level(logging.INFO),
    ):
        worker.init_cuda_graphs()

    worker._draft_worker.init_cuda_graphs.assert_called_once_with(
        capture_decode_cuda_graph=False
    )
    unavailable_memory_probe.assert_not_called()
    assert "target verify CUDA graph remains enabled" in caplog.text
