import torch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _copy_output,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_copy_output_elides_exact_alias() -> None:
    destination = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source = destination.as_strided(destination.shape, destination.stride())
    version_before = destination._version

    result = _copy_output(destination, source)

    assert result is destination
    assert destination._version == version_before


def test_copy_output_preserves_non_alias_writeback() -> None:
    destination = torch.zeros((2, 4), dtype=torch.float32)
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    version_before = destination._version

    result = _copy_output(destination, source)

    assert result is destination
    assert destination._version == version_before + 1
    torch.testing.assert_close(destination, source)
