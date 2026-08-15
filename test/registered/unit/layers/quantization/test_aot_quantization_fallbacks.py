from types import SimpleNamespace

import torch
from sglang.kernels.jit.utils import KERNEL_PATH
from sglang.kernels.ops.quantization import (
    gptq_marlin,
    gptq_marlin_repack,
    hadamard,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_gptq_marlin_prefers_aot_operator(monkeypatch) -> None:
    sentinel = object()
    calls = []

    def aot(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(gptq_marlin, "_aot_gptq_marlin_gemm", aot)
    monkeypatch.setattr(
        gptq_marlin,
        "_jit_gptq_marlin_module",
        lambda _dtype: (_ for _ in ()).throw(AssertionError("JIT must not load")),
    )
    tensor = torch.zeros(1)
    result = gptq_marlin.gptq_marlin_gemm(
        tensor,
        None,
        tensor,
        tensor,
        None,
        None,
        None,
        None,
        tensor,
        object(),
        1,
        1,
        1,
    )

    assert result is sentinel
    assert len(calls) == 1


def test_gptq_marlin_jit_remains_a_fallback(monkeypatch) -> None:
    calls = []

    def jit_gemm(*args) -> None:
        calls.append(args)
        args[7].fill_(7)

    monkeypatch.setattr(gptq_marlin, "_aot_gptq_marlin_gemm", None)
    monkeypatch.setattr(
        gptq_marlin,
        "_jit_gptq_marlin_module",
        lambda _dtype: SimpleNamespace(gptq_marlin_gemm=jit_gemm),
    )
    tensor = torch.zeros(1)
    result = gptq_marlin.gptq_marlin_gemm(
        tensor,
        None,
        tensor,
        tensor,
        None,
        None,
        None,
        None,
        tensor,
        SimpleNamespace(id=3),
        1,
        1,
        1,
    )

    assert len(calls) == 1
    torch.testing.assert_close(result, torch.full((1, 1), 7.0))


def test_gptq_marlin_repack_prefers_aot_operator(monkeypatch) -> None:
    sentinel = object()
    calls = []

    def aot(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(gptq_marlin_repack, "_aot_gptq_marlin_repack", aot)
    monkeypatch.setattr(
        gptq_marlin_repack,
        "_jit_gptq_marlin_repack_module",
        lambda: (_ for _ in ()).throw(AssertionError("JIT must not load")),
    )
    tensor = torch.zeros(1, dtype=torch.int32)
    result = gptq_marlin_repack.gptq_marlin_repack(tensor, tensor, 16, 16, 4)

    assert result is sentinel
    assert len(calls) == 1


def test_hadamard_prefers_aot_and_retains_jit_fallback(monkeypatch) -> None:
    fake_cuda_tensor = SimpleNamespace(is_cuda=True, dtype=torch.float16)
    sentinel = object()
    assert (
        hadamard._run_hadamard_transform(
            fake_cuda_tensor,
            0.5,
            8,
            lambda x, scale: sentinel,
            "hadamard_transform",
        )
        is sentinel
    )

    jit_kernel = object()
    monkeypatch.setattr(
        hadamard,
        "_jit_hadamard_module",
        lambda _dtype: SimpleNamespace(hadamard_transform=jit_kernel),
    )
    seen = []

    def fake_impl(x, scale, pad_multiple, kernel_fn):
        seen.append((x, scale, pad_multiple, kernel_fn))
        return sentinel

    monkeypatch.setattr(hadamard, "_hadamard_transform_impl", fake_impl)
    assert (
        hadamard._run_hadamard_transform(
            fake_cuda_tensor, 0.25, 8, None, "hadamard_transform"
        )
        is sentinel
    )
    assert seen == [(fake_cuda_tensor, 0.25, 8, jit_kernel)]


def test_fused_rope_kernel_table_uses_portable_positional_initializers() -> None:
    source = (
        KERNEL_PATH / "csrc" / "deepseek_v4" / "fused_norm_rope.cuh"
    ).read_text()
    table = source.split("static constexpr KernelType kernel_table[3] = {", 1)[1]
    table = table.split("};", 1)[0]

    assert "[static_cast<int>(" not in table
    assert [line.strip().rstrip(",") for line in table.splitlines() if line.strip()] == [
        "fused_kernel<CompressExtend>",
        "fused_kernel<CompressDecode>",
        "fused_kernel<DefaultForward>",
    ]
