import sys
from types import ModuleType, SimpleNamespace

import torch
from sglang.srt.layers.quantization import v4_triton_kernels_moe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_caller_owned_gemm2_output_matches_allocating_path(monkeypatch) -> None:
    gather_index = object()
    scatter_index = object()
    routing_data = SimpleNamespace(gate_scal=torch.ones((2, 1)))
    monkeypatch.setattr(
        v4_triton_kernels_moe,
        "_make_routing_data_v4",
        lambda *args: (routing_data, gather_index, scatter_index),
    )
    monkeypatch.setattr(v4_triton_kernels_moe, "_patch_strided_mxfp", lambda: None)

    matmul_module = ModuleType("triton_kernels.matmul_ogs")

    def fake_matmul_ogs(
        inputs,
        *args,
        gather_indx=None,
        scatter_indx=None,
        y=None,
        **kwargs,
    ):
        if gather_indx is gather_index:
            return torch.cat((inputs[:, :2] + 1, inputs[:, :2] + 2), dim=-1)
        assert scatter_indx is scatter_index
        computed = torch.cat((inputs, inputs * 2), dim=-1)
        if y is None:
            return computed
        y[0].copy_(computed)
        return y[0]

    matmul_module.matmul_ogs = fake_matmul_ogs
    monkeypatch.setitem(sys.modules, "triton_kernels.matmul_ogs", matmul_module)

    sgl_kernel_module = ModuleType("sgl_kernel")

    def fake_silu_and_mul(inputs, output) -> None:
        midpoint = inputs.shape[-1] // 2
        output.copy_(inputs[:, :midpoint] + inputs[:, midpoint:])

    sgl_kernel_module.silu_and_mul = fake_silu_and_mul
    monkeypatch.setitem(sys.modules, "sgl_kernel", sgl_kernel_module)

    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    common = {
        "w13_swiz": object(),
        "w13_pcg": object(),
        "w2_swiz": object(),
        "w2_pcg": object(),
        "topk_weights": torch.ones((2, 1)),
        "topk_ids": torch.zeros((2, 1), dtype=torch.int32),
        "intermediate_size": 2,
        "num_experts": 1,
    }

    allocating = v4_triton_kernels_moe.apply_v4_triton_kernels_moe(
        hidden_states=hidden_states.clone(),
        **common,
    )
    caller_output = hidden_states.clone()
    inplace = v4_triton_kernels_moe.apply_v4_triton_kernels_moe(
        hidden_states=caller_output,
        caller_output=caller_output,
        **common,
    )

    torch.testing.assert_close(inplace, allocating)
    assert inplace.data_ptr() == caller_output.data_ptr()
