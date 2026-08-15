import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.srt.layers.quantization import v4_triton_kernels_moe
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci
from triton_kernels.routing import GatherIndx, ScatterIndx

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_compact_route_gate_is_disabled_during_graph_capture(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_V4_COMPACT_GPU_ROUTES", "1")
    monkeypatch.setenv("SGLANG_V4_COMPACT_GPU_ROUTE_MIN_TOKENS", "2")
    monkeypatch.setattr(capture_mode, "get_is_capture_mode", lambda: False)
    assert not v4_triton_kernels_moe._compact_gpu_routes_are_eligible(1)
    assert v4_triton_kernels_moe._compact_gpu_routes_are_eligible(2)

    monkeypatch.setattr(capture_mode, "get_is_capture_mode", lambda: True)
    assert not v4_triton_kernels_moe._compact_gpu_routes_are_eligible(1024)


def test_compact_routing_skips_invalid_routes_and_gathers_original_tokens(
    monkeypatch,
) -> None:
    captured = {}
    routing_data = object()
    scatter_indx = ScatterIndx(
        src_indx=torch.tensor([0, 1, 2], dtype=torch.int32),
        dst_indx=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    def make_routing(ids, weights, num_experts):
        captured.update(ids=ids.clone(), weights=weights.clone(), num=num_experts)
        return (
            routing_data,
            GatherIndx(
                src_indx=torch.tensor([2, 0, 1], dtype=torch.int32),
                dst_indx=torch.tensor([0, 1, 2], dtype=torch.int32),
            ),
            scatter_indx,
        )

    monkeypatch.setattr(
        v4_triton_kernels_moe, "_make_routing_data_v4", make_routing
    )
    topk_ids = torch.tensor([[2, -1, 0], [-1, 1, -1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.2, 0.0, 0.3], [0.0, 0.5, 0.0]])

    routing, gather, scatter, lookup = (
        v4_triton_kernels_moe._make_compact_routing_data_v4(
            topk_ids, topk_weights, 3
        )
    )

    assert routing is routing_data
    assert scatter is scatter_indx
    torch.testing.assert_close(
        captured["ids"], torch.tensor([[2], [0], [1]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        captured["weights"], torch.tensor([[0.2], [0.3], [0.5]])
    )
    assert captured["num"] == 3
    torch.testing.assert_close(
        gather.src_indx, torch.tensor([1, 0, 0], dtype=torch.int32)
    )
    torch.testing.assert_close(
        lookup, torch.tensor([[0, -1, 1], [-1, 2, -1]], dtype=torch.int32)
    )


def test_compact_routing_handles_a_batch_with_no_gpu_routes() -> None:
    topk_ids = torch.full((2, 3), -1, dtype=torch.int32)
    topk_weights = torch.zeros((2, 3))

    routing, gather, scatter, lookup = (
        v4_triton_kernels_moe._make_compact_routing_data_v4(
            topk_ids, topk_weights, 4
        )
    )

    assert routing is None
    assert gather is None
    assert scatter is None
    torch.testing.assert_close(lookup, torch.full((2, 3), -1, dtype=torch.int32))


def _install_compact_apply_stubs(monkeypatch) -> torch.Tensor:
    routing_data = SimpleNamespace(gate_scal=torch.ones(3))
    gather_indx = object()
    scatter_indx = object()
    lookup = torch.tensor([[0, -1, 1], [-1, 2, -1]], dtype=torch.int32)
    monkeypatch.setattr(
        v4_triton_kernels_moe,
        "_make_compact_routing_data_v4",
        lambda *_args: (routing_data, gather_indx, scatter_indx, lookup),
    )
    monkeypatch.setattr(v4_triton_kernels_moe, "_patch_strided_mxfp", lambda: None)

    compact_output = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]]
    )
    matmul_module = ModuleType("triton_kernels.matmul_ogs")

    def fake_matmul_ogs(inputs, *_args, gather_indx=None, scatter_indx=None, y=None, **_kwargs):
        assert y is None
        if gather_indx is not None:
            return torch.cat((inputs.new_ones((3, 2)), inputs.new_ones((3, 2))), dim=1)
        assert scatter_indx is not None
        return compact_output

    matmul_module.matmul_ogs = fake_matmul_ogs
    monkeypatch.setitem(sys.modules, "triton_kernels.matmul_ogs", matmul_module)

    sgl_kernel_module = ModuleType("sgl_kernel")
    sgl_kernel_module.silu_and_mul = lambda inputs, output: output.copy_(inputs[:, :2])
    monkeypatch.setitem(sys.modules, "sgl_kernel", sgl_kernel_module)

    def reduce_routes(compact, route_lookup, output):
        output.zero_()
        for token_index, row in enumerate(route_lookup):
            for compact_index in row:
                if compact_index >= 0:
                    output[token_index].add_(compact[int(compact_index)])
        return output

    monkeypatch.setattr(
        v4_triton_kernels_moe, "_reduce_compact_routes", reduce_routes
    )
    return compact_output


@pytest.mark.parametrize("caller_owned", [False, True])
def test_compact_apply_reduces_into_safe_token_output(
    monkeypatch, caller_owned: bool
) -> None:
    monkeypatch.setenv("SGLANG_V4_COMPACT_GPU_ROUTES", "1")
    monkeypatch.setenv("SGLANG_V4_COMPACT_GPU_ROUTE_MIN_TOKENS", "1")
    monkeypatch.setattr(capture_mode, "get_is_capture_mode", lambda: False)
    compact_output = _install_compact_apply_stubs(monkeypatch)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    original_hidden_states = hidden_states.clone()
    caller_output = hidden_states if caller_owned else None

    output = v4_triton_kernels_moe.apply_v4_triton_kernels_moe(
        hidden_states=hidden_states,
        w13_swiz=object(),
        w13_pcg=object(),
        w2_swiz=object(),
        w2_pcg=object(),
        topk_weights=torch.ones((2, 3)),
        topk_ids=torch.zeros((2, 3), dtype=torch.int32),
        intermediate_size=2,
        num_experts=3,
        caller_output=caller_output,
    )

    expected = torch.stack(
        (compact_output[0] + compact_output[1], compact_output[2])
    )
    torch.testing.assert_close(output, expected)
    if caller_owned:
        assert output.data_ptr() == hidden_states.data_ptr()
    else:
        assert output.data_ptr() != hidden_states.data_ptr()
        torch.testing.assert_close(hidden_states, original_hidden_states)


def test_compact_apply_returns_fresh_zeros_when_no_route_is_valid(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_V4_COMPACT_GPU_ROUTES", "1")
    monkeypatch.setenv("SGLANG_V4_COMPACT_GPU_ROUTE_MIN_TOKENS", "1")
    monkeypatch.setattr(capture_mode, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(v4_triton_kernels_moe, "_patch_strided_mxfp", lambda: None)
    monkeypatch.setattr(
        v4_triton_kernels_moe,
        "_make_compact_routing_data_v4",
        lambda *_args: (None, None, None, torch.full((2, 3), -1)),
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    original_hidden_states = hidden_states.clone()

    output = v4_triton_kernels_moe.apply_v4_triton_kernels_moe(
        hidden_states=hidden_states,
        w13_swiz=object(),
        w13_pcg=object(),
        w2_swiz=object(),
        w2_pcg=object(),
        topk_weights=torch.zeros((2, 3)),
        topk_ids=torch.full((2, 3), -1, dtype=torch.int32),
        intermediate_size=2,
        num_experts=3,
    )

    torch.testing.assert_close(output, torch.zeros_like(hidden_states))
    assert output.data_ptr() != hidden_states.data_ptr()
    torch.testing.assert_close(hidden_states, original_hidden_states)
