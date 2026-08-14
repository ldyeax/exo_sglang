from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import mxfp4_triton_kernels_moe
from sglang.srt.layers.quantization.mxfp4_triton_kernels_moe import (
    Mxfp4TritonKernelsMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_production_adapter_keeps_fused_load_apply_contract(monkeypatch) -> None:
    captured = {}

    def apply_v4_triton_kernels_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["caller_output"]

    kernel = SimpleNamespace(
        fused_t5_moe_enabled=lambda: True,
        apply_v4_triton_kernels_moe=apply_v4_triton_kernels_moe,
    )
    monkeypatch.setattr(
        mxfp4_triton_kernels_moe,
        "_portable_kernel_module",
        lambda: kernel,
    )

    method = Mxfp4TritonKernelsMoEMethod(object(), prefix="model.layers.3.mlp")
    method.create_moe_runner(
        SimpleNamespace(),
        SimpleNamespace(
            kt_global_to_local_expert_mapping=torch.arange(8, dtype=torch.int32),
            swiglu_limit=None,
        ),
    )
    hidden_states = torch.randn(5, 16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.rand(5, 6),
        topk_ids=torch.randint(0, 8, (5, 6), dtype=torch.int32),
        router_logits=torch.empty(0),
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=topk_output,
    )
    layer = SimpleNamespace(
        _dsv4_tk_w13=object(),
        _dsv4_tk_w13_precision=object(),
        _dsv4_tk_w2=object(),
        _dsv4_tk_w2_precision=object(),
        _dsv4_tk_intermediate_size=32,
        _dsv4_tk_num_experts=8,
    )
    gpu_experts_mask = torch.ones(8, dtype=torch.bool)
    logical_to_gpu_index = torch.arange(8, dtype=torch.int32)

    result = method.apply_with_kt_fused_routing(
        layer,
        dispatch_output,
        gpu_experts_mask=gpu_experts_mask,
        logical_to_gpu_index=logical_to_gpu_index,
        caller_output=hidden_states,
    )

    assert result.hidden_states.data_ptr() == hidden_states.data_ptr()
    assert captured["fused_t5_moe"] is True
    assert captured["gpu_experts_mask"] is gpu_experts_mask
    assert captured["logical_to_gpu_index"] is logical_to_gpu_index
    assert captured["caller_output"] is hidden_states


def test_production_adapter_rejects_fused_routing_without_compact_ids() -> None:
    method = Mxfp4TritonKernelsMoEMethod(object(), prefix="")
    method._kt_compact_ids = False

    with pytest.raises(RuntimeError, match="requires compact V4 MXFP4 experts"):
        method.apply_with_kt_fused_routing(
            SimpleNamespace(),
            SimpleNamespace(),
            gpu_experts_mask=torch.empty(0, dtype=torch.bool),
            logical_to_gpu_index=torch.empty(0, dtype=torch.int32),
        )
