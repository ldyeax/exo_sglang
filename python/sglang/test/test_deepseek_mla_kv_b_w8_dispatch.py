from types import SimpleNamespace

import pytest
import torch

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    _uses_packed_linear_weights,
)


class _DecodeForwardMode:
    @staticmethod
    def is_decode_or_idle() -> bool:
        return True

    @staticmethod
    def is_target_verify() -> bool:
        return False

    @staticmethod
    def is_draft_extend() -> bool:
        return False

    @staticmethod
    def is_draft_extend_v2() -> bool:
        return False


def _attention(*, compact_w8: bool) -> DeepseekV2AttentionMLA:
    attention = object.__new__(DeepseekV2AttentionMLA)
    torch.nn.Module.__init__(attention)
    attention.kv_b_proj = SimpleNamespace(
        quant_method=SimpleNamespace(is_mla_kv_b_w8=compact_w8)
    )
    return attention


def _select_method(
    monkeypatch: pytest.MonkeyPatch,
    selected_method: AttnForwardMethod,
) -> None:
    server_args = SimpleNamespace(
        get_attention_backends=lambda: ("test_prefill", "test_decode")
    )
    monkeypatch.setattr(deepseek_v2, "get_server_args", lambda: server_args)
    monkeypatch.setattr(
        deepseek_v2,
        "get_attn_backend",
        lambda: SimpleNamespace(
            decode_attention_backend_str=None,
            prefill_attention_backend_str=None,
        ),
    )
    monkeypatch.setattr(
        deepseek_v2.AttentionBackendRegistry,
        "get_handler",
        staticmethod(
            lambda backend_name: (lambda attention, forward_batch: selected_method)
        ),
    )


def test_compact_w8_method_detection_uses_shared_marker() -> None:
    attention = _attention(compact_w8=True)

    assert attention._get_mla_kv_b_w8_method() is attention.kv_b_proj.quant_method


@pytest.mark.parametrize("quantization", ["gptq", "gptq_marlin"])
def test_gptq_body_linears_are_recognized_as_packed(
    quantization: str,
) -> None:
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(
            quant_config=SimpleNamespace(get_name=lambda: quantization)
        )
    )

    assert _uses_packed_linear_weights(layer)
    assert not hasattr(layer, "weight")


def test_compact_w8_dispatch_allows_absorbed_mla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _attention(compact_w8=True)
    _select_method(monkeypatch, AttnForwardMethod.MLA)

    selected = attention.dispatch_attn_forward_method(
        SimpleNamespace(forward_mode=_DecodeForwardMode())
    )

    assert selected == AttnForwardMethod.MLA
    assert attention.current_attention_backend == "test_decode"


@pytest.mark.parametrize(
    "selected_method",
    [
        AttnForwardMethod.MHA,
        AttnForwardMethod.MHA_CHUNKED_KV,
        AttnForwardMethod.MHA_ONE_SHOT,
    ],
)
def test_compact_w8_dispatch_forces_mha_heuristics_to_absorbed_mla(
    monkeypatch: pytest.MonkeyPatch,
    selected_method: AttnForwardMethod,
) -> None:
    attention = _attention(compact_w8=True)
    _select_method(monkeypatch, selected_method)

    selected = attention.dispatch_attn_forward_method(
        SimpleNamespace(forward_mode=_DecodeForwardMode())
    )

    assert selected == AttnForwardMethod.MLA
    assert attention.current_attention_backend == "test_decode"


@pytest.mark.parametrize(
    "selected_method",
    [
        AttnForwardMethod.MLA_FUSED_ROPE_ROCM,
        AttnForwardMethod.MLA_FUSED_ROPE_CPU,
    ],
)
def test_compact_w8_dispatch_rejects_platform_fused_paths(
    monkeypatch: pytest.MonkeyPatch,
    selected_method: AttnForwardMethod,
) -> None:
    attention = _attention(compact_w8=True)
    _select_method(monkeypatch, selected_method)

    with pytest.raises(RuntimeError, match="compact MLA kv_b W8 requires"):
        attention.dispatch_attn_forward_method(
            SimpleNamespace(forward_mode=_DecodeForwardMode())
        )


def test_other_quant_methods_keep_fused_rope_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _attention(compact_w8=False)
    _select_method(monkeypatch, AttnForwardMethod.MLA_FUSED_ROPE_ROCM)

    selected = attention.dispatch_attn_forward_method(
        SimpleNamespace(forward_mode=_DecodeForwardMode())
    )

    assert selected == AttnForwardMethod.MLA_FUSED_ROPE_ROCM
