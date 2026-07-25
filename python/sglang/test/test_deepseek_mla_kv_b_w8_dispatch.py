from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.flashinfer_mla_backend as flashinfer_mla_backend
import sglang.srt.model_executor.model_runner as model_runner_module
import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.deepseek_v2 import _uses_packed_linear_weights


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


class _ZeroPrefixExtendForwardMode:
    @staticmethod
    def is_decode_or_idle() -> bool:
        return False

    @staticmethod
    def is_target_verify() -> bool:
        return False

    @staticmethod
    def is_draft_extend() -> bool:
        return False

    @staticmethod
    def is_extend_without_speculative() -> bool:
        return True


def _attention(*, compact_w8: bool) -> DeepseekV2AttentionMLA:
    attention = object.__new__(DeepseekV2AttentionMLA)
    torch.nn.Module.__init__(attention)
    attention.kv_b_proj = SimpleNamespace(
        quant_method=SimpleNamespace(
            is_mla_kv_b_w8=compact_w8,
            requires_mla_absorb=compact_w8,
        )
    )
    attention.flashinfer_mla_disable_ragged = False
    return attention


def _select_method(
    monkeypatch: pytest.MonkeyPatch,
    selected_method: AttnForwardMethod,
) -> None:
    server_args = SimpleNamespace(decode_attention_backend="test_decode")
    monkeypatch.setattr(deepseek_v2, "get_global_server_args", lambda: server_args)
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


@pytest.mark.parametrize(
    "selected_method",
    [
        AttnForwardMethod.MLA,
        AttnForwardMethod.MLA_FUSED_ROPE,
    ],
)
def test_compact_w8_dispatch_allows_gpu_absorbed_mla(
    monkeypatch: pytest.MonkeyPatch,
    selected_method: AttnForwardMethod,
) -> None:
    attention = _attention(compact_w8=True)
    _select_method(monkeypatch, selected_method)

    selected = attention.dispatch_attn_forward_method(
        SimpleNamespace(forward_mode=_DecodeForwardMode())
    )

    assert selected == selected_method
    assert attention.current_attention_backend == "test_decode"


@pytest.mark.parametrize(
    "selected_method",
    [
        AttnForwardMethod.MHA,
        AttnForwardMethod.MHA_CHUNKED_KV,
        AttnForwardMethod.MHA_ONE_SHOT,
    ],
)
def test_compact_w8_dispatch_rejects_mha_after_metadata_planning(
    monkeypatch: pytest.MonkeyPatch,
    selected_method: AttnForwardMethod,
) -> None:
    attention = _attention(compact_w8=True)
    _select_method(monkeypatch, selected_method)

    with pytest.raises(RuntimeError, match="compact MLA kv_b W8 requires"):
        attention.dispatch_attn_forward_method(
            SimpleNamespace(forward_mode=_DecodeForwardMode())
        )


def test_compact_w8_dispatch_rejects_cpu_fused_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _attention(compact_w8=True)
    _select_method(monkeypatch, AttnForwardMethod.MLA_FUSED_ROPE_CPU)

    with pytest.raises(RuntimeError, match="compact MLA kv_b W8 requires"):
        attention.dispatch_attn_forward_method(
            SimpleNamespace(forward_mode=_DecodeForwardMode())
        )


def test_other_quant_methods_keep_fused_rope_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _attention(compact_w8=False)
    _select_method(monkeypatch, AttnForwardMethod.MLA_FUSED_ROPE)

    selected = attention.dispatch_attn_forward_method(
        SimpleNamespace(forward_mode=_DecodeForwardMode())
    )

    assert selected == AttnForwardMethod.MLA_FUSED_ROPE


def test_model_runner_configures_flashinfer_metadata_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _attention(compact_w8=True)
    model = torch.nn.Module()
    model.add_module("attention", attention)

    server_args = SimpleNamespace(
        decode_attention_backend="flashinfer",
        flashinfer_mla_disable_ragged=False,
        prefill_attention_backend="flashinfer",
        speculative_attention_mode="prefill",
    )
    global_server_args = SimpleNamespace(
        decode_attention_backend="flashinfer",
        flashinfer_mla_disable_ragged=False,
        prefill_attention_backend="flashinfer",
        speculative_attention_mode="prefill",
    )
    runner = object.__new__(ModelRunner)
    runner.model = model
    runner.server_args = server_args
    monkeypatch.setattr(
        model_runner_module,
        "get_global_server_args",
        lambda: global_server_args,
    )

    runner.configure_compact_mla_kv_b_attention()

    assert server_args.flashinfer_mla_disable_ragged
    assert global_server_args.flashinfer_mla_disable_ragged
    assert attention.flashinfer_mla_disable_ragged

    metadata_update = {}

    class _IndicesUpdater:
        @staticmethod
        def update(*args, **kwargs) -> None:
            metadata_update["use_ragged"] = kwargs["use_ragged"]

    backend = object.__new__(FlashInferMLAAttnBackend)
    backend.indices_updater_prefill = _IndicesUpdater()
    backend.prefill_wrapper_paged = object()
    monkeypatch.setattr(
        flashinfer_mla_backend,
        "get_global_server_args",
        lambda: global_server_args,
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        extend_prefix_lens_cpu=[0],
        forward_mode=_ZeroPrefixExtendForwardMode(),
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        seq_lens_sum=3,
    )

    backend.init_forward_metadata(forward_batch)

    assert metadata_update == {"use_ragged": False}
    assert not backend.forward_metadata.use_ragged

    monkeypatch.setattr(
        deepseek_v2,
        "get_global_server_args",
        lambda: global_server_args,
    )
    selected = attention.dispatch_attn_forward_method(forward_batch)
    assert selected == AttnForwardMethod.MLA


def test_model_runner_synchronizes_mixed_mla_attention_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact_attention = _attention(compact_w8=True)
    ordinary_attention = _attention(compact_w8=False)
    model = torch.nn.Module()
    model.add_module("compact_attention", compact_attention)
    model.add_module("ordinary_attention", ordinary_attention)

    server_args = SimpleNamespace(flashinfer_mla_disable_ragged=False)
    global_server_args = SimpleNamespace(
        decode_attention_backend="flashinfer",
        flashinfer_mla_disable_ragged=False,
        prefill_attention_backend="flashinfer",
        speculative_attention_mode="prefill",
    )
    runner = object.__new__(ModelRunner)
    runner.model = model
    runner.server_args = server_args
    monkeypatch.setattr(
        model_runner_module,
        "get_global_server_args",
        lambda: global_server_args,
    )

    runner.configure_compact_mla_kv_b_attention()

    assert server_args.flashinfer_mla_disable_ragged
    assert global_server_args.flashinfer_mla_disable_ragged
    assert compact_attention.flashinfer_mla_disable_ragged
    assert ordinary_attention.flashinfer_mla_disable_ragged

    monkeypatch.setattr(
        deepseek_v2,
        "get_global_server_args",
        lambda: global_server_args,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        forward_mode=_ZeroPrefixExtendForwardMode(),
        get_max_chunk_capacity=lambda: 3,
        seq_lens_cpu=[3],
    )

    assert (
        compact_attention.dispatch_attn_forward_method(forward_batch)
        == AttnForwardMethod.MLA
    )
    assert (
        ordinary_attention.dispatch_attn_forward_method(forward_batch)
        == AttnForwardMethod.MLA
    )
