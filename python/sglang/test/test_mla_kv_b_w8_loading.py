from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    _fuse_qkv_a_projection_tensors,
)


_CONFIG_KEY = "exo_mla_kv_b_w8"
_PARAMETER_NAMES = ("kc_qweight", "kc_scales", "vc_qweight", "vc_scales")


def _canonical_weight_names(layer_id: int) -> set[str]:
    prefix = f"model.layers.{layer_id}.self_attn.kv_b_proj."
    return {prefix + name for name in _PARAMETER_NAMES}


def _compact_attention() -> SimpleNamespace:
    projection = SimpleNamespace(
        quant_method=SimpleNamespace(is_mla_kv_b_w8=True),
        **{name: object() for name in _PARAMETER_NAMES},
    )
    return SimpleNamespace(kv_b_proj=projection)


def test_plain_gptq_rejects_compact_mla_checkpoint() -> None:
    config = {_CONFIG_KEY: {}}

    with pytest.raises(ValueError, match="plain GPTQ loader"):
        GPTQConfig.from_config(config)
    with pytest.raises(ValueError, match="explicit --quantization gptq"):
        GPTQMarlinConfig.override_quantization_method(config, "gptq")


def test_gptq_marlin_selects_canonical_mla_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("sglang.srt.layers.quantization.mla_kv_b_w8")

    class FakeMLAKVW8Method:
        def __init__(self, quant_config, format_config) -> None:
            self.quant_config = quant_config
            self.format_config = format_config

    fake_module.GPTQMLAKVW8Method = FakeMLAKVW8Method
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.mla_kv_b_w8",
        fake_module,
    )
    format_config = {"format": "test"}
    config = GPTQMarlinConfig(
        weight_bits=8,
        group_size=-1,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={_CONFIG_KEY: format_config},
    )

    method = config.get_quant_method(
        torch.nn.Module(),
        "model.layers.0.self_attn.kv_b_proj",
    )

    assert isinstance(method, FakeMLAKVW8Method)
    assert method.quant_config is config
    assert method.format_config is format_config


def test_fused_gptq_g_idx_is_shared_not_concatenated() -> None:
    q_a_g_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    kv_a_g_idx = q_a_g_idx.clone()

    fused = _fuse_qkv_a_projection_tensors(
        q_a_g_idx,
        kv_a_g_idx,
        q_a_proj_name="model.q_a_proj.g_idx",
        kv_a_proj_name="model.kv_a_proj_with_mqa.g_idx",
        cat_dim=1,
    )

    assert fused is q_a_g_idx
    with pytest.raises(ValueError, match="different GPTQ g_idx"):
        _fuse_qkv_a_projection_tensors(
            q_a_g_idx,
            torch.tensor([0, 1, 1, 1], dtype=torch.int32),
            q_a_proj_name="model.q_a_proj.g_idx",
            kv_a_proj_name="model.kv_a_proj_with_mqa.g_idx",
            cat_dim=1,
        )


def test_compact_validation_covers_whole_omitted_local_layer() -> None:
    loader = DeepseekV2WeightLoaderMixin()
    loader.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(self_attn=_compact_attention()),
            SimpleNamespace(self_attn=_compact_attention()),
        ],
    )
    loader.config = SimpleNamespace(num_hidden_layers=2)

    with pytest.raises(ValueError, match=r"model\.layers\.1.*kc_qweight"):
        loader.post_load_weights(weight_names=_canonical_weight_names(0))


@pytest.mark.parametrize("missing_parameter", _PARAMETER_NAMES)
def test_compact_validation_covers_each_missing_mtp_tensor(
    missing_parameter: str,
) -> None:
    layer_id = 78
    loader = DeepseekV2WeightLoaderMixin()
    loader.model = SimpleNamespace(
        decoder=SimpleNamespace(self_attn=_compact_attention()),
    )
    loader.config = SimpleNamespace(num_hidden_layers=layer_id)
    names = _canonical_weight_names(layer_id)
    names.remove(
        f"model.layers.{layer_id}.self_attn.kv_b_proj.{missing_parameter}",
    )

    with pytest.raises(
        ValueError,
        match=rf"model\.layers\.78.*{missing_parameter}",
    ):
        loader.post_load_weights(is_nextn=True, weight_names=names)


def test_compact_validation_accepts_complete_target_and_mtp_layers() -> None:
    target_loader = DeepseekV2WeightLoaderMixin()
    target_loader.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(self_attn=_compact_attention()),
            SimpleNamespace(self_attn=_compact_attention()),
        ],
    )
    target_loader.config = SimpleNamespace(num_hidden_layers=2)
    target_loader.post_load_weights(
        weight_names=_canonical_weight_names(0) | _canonical_weight_names(1)
    )

    mtp_loader = DeepseekV2WeightLoaderMixin()
    mtp_loader.model = SimpleNamespace(
        decoder=SimpleNamespace(self_attn=_compact_attention()),
    )
    mtp_loader.config = SimpleNamespace(num_hidden_layers=78)
    mtp_loader.post_load_weights(
        is_nextn=True,
        weight_names=_canonical_weight_names(78),
    )
