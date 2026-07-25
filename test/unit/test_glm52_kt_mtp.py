from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.utils import (
    get_kt_ep_weight_layer_index,
    speculative_kt_ep_context,
)
from sglang.srt.speculative.kt_mtp import (
    KTMTPAdmission,
    KTMTPAdmissionError,
    admit_glm52_kt_mtp,
    select_glm52_mtp_nonexpert_weights,
    validate_loaded_glm52_kt_mtp,
)


def _glm52_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
        hidden_size=6144,
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        index_share_for_mtp_iteration=True,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )


def _server_args(model_path: str = "/models/glm52") -> SimpleNamespace:
    return SimpleNamespace(
        model_path=model_path,
        speculative_draft_model_path=model_path,
        speculative_algorithm="EAGLE",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=2,
        speculative_draft_model_quantization=None,
        tp_size=2,
        pp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        speculative_moe_a2a_backend="none",
        kt_weight_path="/models/glm52-AMXINT4",
        kt_method="AMXINT4",
        kt_cpuinfer=112,
        kt_threadpool_count=2,
        kt_numa_nodes=[0, 1],
        kt_num_gpu_experts=0,
        kt_gpu_experts_ratio=None,
        kt_max_deferred_experts_per_token=0,
        kt_gpu_prefill_token_threshold=None,
        kt_enable_dynamic_expert_update=False,
        kt_expert_lora_path=None,
        disable_shared_experts_fusion=True,
        disable_cuda_graph=True,
        enable_dp_attention=False,
        enable_eplb=False,
        record_kt_gpu_expert_distribution=False,
        init_expert_location="trivial",
        chunked_prefill_size=8192,
        kt_expert_placement_strategy="uniform",
        kt_lora_path=None,
    )


def test_admits_exact_glm52_tp2_pp1_topology_without_artifact_io():
    admission = admit_glm52_kt_mtp(
        _server_args(), _glm52_config(), validate_artifacts=False
    )

    assert admission.enabled
    assert admission.physical_layer_index == 78
    assert "AMXINT4" in admission.reason


def test_non_glm_draft_retains_gpu_only_behavior():
    config = _glm52_config()
    config.architectures = ["DeepseekV3ForCausalLM"]

    admission = admit_glm52_kt_mtp(
        _server_args(), config, validate_artifacts=False
    )

    assert not admission.enabled
    assert admission.physical_layer_index is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pp_size", 3),
        ("tp_size", 1),
        ("kt_num_gpu_experts", 1),
        ("kt_method", "BF16"),
        ("speculative_num_steps", 2),
        ("disable_cuda_graph", False),
    ],
)
def test_glm52_candidate_rejects_unsupported_runtime_values(field: str, value):
    server_args = _server_args()
    setattr(server_args, field, value)

    with pytest.raises(KTMTPAdmissionError, match=field):
        admit_glm52_kt_mtp(
            server_args, _glm52_config(), validate_artifacts=False
        )


def test_glm52_candidate_requires_same_checkpoint_for_target_and_draft():
    server_args = _server_args()
    server_args.speculative_draft_model_path = "/models/other"

    with pytest.raises(KTMTPAdmissionError, match="must resolve"):
        admit_glm52_kt_mtp(
            server_args, _glm52_config(), validate_artifacts=False
        )


def test_speculative_context_maps_only_local_layer_zero_and_restores():
    assert get_kt_ep_weight_layer_index(7) == 7

    with speculative_kt_ep_context(enabled=True, physical_layer_index=78):
        assert get_kt_ep_weight_layer_index(0) == 78
        with pytest.raises(RuntimeError, match="local layer zero"):
            get_kt_ep_weight_layer_index(1)

    assert get_kt_ep_weight_layer_index(7) == 7


class _Scalar:
    def __init__(self, value: int):
        self.value = value

    def item(self) -> int:
        return self.value


class _Mask:
    def __init__(self, count: int):
        self.count = count

    def sum(self) -> _Scalar:
        return _Scalar(self.count)


def _loaded_draft(*, layer_index: int = 78, gpu_expert_count: int = 0):
    kt_config = SimpleNamespace(
        layer_idx=layer_index,
        gpu_experts_mask=_Mask(gpu_expert_count),
    )
    quant_method = SimpleNamespace(_quant_wrapper_id="kt_ep", kt_config=kt_config)
    experts = SimpleNamespace(quant_method=quant_method)
    mlp = SimpleNamespace(experts=experts)
    decoder = SimpleNamespace(mlp=mlp)
    return SimpleNamespace(model=SimpleNamespace(decoder=decoder))


def test_loaded_draft_receipt_proves_physical_layer_and_cpu_ownership():
    receipt = validate_loaded_glm52_kt_mtp(
        _loaded_draft(),
        KTMTPAdmission(enabled=True, physical_layer_index=78),
    )

    assert receipt == {
        "enabled": True,
        "physical_layer_index": 78,
        "gpu_expert_count": 0,
        "wrapper_id": "kt_ep",
    }


def test_loaded_draft_receipt_rejects_local_layer_zero_mapping():
    with pytest.raises(KTMTPAdmissionError, match="physical_layer_index=0"):
        validate_loaded_glm52_kt_mtp(
            _loaded_draft(layer_index=0),
            KTMTPAdmission(enabled=True, physical_layer_index=78),
        )


def test_draft_weight_filter_excludes_bf16_routed_experts_before_loading():
    weight_map = {
        "model.layers.77.self_attn.q_proj.weight": "target.safetensors",
        "model.layers.78.self_attn.q_proj.weight": "mtp-header.safetensors",
        "model.layers.78.mlp.gate.weight": "mtp-tail.safetensors",
        "model.layers.78.mlp.experts.0.gate_proj.weight": "mtp-expert.safetensors",
        "model.layers.78.mlp.experts.255.down_proj.weight": "mtp-expert.safetensors",
    }

    names, shards = select_glm52_mtp_nonexpert_weights(weight_map, 78)

    assert names == {
        "model.layers.78.self_attn.q_proj.weight",
        "model.layers.78.mlp.gate.weight",
    }
    assert shards == {"mtp-header.safetensors", "mtp-tail.safetensors"}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _config_json() -> dict[str, object]:
    config = vars(_glm52_config()).copy()
    return config


def _make_artifacts(root: Path) -> tuple[Path, Path]:
    model_path = root / "model"
    weight_path = root / "weights"
    model_path.mkdir()
    weight_path.mkdir()

    model_shard = "model-00001-of-00001.safetensors"
    weight_shard = "model-00001-of-00001.safetensors"
    (model_path / model_shard).touch()
    (weight_path / weight_shard).touch()

    prefix = "model.layers.78."
    source_names = [
        prefix + "eh_proj.weight",
        prefix + "enorm.weight",
        prefix + "hnorm.weight",
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
        prefix + "mlp.experts.0.gate_proj.weight",
        prefix + "mlp.experts.255.down_proj.weight",
        prefix + "self_attn.kv_b_proj.weight",
        prefix + "shared_head.norm.weight",
    ]
    _write_json(
        model_path / "model.safetensors.index.json",
        {"weight_map": {name: model_shard for name in source_names}},
    )

    kt_names = []
    for projection in ("gate", "up", "down"):
        for expert_index in range(256):
            for numa_slot in range(2):
                stem = (
                    f"blk.78.ffn_{projection}_exps.{expert_index}."
                    f"numa.{numa_slot}"
                )
                kt_names.extend((stem + ".weight", stem + ".scale"))
    _write_json(weight_path / "config.json", _config_json())
    _write_json(
        weight_path / "model.safetensors.index.json",
        {"weight_map": {name: weight_shard for name in kt_names}},
    )
    return model_path, weight_path


def test_artifact_admission_proves_model_layer_78_and_blk_78(tmp_path: Path):
    model_path, weight_path = _make_artifacts(tmp_path)
    server_args = _server_args(str(model_path))
    server_args.kt_weight_path = str(weight_path)

    admission = admit_glm52_kt_mtp(server_args, _glm52_config())

    assert admission.enabled
    assert admission.physical_layer_index == 78


def test_artifact_admission_rejects_incomplete_blk_78(tmp_path: Path):
    model_path, weight_path = _make_artifacts(tmp_path)
    index_path = weight_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["blk.78.ffn_down_exps.255.numa.1.weight"]
    _write_json(index_path, index)
    server_args = _server_args(str(model_path))
    server_args.kt_weight_path = str(weight_path)

    with pytest.raises(KTMTPAdmissionError, match="missing 1 required"):
        admit_glm52_kt_mtp(server_args, _glm52_config())
