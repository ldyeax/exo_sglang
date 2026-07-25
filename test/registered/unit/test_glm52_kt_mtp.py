from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang.srt.speculative.kt_mtp as kt_mtp_module
from sglang.srt.layers.moe.utils import (
    get_kt_ep_weight_layer_index,
    speculative_kt_ep_context,
)
from sglang.srt.speculative.kt_mtp import (
    KTMTPAdmission,
    KTMTPAdmissionError,
    admit_glm52_kt_mtp,
    get_glm52_kt_mtp_shared_modules,
    glm52_kt_mtp_shared_modules,
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
        speculative_draft_load_format="safetensors",
        load_format="safetensors",
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
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
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

    admission = admit_glm52_kt_mtp(_server_args(), config, validate_artifacts=False)

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
        admit_glm52_kt_mtp(server_args, _glm52_config(), validate_artifacts=False)


def test_glm52_candidate_requires_same_checkpoint_for_target_and_draft():
    server_args = _server_args()
    server_args.speculative_draft_model_path = "/models/other"

    with pytest.raises(KTMTPAdmissionError, match="must resolve"):
        admit_glm52_kt_mtp(server_args, _glm52_config(), validate_artifacts=False)


def test_speculative_context_maps_only_local_layer_zero_and_restores():
    assert get_kt_ep_weight_layer_index(7) == 7

    with speculative_kt_ep_context(enabled=True, physical_layer_index=78):
        assert get_kt_ep_weight_layer_index(0) == 78
        with pytest.raises(RuntimeError, match="local layer zero"):
            get_kt_ep_weight_layer_index(1)

    assert get_kt_ep_weight_layer_index(7) == 7


def test_shared_module_context_exposes_exact_target_modules_and_restores():
    embed_tokens = object()
    lm_head = object()
    assert get_glm52_kt_mtp_shared_modules() is None

    with glm52_kt_mtp_shared_modules(embed_tokens, lm_head) as shared:
        assert shared.embed_tokens is embed_tokens
        assert shared.lm_head is lm_head
        assert get_glm52_kt_mtp_shared_modules() is shared

    assert get_glm52_kt_mtp_shared_modules() is None


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


def _loaded_draft(
    *,
    layer_index: int = 78,
    gpu_expert_count: int = 0,
    shared_at_construction: bool = True,
    compact_mla_kv_b_w8: bool = False,
):
    kt_config = SimpleNamespace(
        layer_idx=layer_index,
        gpu_experts_mask=_Mask(gpu_expert_count),
    )
    quant_method = SimpleNamespace(_quant_wrapper_id="kt_ep", kt_config=kt_config)
    experts = SimpleNamespace(quant_method=quant_method)
    mlp = SimpleNamespace(experts=experts)
    decoder = SimpleNamespace(mlp=mlp)
    if compact_mla_kv_b_w8:
        decoder.self_attn = SimpleNamespace(
            kv_b_proj=SimpleNamespace(
                quant_method=SimpleNamespace(is_mla_kv_b_w8=True),
                kc_qweight=object(),
                kc_scales=object(),
                vc_qweight=object(),
                vc_scales=object(),
            )
        )
    return SimpleNamespace(
        model=SimpleNamespace(decoder=decoder),
        kt_mtp_shared_embed_and_head_at_construction=shared_at_construction,
    )


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
        "shared_embed_and_head_at_construction": True,
    }


def test_loaded_draft_receipt_rejects_local_layer_zero_mapping():
    with pytest.raises(KTMTPAdmissionError, match="physical_layer_index=0"):
        validate_loaded_glm52_kt_mtp(
            _loaded_draft(layer_index=0),
            KTMTPAdmission(enabled=True, physical_layer_index=78),
        )


def test_loaded_draft_receipt_requires_construction_time_weight_sharing():
    with pytest.raises(KTMTPAdmissionError, match="not shared at draft construction"):
        validate_loaded_glm52_kt_mtp(
            _loaded_draft(shared_at_construction=False),
            KTMTPAdmission(enabled=True, physical_layer_index=78),
        )


def test_loaded_draft_receipt_proves_direct_compact_kv_b_consumption():
    receipt = validate_loaded_glm52_kt_mtp(
        _loaded_draft(compact_mla_kv_b_w8=True),
        KTMTPAdmission(
            enabled=True,
            physical_layer_index=78,
            compact_mla_kv_b_w8=True,
        ),
    )

    assert receipt["compact_mla_kv_b_w8"] is True


def test_draft_weight_filter_excludes_bf16_routed_experts_before_loading():
    weight_map = {
        "model.layers.77.self_attn.q_proj.weight": "target.safetensors",
        "model.layers.78.self_attn.q_proj.weight": "mtp-header.safetensors",
        "model.layers.78.mlp.gate.weight": "mtp-tail.safetensors",
        "model.layers.78.mlp.experts.0.gate_proj.weight": "mtp-expert.safetensors",
        "model.layers.78.mlp.experts.255.down_proj.weight": "mtp-expert.safetensors",
        "model.layers.78.self_attn.kv_b_proj.kc_qweight": "mtp-header.safetensors",
        "model.layers.78.self_attn.kv_b_proj.kc_scales": "mtp-header.safetensors",
        "model.layers.78.self_attn.kv_b_proj.vc_qweight": "mtp-header.safetensors",
        "model.layers.78.self_attn.kv_b_proj.vc_scales": "mtp-header.safetensors",
    }

    names, shards = select_glm52_mtp_nonexpert_weights(weight_map, 78)

    assert names == {
        "model.layers.78.self_attn.q_proj.weight",
        "model.layers.78.mlp.gate.weight",
        "model.layers.78.self_attn.kv_b_proj.kc_qweight",
        "model.layers.78.self_attn.kv_b_proj.kc_scales",
        "model.layers.78.self_attn.kv_b_proj.vc_qweight",
        "model.layers.78.self_attn.kv_b_proj.vc_scales",
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
    _write_json(model_path / "config.json", _config_json())

    kt_names = []
    for projection in ("gate", "up", "down"):
        for expert_index in range(256):
            for numa_slot in range(2):
                stem = (
                    f"blk.78.ffn_{projection}_exps.{expert_index}." f"numa.{numa_slot}"
                )
                kt_names.extend((stem + ".weight", stem + ".scale"))
    _write_json(weight_path / "config.json", _config_json())
    _write_json(
        weight_path / "model.safetensors.index.json",
        {"weight_map": {name: weight_shard for name in kt_names}},
    )
    return model_path, weight_path


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_compact_safetensors(
    path: Path,
    prefix: str,
    *,
    overrides: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> None:
    tensor_abi = dict(kt_mtp_module._COMPACT_KV_B_ABI)
    tensor_abi.update(overrides or {})
    dtype_bytes = {"BF16": 2, "I32": 4}
    cursor = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for suffix, (dtype, shape) in tensor_abi.items():
        extent = dtype_bytes[dtype]
        for dimension in shape:
            extent *= dimension
        header[prefix + suffix] = {
            "data_offsets": [cursor, cursor + extent],
            "dtype": dtype,
            "shape": list(shape),
        }
        cursor += extent
    raw_header = _canonical_json_bytes(header)
    raw_header += b" " * ((-len(raw_header)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        stream.truncate(8 + len(raw_header) + cursor)


def _file_row(
    path: Path, root: Path, *, hash_contents: bool = True
) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path) if hash_contents else "0" * 64,
        "size_bytes": path.stat().st_size,
    }


def _make_hybrid_artifacts(
    root: Path,
    *,
    compact_overrides: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> tuple[Path, Path, Path, str]:
    model_path = root / "hybrid-model"
    weight_path = root / "hybrid-weights"
    model_path.mkdir()
    weight_path.mkdir()

    quantization_config = {
        **kt_mtp_module._HYBRID_QUANTIZATION_FIELDS,
        "dynamic": {
            r"-:.*\.mlp\.experts$": {},
            r"-:.*\.self_attn\.indexer\.weights_proj$": {},
        },
        "exo_mla_kv_b_w8": dict(kt_mtp_module._MLA_KV_B_W8_FORMAT),
    }
    model_config = _config_json()
    model_config["quantization_config"] = quantization_config
    _write_json(model_path / "config.json", model_config)
    _write_json(model_path / "quantize_config.json", quantization_config)

    prefix = "model.layers.78."
    source_names = [prefix + suffix for suffix in kt_mtp_module._HYBRID_LAYER_SUFFIXES]
    model_shard = "model-00001-of-00001.safetensors"
    _write_compact_safetensors(
        model_path / model_shard,
        prefix + "self_attn.kv_b_proj.",
        overrides=compact_overrides,
    )
    _write_json(
        model_path / "model.safetensors.index.json",
        {"weight_map": {name: model_shard for name in source_names}},
    )

    weight_shard = "model-00001-of-00001.safetensors"
    (weight_path / weight_shard).touch()
    kt_names = []
    for projection in ("gate", "up", "down"):
        for expert_index in range(256):
            for numa_slot in range(2):
                stem = f"blk.78.ffn_{projection}_exps.{expert_index}.numa.{numa_slot}"
                kt_names.extend((stem + ".weight", stem + ".scale"))
    _write_json(weight_path / "config.json", _config_json())
    _write_json(
        weight_path / "model.safetensors.index.json",
        {"weight_map": {name: weight_shard for name in kt_names}},
    )

    shared_files = sorted(
        (
            _file_row(weight_path / "config.json", weight_path),
            _file_row(
                weight_path / "model.safetensors.index.json",
                weight_path,
            ),
            _file_row(
                weight_path / weight_shard,
                weight_path,
                hash_contents=False,
            ),
        ),
        key=lambda row: str(row["path"]),
    )
    shared_content_contract = {
        "files": shared_files,
        "kind": "kt_shared_host_weights_content",
        "schema_version": 1,
    }
    shared_content_id = hashlib.sha256(
        _canonical_json_bytes(shared_content_contract)
    ).hexdigest()
    shared_manifest = root / "shared-host-weights-manifest.json"
    _write_json(
        shared_manifest,
        {
            "content_id": shared_content_id,
            "files": shared_files,
            "kind": "kt_shared_host_weights_manifest",
            "numa_nodes": [0, 1],
            "schema_version": 1,
        },
    )

    target_files = sorted(
        (
            _file_row(model_path / "config.json", model_path),
            _file_row(
                model_path / "model.safetensors.index.json",
                model_path,
            ),
            _file_row(model_path / "quantize_config.json", model_path),
            _file_row(
                model_path / model_shard,
                model_path,
                hash_contents=False,
            ),
        ),
        key=lambda row: str(row["path"]),
    )
    content_contract = {
        "expert_checkpoint": {
            "content_id": shared_content_id,
            "manifest_path": str(shared_manifest),
            "manifest_sha256": _sha256_file(shared_manifest),
            "method": "AMXINT4",
            "weight_path": str(weight_path),
        },
        "files": target_files,
        "kind": "glm52_amxint4_ampere_w8a16_hybrid_checkpoint",
        "policy": {
            "omit_expert": {"source_tensor_count": 58_368},
            "omitted_expert_names_sha256": (
                "dc2323c334d9af5eebbce9dd266059d4b9927945eeeff1b57cae9139f30ae1cc"
            ),
        },
        "schema_version": 1,
    }
    hybrid_manifest = {
        **content_contract,
        "content_id": hashlib.sha256(
            _canonical_json_bytes(content_contract)
        ).hexdigest(),
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "immutability": {"materialized_offline": True},
        "tensors": [],
    }
    _write_json(
        model_path / "hybrid-checkpoint-manifest.json",
        hybrid_manifest,
    )

    for directory in (model_path, weight_path):
        for path in directory.iterdir():
            path.chmod(0o444)
    shared_manifest.chmod(0o444)
    return model_path, weight_path, shared_manifest, shared_content_id


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


def _clear_shared_weight_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KT_SHARED_HOST_WEIGHTS",
        "KT_SHARED_HOST_WEIGHTS_MANIFEST",
        "KT_SHARED_HOST_WEIGHTS_CONTENT_ID",
        "KT_SHARED_HOST_WEIGHTS_STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def _hybrid_server_args(model_path: Path, weight_path: Path) -> SimpleNamespace:
    server_args = _server_args(str(model_path))
    server_args.kt_weight_path = str(weight_path)
    return server_args


def _rewrite_hybrid_manifest(
    model_path: Path,
    mutate,
) -> None:
    manifest_path = model_path / "hybrid-checkpoint-manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    content_contract = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_id", "created_at_utc", "immutability", "tensors"}
    }
    manifest["content_id"] = hashlib.sha256(
        _canonical_json_bytes(content_contract)
    ).hexdigest()
    _write_json(manifest_path, manifest)
    manifest_path.chmod(0o444)


def test_hybrid_admission_proves_compact_kv_b_and_external_mtp_experts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)

    admission = admit_glm52_kt_mtp(
        _hybrid_server_args(model_path, weight_path),
        _glm52_config(),
    )

    assert admission.enabled
    assert admission.compact_mla_kv_b_w8
    assert "direct compact MLA kv_b W8" in admission.reason


@pytest.mark.parametrize(
    "unexpected_name",
    [
        "model.layers.78.self_attn.kv_b_proj.weight",
        "model.layers.78.mlp.experts.0.gate_proj.weight",
    ],
)
def test_hybrid_admission_rejects_ambiguous_target_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unexpected_name: str,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    index_path = model_path / "model.safetensors.index.json"
    index_path.chmod(0o644)
    index = json.loads(index_path.read_text())
    index["weight_map"][unexpected_name] = "model-00001-of-00001.safetensors"
    _write_json(index_path, index)

    with pytest.raises(
        KTMTPAdmissionError,
        match="ambiguously|must omit routed MTP experts",
    ):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_admission_requires_all_four_compact_kv_b_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    index_path = model_path / "model.safetensors.index.json"
    index_path.chmod(0o644)
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.layers.78.self_attn.kv_b_proj.vc_scales"]
    _write_json(index_path, index)

    with pytest.raises(KTMTPAdmissionError, match="35-tensor contract"):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_manifest_cannot_fall_back_without_exact_exo_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    config_path = model_path / "config.json"
    config_path.chmod(0o644)
    config = json.loads(config_path.read_text())
    del config["quantization_config"]["exo_mla_kv_b_w8"]
    _write_json(config_path, config)

    with pytest.raises(
        KTMTPAdmissionError,
        match="manifest is present but exo_mla_kv_b_w8 metadata is absent",
    ):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_admission_rejects_wrong_compact_tensor_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(
        tmp_path,
        compact_overrides={"kc_qweight": ("I32", (64, 47, 512))},
    )

    with pytest.raises(KTMTPAdmissionError, match="tensor ABI mismatch"):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_admission_rejects_non_string_compact_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    index_path = model_path / "model.safetensors.index.json"
    index_path.chmod(0o644)
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.layers.78.self_attn.kv_b_proj.kc_qweight"] = 1
    _write_json(index_path, index)

    with pytest.raises(KTMTPAdmissionError, match="non-string shard"):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_admission_binds_declared_expert_weight_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    other_weight_path = tmp_path / "other-weights"
    other_weight_path.mkdir()

    def mutate(manifest):
        manifest["expert_checkpoint"]["weight_path"] = str(other_weight_path)

    _rewrite_hybrid_manifest(model_path, mutate)

    with pytest.raises(KTMTPAdmissionError, match="weight_path differs"):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_admission_requires_exact_shared_manifest_numa_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, shared_manifest, _ = _make_hybrid_artifacts(tmp_path)
    shared_manifest.chmod(0o644)
    shared = json.loads(shared_manifest.read_text())
    shared["numa_nodes"] = [1, 0]
    _write_json(shared_manifest, shared)
    shared_manifest.chmod(0o444)

    def mutate(manifest):
        manifest["expert_checkpoint"]["manifest_sha256"] = _sha256_file(shared_manifest)

    _rewrite_hybrid_manifest(model_path, mutate)

    with pytest.raises(KTMTPAdmissionError, match="NUMA order"):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_hybrid_admission_binds_enabled_shared_weight_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_path, weight_path, _, content_id = _make_hybrid_artifacts(tmp_path)
    monkeypatch.setenv("KT_SHARED_HOST_WEIGHTS", "1")
    monkeypatch.setenv(
        "KT_SHARED_HOST_WEIGHTS_MANIFEST",
        str(tmp_path / "wrong-manifest.json"),
    )
    monkeypatch.setenv("KT_SHARED_HOST_WEIGHTS_CONTENT_ID", content_id)
    monkeypatch.setenv(
        "KT_SHARED_HOST_WEIGHTS_STATE_DIR",
        str(tmp_path / "state"),
    )

    with pytest.raises(
        KTMTPAdmissionError,
        match="MANIFEST differs",
    ):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )
