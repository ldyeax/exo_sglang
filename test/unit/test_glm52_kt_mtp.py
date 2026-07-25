from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import sglang.srt.speculative.kt_mtp as kt_mtp_module
import torch
from sglang.srt.layers.moe.utils import (
    get_kt_ep_weight_layer_index,
    speculative_kt_ep_context,
)
from sglang.srt.speculative.kt_mtp import (
    KTMTPAdmission,
    KTMTPAdmissionError,
    admit_glm52_kt_mtp,
    admit_glm52_kt_remote_draft,
    get_glm52_kt_mtp_shared_modules,
    glm52_kt_mtp_shared_modules,
    is_glm52_kt_mtp_shared_module,
    select_glm52_mtp_nonexpert_weights,
    select_glm52_remote_mtp_weights,
    validate_loaded_glm52_kt_mtp,
)

_REMOTE_DRAFT_HEADER_SHARD = "model-00001-of-00005.safetensors"
_REMOTE_DRAFT_LAYER_SHARD = "model-00005-of-00005.safetensors"
_REMOTE_DRAFT_WEIGHT_NAMES = (
    "model.embed_tokens.weight",
    "lm_head.qweight",
    "lm_head.scales",
    "model.layers.78.eh_proj.weight",
    "model.layers.78.enorm.weight",
    "model.layers.78.hnorm.weight",
    "model.layers.78.input_layernorm.weight",
    "model.layers.78.mlp.gate.e_score_correction_bias",
    "model.layers.78.mlp.gate.weight",
    "model.layers.78.mlp.shared_experts.down_proj.qweight",
    "model.layers.78.mlp.shared_experts.down_proj.scales",
    "model.layers.78.mlp.shared_experts.gate_proj.qweight",
    "model.layers.78.mlp.shared_experts.gate_proj.scales",
    "model.layers.78.mlp.shared_experts.up_proj.qweight",
    "model.layers.78.mlp.shared_experts.up_proj.scales",
    "model.layers.78.post_attention_layernorm.weight",
    "model.layers.78.self_attn.indexer.k_norm.bias",
    "model.layers.78.self_attn.indexer.k_norm.weight",
    "model.layers.78.self_attn.indexer.weights_proj.weight",
    "model.layers.78.self_attn.indexer.wk.qweight",
    "model.layers.78.self_attn.indexer.wk.scales",
    "model.layers.78.self_attn.indexer.wq_b.qweight",
    "model.layers.78.self_attn.indexer.wq_b.scales",
    "model.layers.78.self_attn.kv_a_layernorm.weight",
    "model.layers.78.self_attn.kv_a_proj_with_mqa.qweight",
    "model.layers.78.self_attn.kv_a_proj_with_mqa.scales",
    "model.layers.78.self_attn.kv_b_proj.kc_qweight",
    "model.layers.78.self_attn.kv_b_proj.kc_scales",
    "model.layers.78.self_attn.kv_b_proj.vc_qweight",
    "model.layers.78.self_attn.kv_b_proj.vc_scales",
    "model.layers.78.self_attn.o_proj.qweight",
    "model.layers.78.self_attn.o_proj.scales",
    "model.layers.78.self_attn.q_a_layernorm.weight",
    "model.layers.78.self_attn.q_a_proj.qweight",
    "model.layers.78.self_attn.q_a_proj.scales",
    "model.layers.78.self_attn.q_b_proj.qweight",
    "model.layers.78.self_attn.q_b_proj.scales",
    "model.layers.78.shared_head.norm.weight",
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
        record_kt_gpu_expert_distribution=False,
        init_expert_location="trivial",
        chunked_prefill_size=8192,
        kt_expert_placement_strategy="uniform",
        kt_lora_path=None,
        speculative_token_map=None,
    )


def _remote_server_args(model_path: str = "/models/glm52") -> SimpleNamespace:
    server_args = _server_args(model_path)
    server_args.tp_size = 1
    server_args.kt_cpuinfer = 60
    server_args.kt_threadpool_count = 2
    server_args.kt_numa_nodes = [0, 0]
    server_args.kt_stream_prefill = False
    return server_args


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
        ("load_format", None),
        ("speculative_draft_load_format", None),
        ("kt_threadpool_count", True),
        ("kt_numa_nodes", [0, -1]),
        ("speculative_token_map", "/models/hot-token-map.json"),
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


def test_remote_admission_binds_two_logical_slots_to_fwuff_numa_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    monkeypatch.setenv("KT_SHARED_HOST_WEIGHTS", "0")

    admission = admit_glm52_kt_remote_draft(
        _remote_server_args(),
        _glm52_config(),
        validate_artifacts=False,
    )

    assert admission.enabled
    assert admission.physical_layer_index == 78
    assert admission.logical_amx_slots == (0, 1)
    assert admission.physical_numa_nodes == (0, 0)
    assert admission.cpuinfer_threads == 60
    assert admission.threadpool_count == 2
    assert admission.standalone_tensor_count == 38
    assert admission.standalone_shards == (
        _REMOTE_DRAFT_HEADER_SHARD,
        _REMOTE_DRAFT_LAYER_SHARD,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tp_size", 2),
        ("tp_size", True),
        ("kt_cpuinfer", 120),
        ("kt_cpuinfer", 60.0),
        ("kt_threadpool_count", 1),
        ("kt_threadpool_count", 2.0),
        ("kt_numa_nodes", [0, 1]),
        ("kt_numa_nodes", [False, 0]),
        ("kt_stream_prefill", True),
    ],
)
def test_remote_admission_rejects_non_fwuff_runtime_values(
    field: str,
    value,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    server_args = _remote_server_args()
    setattr(server_args, field, value)

    with pytest.raises(KTMTPAdmissionError, match=field):
        admit_glm52_kt_remote_draft(
            server_args,
            _glm52_config(),
            validate_artifacts=False,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KT_SHARED_HOST_WEIGHTS", "1"),
        ("KT_SHARED_HOST_WEIGHTS", ""),
        ("KT_SHARED_HOST_WEIGHTS_MANIFEST", "/tmp/dormant-manifest.json"),
        ("KT_SHARED_HOST_WEIGHTS_CONTENT_ID", "0" * 64),
        ("KT_SHARED_HOST_WEIGHTS_STATE_DIR", "/tmp/dormant-state"),
    ],
)
def test_remote_admission_requires_shared_host_weights_strictly_off(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(KTMTPAdmissionError, match="shared|SHARED"):
        admit_glm52_kt_remote_draft(
            _remote_server_args(),
            _glm52_config(),
            validate_artifacts=False,
        )


def test_remote_runtime_remains_disallowed_by_local_tp2_admission(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)

    with pytest.raises(KTMTPAdmissionError, match="tp_size"):
        admit_glm52_kt_mtp(
            _remote_server_args(),
            _glm52_config(),
            validate_artifacts=False,
        )


def test_speculative_context_maps_only_local_layer_zero_and_restores():
    assert get_kt_ep_weight_layer_index(7) == 7

    with speculative_kt_ep_context(enabled=True, physical_layer_index=78):
        assert get_kt_ep_weight_layer_index(0) == 78
        with pytest.raises(RuntimeError, match="local layer zero"):
            get_kt_ep_weight_layer_index(1)

    assert get_kt_ep_weight_layer_index(7) == 7


def test_shared_module_context_exposes_exact_target_modules_and_restores():
    embed_tokens = torch.nn.Module()
    lm_head = torch.nn.Sequential(torch.nn.Module())
    lm_head_child = next(lm_head.children())
    unrelated = torch.nn.Module()
    assert get_glm52_kt_mtp_shared_modules() is None
    assert not is_glm52_kt_mtp_shared_module(embed_tokens)

    with glm52_kt_mtp_shared_modules(embed_tokens, lm_head) as shared:
        assert shared.embed_tokens is embed_tokens
        assert shared.lm_head is lm_head
        assert get_glm52_kt_mtp_shared_modules() is shared
        assert is_glm52_kt_mtp_shared_module(embed_tokens)
        assert is_glm52_kt_mtp_shared_module(lm_head)
        assert is_glm52_kt_mtp_shared_module(lm_head_child)
        assert not is_glm52_kt_mtp_shared_module(unrelated)

    assert get_glm52_kt_mtp_shared_modules() is None
    assert not is_glm52_kt_mtp_shared_module(lm_head)


class _CountingQuantMethod:
    def __init__(self) -> None:
        self.calls = 0

    def process_weights_after_loading(self, module: torch.nn.Module) -> None:
        del module
        self.calls += 1


class _QuantizedLeaf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant_method = _CountingQuantMethod()


class _LoaderModel(torch.nn.Module):
    def __init__(
        self,
        borrowed_head: torch.nn.Module,
        owned_leaf: _QuantizedLeaf,
    ) -> None:
        super().__init__()
        self.borrowed_head = borrowed_head
        self.owned_leaf = owned_leaf
        self.load_calls = 0

    def load_weights(self, weights) -> None:
        tuple(weights)
        self.load_calls += 1


def test_default_loader_skips_only_context_borrowed_module_subtrees():
    if not torch.cuda.is_available():
        pytest.skip("SGLang model-loader imports require a visible CUDA device")
    from sglang.srt.model_loader.loader import DefaultModelLoader

    embed_tokens = torch.nn.Module()
    borrowed_leaf = _QuantizedLeaf()
    borrowed_head = torch.nn.Sequential(borrowed_leaf)
    owned_leaf = _QuantizedLeaf()
    model = _LoaderModel(borrowed_head, owned_leaf)

    with glm52_kt_mtp_shared_modules(embed_tokens, borrowed_head) as shared:
        model.kt_mtp_borrowed_module_ids_at_construction = shared.borrowed_module_ids
        DefaultModelLoader.load_weights_and_postprocess(
            model,
            (),
            torch.device("cpu"),
        )

    assert model.load_calls == 1
    assert borrowed_leaf.quant_method.calls == 0
    assert owned_leaf.quant_method.calls == 1

    DefaultModelLoader.load_weights_and_postprocess(
        model,
        (),
        torch.device("cpu"),
    )

    assert model.load_calls == 2
    assert borrowed_leaf.quant_method.calls == 0
    assert owned_leaf.quant_method.calls == 2

    del model.kt_mtp_borrowed_module_ids_at_construction
    DefaultModelLoader.load_weights_and_postprocess(
        model,
        (),
        torch.device("cpu"),
    )

    assert model.load_calls == 3
    assert borrowed_leaf.quant_method.calls == 1
    assert owned_leaf.quant_method.calls == 3


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
    embed_tokens: object,
    lm_head: object,
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
        model=SimpleNamespace(decoder=decoder, embed_tokens=embed_tokens),
        lm_head=lm_head,
        kt_mtp_shared_embed_and_head_at_construction=shared_at_construction,
        kt_mtp_borrowed_module_ids_at_construction=frozenset(
            (id(embed_tokens), id(lm_head))
        ),
    )


def _loaded_models(**kwargs):
    embed_tokens = object()
    lm_head = object()
    draft_model = _loaded_draft(
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        **kwargs,
    )
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=embed_tokens),
        lm_head=lm_head,
    )
    return draft_model, target_model


def test_loaded_draft_receipt_proves_physical_layer_and_cpu_ownership():
    draft_model, target_model = _loaded_models()
    receipt = validate_loaded_glm52_kt_mtp(
        draft_model,
        KTMTPAdmission(enabled=True, physical_layer_index=78),
        target_model=target_model,
    )

    assert receipt == {
        "enabled": True,
        "physical_layer_index": 78,
        "gpu_expert_count": 0,
        "wrapper_id": "kt_ep",
        "shared_embed_and_head_at_construction": True,
    }


def test_loaded_draft_receipt_rejects_local_layer_zero_mapping():
    draft_model, target_model = _loaded_models(layer_index=0)
    with pytest.raises(KTMTPAdmissionError, match="physical_layer_index=0"):
        validate_loaded_glm52_kt_mtp(
            draft_model,
            KTMTPAdmission(enabled=True, physical_layer_index=78),
            target_model=target_model,
        )


def test_loaded_draft_receipt_requires_construction_time_weight_sharing():
    draft_model, target_model = _loaded_models(shared_at_construction=False)
    with pytest.raises(KTMTPAdmissionError, match="not shared at draft construction"):
        validate_loaded_glm52_kt_mtp(
            draft_model,
            KTMTPAdmission(enabled=True, physical_layer_index=78),
            target_model=target_model,
        )


def test_loaded_draft_receipt_requires_exact_target_module_identities():
    draft_model, target_model = _loaded_models()
    draft_model.lm_head = object()

    with pytest.raises(KTMTPAdmissionError, match="not the target LM head module"):
        validate_loaded_glm52_kt_mtp(
            draft_model,
            KTMTPAdmission(enabled=True, physical_layer_index=78),
            target_model=target_model,
        )


def test_loaded_draft_receipt_proves_direct_compact_kv_b_consumption():
    draft_model, target_model = _loaded_models(compact_mla_kv_b_w8=True)
    receipt = validate_loaded_glm52_kt_mtp(
        draft_model,
        KTMTPAdmission(
            enabled=True,
            physical_layer_index=78,
            compact_mla_kv_b_w8=True,
        ),
        target_model=target_model,
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


def _remote_weight_map() -> dict[str, object]:
    weight_map: dict[str, object] = {}
    for name in _REMOTE_DRAFT_WEIGHT_NAMES:
        weight_map[name] = (
            _REMOTE_DRAFT_HEADER_SHARD
            if not name.startswith("model.layers.78.")
            else _REMOTE_DRAFT_LAYER_SHARD
        )
    weight_map["model.layers.77.input_layernorm.weight"] = (
        "model-00004-of-00005.safetensors"
    )
    weight_map["model.layers.78.mlp.experts.0.gate_proj.weight"] = (
        _REMOTE_DRAFT_LAYER_SHARD
    )
    return weight_map


def test_remote_weight_filter_selects_exact_standalone_38_tensor_contract():
    names, shards = select_glm52_remote_mtp_weights(_remote_weight_map(), 78)

    assert names == set(_REMOTE_DRAFT_WEIGHT_NAMES)
    assert len(names) == 38
    assert shards == {
        _REMOTE_DRAFT_HEADER_SHARD,
        _REMOTE_DRAFT_LAYER_SHARD,
    }
    assert not any(".mlp.experts." in name for name in names)


def test_remote_weight_filter_rejects_missing_tensor():
    weight_map = _remote_weight_map()
    del weight_map["lm_head.scales"]

    with pytest.raises(KTMTPAdmissionError, match="missing 1 of 38"):
        select_glm52_remote_mtp_weights(weight_map, 78)


def test_remote_weight_filter_rejects_non_string_shard():
    weight_map = _remote_weight_map()
    weight_map["model.embed_tokens.weight"] = 1

    with pytest.raises(KTMTPAdmissionError, match="non-string shard"):
        select_glm52_remote_mtp_weights(weight_map, 78)


def test_remote_weight_filter_rejects_wrong_exact_shard():
    weight_map = _remote_weight_map()
    weight_map["lm_head.qweight"] = _REMOTE_DRAFT_LAYER_SHARD

    with pytest.raises(KTMTPAdmissionError, match="unexpected shard"):
        select_glm52_remote_mtp_weights(weight_map, 78)


def test_remote_weight_filter_rejects_non_layer_78_request():
    with pytest.raises(KTMTPAdmissionError, match="physical layer 78"):
        select_glm52_remote_mtp_weights(_remote_weight_map(), 77)


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
                stem = f"blk.78.ffn_{projection}_exps.{expert_index}.numa.{numa_slot}"
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
    remote_standalone: bool = False,
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
    model_shard = (
        _REMOTE_DRAFT_LAYER_SHARD
        if remote_standalone
        else "model-00001-of-00001.safetensors"
    )
    _write_compact_safetensors(
        model_path / model_shard,
        prefix + "self_attn.kv_b_proj.",
        overrides=compact_overrides,
    )
    source_weight_map = {name: model_shard for name in source_names}
    if remote_standalone:
        (model_path / _REMOTE_DRAFT_HEADER_SHARD).touch()
        source_weight_map.update(
            {
                name: _REMOTE_DRAFT_HEADER_SHARD
                for name in _REMOTE_DRAFT_WEIGHT_NAMES[:3]
            }
        )
    _write_json(
        model_path / "model.safetensors.index.json",
        {"weight_map": source_weight_map},
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

    target_file_paths = [
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
        model_path / "quantize_config.json",
        model_path / model_shard,
    ]
    if remote_standalone:
        target_file_paths.append(model_path / _REMOTE_DRAFT_HEADER_SHARD)
    target_files = sorted(
        (
            _file_row(
                path,
                model_path,
                hash_contents=path.suffix != ".safetensors",
            )
            for path in target_file_paths
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
        "output": {
            "payload_bytes": 20_056_714_112,
            "shard_count": 5,
            "tensor_count": 2_052,
        },
        "policy": {
            "omit_expert": {"source_tensor_count": 58_368},
            "omitted_expert_names_sha256": (
                "dc2323c334d9af5eebbce9dd266059d4b9927945eeeff1b57cae9139f30ae1cc"
            ),
        },
        "quantization": {
            "activation_dtype": "BF16",
            "checkpoint_layout": {
                "mla_kv_b": "gptq_packed_rows_per_head_v1",
                "ordinary_linear": "gptq_packed_rows",
            },
            "mla_kv_b_compact_to_compact_marlin_repack_at_load": True,
            "mla_kv_b_triton_uses_serialized_words_directly": True,
            "ordinary_linear_compact_to_compact_marlin_repack_at_load": True,
            "serialized_scale_dtype": "BF16",
            "serialized_weight_dtype": ("INT8_biased_by_128_packed_in_INT32"),
            "temporary_bf16_expansion_at_load": False,
        },
        "schema_version": 1,
    }
    hybrid_manifest = {
        **content_contract,
        "content_id": hashlib.sha256(
            _canonical_json_bytes(content_contract)
        ).hexdigest(),
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "immutability": {
            "directory_mode": "0555",
            "file_mode": "0444",
            "materialized_offline": True,
        },
        "tensors": [],
    }
    _write_json(
        model_path / "hybrid-checkpoint-manifest.json",
        hybrid_manifest,
    )

    for directory in (model_path, weight_path):
        for path in directory.iterdir():
            path.chmod(0o444)
    model_path.chmod(0o555)
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


def test_remote_artifact_admission_separates_logical_slots_from_physical_numa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(
        tmp_path,
        remote_standalone=True,
    )
    server_args = _remote_server_args(str(model_path))
    server_args.kt_weight_path = str(weight_path)

    admission = admit_glm52_kt_remote_draft(
        server_args,
        _glm52_config(),
    )

    assert admission.enabled
    assert admission.compact_mla_kv_b_w8
    assert admission.logical_amx_slots == (0, 1)
    assert admission.physical_numa_nodes == (0, 0)
    assert admission.standalone_tensor_count == 38


def test_hybrid_admission_requires_target_shard_manifest_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)

    def mutate(manifest):
        manifest["files"] = [
            entry
            for entry in manifest["files"]
            if entry["path"] != "model-00001-of-00001.safetensors"
        ]

    _rewrite_hybrid_manifest(model_path, mutate)

    with pytest.raises(
        KTMTPAdmissionError,
        match="does not attest target layer-78 shards",
    ):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


@pytest.mark.parametrize("writable_target", ["root", "manifest"])
def test_hybrid_admission_requires_live_immutable_root_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writable_target: str,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    if writable_target == "root":
        model_path.chmod(0o755)
    else:
        (model_path / "hybrid-checkpoint-manifest.json").chmod(0o644)

    with pytest.raises(KTMTPAdmissionError, match="is not immutable"):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


def test_named_dangling_hybrid_manifest_cannot_fall_back_to_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path = _make_artifacts(tmp_path)
    (model_path / "hybrid-checkpoint-manifest.json").symlink_to(
        model_path / "missing-manifest.json"
    )

    with pytest.raises(
        KTMTPAdmissionError,
        match="Hybrid target manifest is present",
    ):
        admit_glm52_kt_mtp(
            _hybrid_server_args(model_path, weight_path),
            _glm52_config(),
        )


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


def test_hybrid_admission_rejects_nested_target_shard_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_weight_environment(monkeypatch)
    model_path, weight_path, _, _ = _make_hybrid_artifacts(tmp_path)
    index_path = model_path / "model.safetensors.index.json"
    index_path.chmod(0o644)
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.layers.78.self_attn.kv_b_proj.kc_qweight"] = (
        "nested/model.safetensors"
    )
    _write_json(index_path, index)

    with pytest.raises(KTMTPAdmissionError, match="unsafe shard paths"):
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
