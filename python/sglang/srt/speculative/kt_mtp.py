# SPDX-License-Identifier: Apache-2.0
"""Fail-closed admission for GLM-5.2 MTP on KTransformers.

The target model and its MTP draft layer live in the same Hugging Face
checkpoint.  SGLang numbers the one-layer draft model locally as layer zero,
while KTransformers' persistent expert artifact keeps the physical checkpoint
number (``blk.78``).  This module validates that exact contract before the
draft worker is allowed to construct a KT MoE wrapper.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

GLM52_ARCHITECTURE = "GlmMoeDsaForCausalLM"
GLM52_MODEL_TYPE = "glm_moe_dsa"


class KTMTPAdmissionError(ValueError):
    """Raised when a GLM-5.2 KT MTP request is not provably supported."""


@dataclass(frozen=True)
class KTMTPAdmission:
    enabled: bool = False
    physical_layer_index: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class KTMTPSharedModules:
    """Target modules reused while constructing the one-layer draft model."""

    embed_tokens: Any
    lm_head: Any


_ACTIVE_SHARED_MODULES: ContextVar[KTMTPSharedModules | None] = ContextVar(
    "glm52_kt_mtp_shared_modules",
    default=None,
)


@contextmanager
def glm52_kt_mtp_shared_modules(
    embed_tokens: Any,
    lm_head: Any,
) -> Iterator[KTMTPSharedModules]:
    """Expose target embed/head modules during draft construction.

    SGLang normally allocates a second vocabulary embedding and LM head and
    replaces their weights only after the draft model has loaded. GLM-5.2's
    TP2 BF16 vocabulary matrices are roughly 908 MiB each per rank, so that
    transient duplication exceeds the RTX 3090 headroom.
    """

    if embed_tokens is None or lm_head is None:
        raise KTMTPAdmissionError(
            "GLM-5.2 KT MTP shared modules must provide embedding and LM head"
        )
    shared = KTMTPSharedModules(embed_tokens=embed_tokens, lm_head=lm_head)
    token: Token[KTMTPSharedModules | None] = _ACTIVE_SHARED_MODULES.set(shared)
    try:
        yield shared
    finally:
        _ACTIVE_SHARED_MODULES.reset(token)


def get_glm52_kt_mtp_shared_modules() -> KTMTPSharedModules | None:
    """Return construction-scoped target modules, if the exact path is active."""

    return _ACTIVE_SHARED_MODULES.get()


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _architecture(config: Any) -> str | None:
    architectures = _get(config, "architectures", None)
    if not architectures:
        return None
    return architectures[0]


def _same_local_path(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    left_path = os.path.realpath(os.path.abspath(left))
    right_path = os.path.realpath(os.path.abspath(right))
    return left_path == right_path


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise KTMTPAdmissionError(
            f"Cannot read {description} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise KTMTPAdmissionError(f"{description} at {path} must be a JSON object")
    return value


def _validate_glm52_fingerprint(config: Any, *, description: str) -> None:
    expected = {
        "model_type": GLM52_MODEL_TYPE,
        "num_hidden_layers": 78,
        "num_nextn_predict_layers": 1,
        "hidden_size": 6144,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
    }
    actual_architecture = _architecture(config)
    if actual_architecture != GLM52_ARCHITECTURE:
        raise KTMTPAdmissionError(
            f"{description} architecture must be {GLM52_ARCHITECTURE}, "
            f"got {actual_architecture!r}"
        )
    mismatches = [
        f"{name}={_get(config, name)!r} (expected {expected_value!r})"
        for name, expected_value in expected.items()
        if _get(config, name) != expected_value
    ]
    if mismatches:
        raise KTMTPAdmissionError(
            f"{description} is not the admitted GLM-5.2 topology: "
            + ", ".join(mismatches)
        )
    if _get(config, "index_share_for_mtp_iteration") is not True:
        raise KTMTPAdmissionError(
            f"{description} must set index_share_for_mtp_iteration=true"
        )


def _validate_source_checkpoint(model_path: Path, physical_layer_index: int) -> None:
    index_path = model_path / "model.safetensors.index.json"
    index = _load_json(index_path, "target checkpoint index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise KTMTPAdmissionError(
            f"Target checkpoint index {index_path} has no object weight_map"
        )

    prefix = f"model.layers.{physical_layer_index}."
    required = (
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
    )
    _validate_index_entries(model_path, weight_map, required, "target checkpoint")


def _validate_kt_artifact(
    weight_path: Path,
    *,
    physical_layer_index: int,
    numa_slot_count: int,
) -> None:
    artifact_config = _load_json(weight_path / "config.json", "KT artifact config")
    _validate_glm52_fingerprint(artifact_config, description="KT artifact config")

    index_path = weight_path / "model.safetensors.index.json"
    index = _load_json(index_path, "KT artifact index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise KTMTPAdmissionError(
            f"KT artifact index {index_path} has no object weight_map"
        )

    required: list[str] = []
    for projection in ("gate", "up", "down"):
        for expert_index in range(256):
            for numa_slot in range(numa_slot_count):
                stem = (
                    f"blk.{physical_layer_index}.ffn_{projection}_exps."
                    f"{expert_index}.numa.{numa_slot}"
                )
                required.extend((stem + ".weight", stem + ".scale"))
    _validate_index_entries(weight_path, weight_map, required, "KT artifact")


def _validate_index_entries(
    root: Path,
    weight_map: Mapping[str, Any],
    required: tuple[str, ...] | list[str],
    description: str,
) -> None:
    missing = [name for name in required if name not in weight_map]
    if missing:
        preview = ", ".join(missing[:3])
        raise KTMTPAdmissionError(
            f"{description} is missing {len(missing)} required layer-78 "
            f"entries; first missing: {preview}"
        )
    missing_shards = sorted(
        {
            str(shard)
            for name in required
            if isinstance((shard := weight_map[name]), str)
            and not (root / shard).is_file()
        }
    )
    if missing_shards:
        raise KTMTPAdmissionError(
            f"{description} references missing shard files: "
            + ", ".join(missing_shards[:3])
        )


def select_glm52_mtp_nonexpert_weights(
    weight_map: Mapping[str, Any], physical_layer_index: int
) -> tuple[set[str], set[str]]:
    """Select MTP tensors that remain on GPU and their checkpoint shards."""
    if physical_layer_index != 78:
        raise KTMTPAdmissionError(
            "GLM-5.2 MTP weight filtering only supports physical layer 78"
        )
    prefix = f"model.layers.{physical_layer_index}."
    names = {
        name
        for name in weight_map
        if name.startswith(prefix) and ".mlp.experts." not in name
    }
    if not names:
        raise KTMTPAdmissionError(f"No non-routed MTP tensors with prefix {prefix!r}")
    invalid_shards = [name for name in names if not isinstance(weight_map[name], str)]
    if invalid_shards:
        raise KTMTPAdmissionError(
            f"MTP weight map has non-string shard values for {invalid_shards[:3]}"
        )
    shards = {weight_map[name] for name in names}
    return names, shards


def admit_glm52_kt_mtp(
    server_args: Any,
    target_hf_config: Any,
    *,
    validate_artifacts: bool = True,
) -> KTMTPAdmission:
    """Return an enabled plan only for the exact supported GLM-5.2 KT path.

    Non-GLM draft models retain SGLang's existing GPU-only behavior.  A
    GLM-5.2 request with KT configured is treated as an explicit request for
    this path, so every unsupported combination raises instead of silently
    loading the 13.5-GiB BF16 routed-expert draft layer onto the GPUs.
    """

    if _get(server_args, "kt_weight_path") is None:
        return KTMTPAdmission(reason="KTransformers is not configured")
    if _architecture(target_hf_config) != GLM52_ARCHITECTURE:
        return KTMTPAdmission(reason="draft target is not GLM-5.2 DSA")

    _validate_glm52_fingerprint(target_hf_config, description="target config")

    required_values = {
        "speculative_algorithm": "EAGLE",
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 1,
        "moe_a2a_backend": "none",
        "speculative_moe_a2a_backend": "none",
        "kt_method": "AMXINT4",
        "kt_num_gpu_experts": 0,
        "kt_max_deferred_experts_per_token": 0,
        "disable_shared_experts_fusion": True,
        "disable_cuda_graph": True,
        "enable_dp_attention": False,
        "enable_eplb": False,
        "enable_two_batch_overlap": False,
        "enable_single_batch_overlap": False,
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 2,
    }
    mismatches = [
        f"{name}={_get(server_args, name)!r} (expected {expected!r})"
        for name, expected in required_values.items()
        if _get(server_args, name) != expected
    ]

    optional_disabled = {
        "kt_gpu_experts_ratio": (None,),
        "kt_gpu_prefill_token_threshold": (None, 0),
        "kt_expert_lora_path": (None,),
        "speculative_draft_model_quantization": (None,),
        "init_expert_location": (None, "trivial"),
    }
    for name, admitted_values in optional_disabled.items():
        actual = _get(server_args, name)
        if actual not in admitted_values:
            mismatches.append(
                f"{name}={actual!r} (expected one of {admitted_values!r})"
            )

    incompatible_environment = (
        "SGLANG_KT_REMOTE_EXPERT_ENDPOINT",
        "SGLANG_KT_REMOTE_EXPERT_PLAN",
        "SGLANG_KT_REMOTE_EXPERT_ENDPOINTS",
        "SGLANG_KT_REMOTE_EXPERT_PLANS",
        "SGLANG_KT_CPU_EXPERT_SHARD_PLAN",
        "SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN",
        "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
        "SGLANG_KT_GPU_EXPERT_MASK_PLAN",
        "SGLANG_KT_EXPERT_PROFILE",
        "SGLANG_KT_EXPERT_LORA_PATH",
    )
    configured_environment = [
        name for name in incompatible_environment if os.environ.get(name)
    ]
    if configured_environment:
        mismatches.append(
            "unsupported KT placement/offload environment is set: "
            + ", ".join(configured_environment)
        )
    if os.environ.get("SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU", "0") != "0":
        mismatches.append("SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU must be disabled")
    if int(os.environ.get("SGLANG_KT_DRAFT_GPU_EXPERTS", "0")) != 0:
        mismatches.append("SGLANG_KT_DRAFT_GPU_EXPERTS must be 0")
    if int(os.environ.get("SGLANG_KT_GPU_PREFILL_TOKEN_THRESHOLD", "0")) != 0:
        mismatches.append("SGLANG_KT_GPU_PREFILL_TOKEN_THRESHOLD must be 0")
    if os.environ.get("SGLANG_KT_HOTSPOT_EXPERT_CACHE", "0") != "0":
        mismatches.append("SGLANG_KT_HOTSPOT_EXPERT_CACHE must be disabled")

    model_path = _get(server_args, "model_path")
    draft_model_path = _get(server_args, "speculative_draft_model_path")
    if not _same_local_path(model_path, draft_model_path):
        mismatches.append(
            "speculative_draft_model_path must resolve to the target model_path"
        )

    numa_nodes = _get(server_args, "kt_numa_nodes")
    threadpool_count = _get(server_args, "kt_threadpool_count")
    if not isinstance(threadpool_count, int) or threadpool_count <= 0:
        mismatches.append("kt_threadpool_count must be a positive integer")
    if (
        not isinstance(numa_nodes, (list, tuple))
        or not isinstance(threadpool_count, int)
        or len(numa_nodes) != threadpool_count
        or len(set(numa_nodes)) != len(numa_nodes)
    ):
        mismatches.append(
            "kt_numa_nodes must contain one distinct node per KT thread pool"
        )

    if mismatches:
        raise KTMTPAdmissionError(
            "GLM-5.2 KT MTP admission rejected: " + "; ".join(mismatches)
        )

    physical_layer_index = int(_get(target_hf_config, "num_hidden_layers"))
    if validate_artifacts:
        assert isinstance(threadpool_count, int)
        _validate_source_checkpoint(Path(model_path), physical_layer_index)
        _validate_kt_artifact(
            Path(_get(server_args, "kt_weight_path")),
            physical_layer_index=physical_layer_index,
            numa_slot_count=threadpool_count,
        )

    return KTMTPAdmission(
        enabled=True,
        physical_layer_index=physical_layer_index,
        reason="admitted GLM-5.2 layer-78 NEXTN with persistent AMXINT4 experts",
    )


def validate_loaded_glm52_kt_mtp(
    draft_model: Any, admission: KTMTPAdmission
) -> dict[str, Any]:
    """Prove that the loaded one-layer draft actually received the KT wrapper."""
    if not admission.enabled or admission.physical_layer_index is None:
        return {"enabled": False}
    try:
        decoder = draft_model.model.decoder
        quant_method = decoder.mlp.experts.quant_method
        kt_config = quant_method.kt_config
    except AttributeError as error:
        raise KTMTPAdmissionError(
            "Admitted GLM-5.2 draft model did not expose "
            "model.decoder.mlp.experts.quant_method.kt_config"
        ) from error

    wrapper_id = getattr(quant_method, "_quant_wrapper_id", None)
    physical_layer_index = getattr(kt_config, "layer_idx", None)
    gpu_experts_mask = getattr(kt_config, "gpu_experts_mask", None)
    if gpu_experts_mask is None:
        gpu_expert_count = None
    else:
        gpu_expert_count = int(gpu_experts_mask.sum().item())
    errors = []
    if wrapper_id != "kt_ep":
        errors.append(f"wrapper_id={wrapper_id!r}")
    if physical_layer_index != admission.physical_layer_index:
        errors.append(
            f"physical_layer_index={physical_layer_index!r} "
            f"(expected {admission.physical_layer_index})"
        )
    if gpu_expert_count != 0:
        errors.append(f"gpu_expert_count={gpu_expert_count!r} (expected 0)")
    shared_at_construction = getattr(
        draft_model,
        "kt_mtp_shared_embed_and_head_at_construction",
        False,
    )
    if shared_at_construction is not True:
        errors.append("embed/head modules were not shared at draft construction")
    if errors:
        raise KTMTPAdmissionError(
            "Loaded GLM-5.2 KT MTP proof failed: " + "; ".join(errors)
        )
    return {
        "enabled": True,
        "physical_layer_index": physical_layer_index,
        "gpu_expert_count": gpu_expert_count,
        "wrapper_id": wrapper_id,
        "shared_embed_and_head_at_construction": shared_at_construction,
    }
