# SPDX-License-Identifier: Apache-2.0
"""Fail-closed admission for GLM-5.2 MTP on KTransformers.

The target model and its MTP draft layer live in the same Hugging Face
checkpoint.  SGLang numbers the one-layer draft model locally as layer zero,
while KTransformers' persistent expert artifact keeps the physical checkpoint
number (``blk.78``).  This module validates that exact contract before the
draft worker is allowed to construct a KT MoE wrapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterator, Mapping

GLM52_ARCHITECTURE = "GlmMoeDsaForCausalLM"
GLM52_MODEL_TYPE = "glm_moe_dsa"
_HYBRID_MANIFEST_FILENAME: Final = "hybrid-checkpoint-manifest.json"
_HYBRID_MANIFEST_KIND: Final = "glm52_amxint4_ampere_w8a16_hybrid_checkpoint"
_SHARED_WEIGHT_MANIFEST_KIND: Final = "kt_shared_host_weights_manifest"
_SHARED_WEIGHT_CONTENT_KIND: Final = "kt_shared_host_weights_content"
_MANIFEST_SCHEMA_VERSION: Final = 1
_MAXIMUM_JSON_BYTES: Final = 64 * 1024 * 1024
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_OMITTED_EXPERT_COUNT: Final = 58_368
_EXPECTED_OMITTED_EXPERT_NAMES_SHA256: Final = (
    "dc2323c334d9af5eebbce9dd266059d4b9927945eeeff1b57cae9139f30ae1cc"
)
_EXPECTED_HYBRID_OUTPUT: Final[Mapping[str, object]] = {
    "payload_bytes": 20_056_714_112,
    "shard_count": 5,
    "tensor_count": 2_052,
}
_EXPECTED_HYBRID_QUANTIZATION: Final[Mapping[str, object]] = {
    "activation_dtype": "BF16",
    "checkpoint_layout": {
        "mla_kv_b": "gptq_packed_rows_per_head_v1",
        "ordinary_linear": "gptq_packed_rows",
    },
    "mla_kv_b_compact_to_compact_marlin_repack_at_load": True,
    "mla_kv_b_triton_uses_serialized_words_directly": True,
    "ordinary_linear_compact_to_compact_marlin_repack_at_load": True,
    "serialized_scale_dtype": "BF16",
    "serialized_weight_dtype": "INT8_biased_by_128_packed_in_INT32",
    "temporary_bf16_expansion_at_load": False,
}
_MLA_KV_B_W8_FORMAT: Final[Mapping[str, object]] = {
    "bits": 8,
    "block_size_m": 8,
    "format": "gptq_packed_rows_per_head_v1",
    "implicit_bias": 128,
    "kv_lora_rank": 512,
    "num_attention_heads": 64,
    "pack_axis": "K",
    "pack_order": "little_endian_k_lanes_0_1_2_3",
    "qk_nope_head_dim": 192,
    "scale_compute_dtype": "float32",
    "scale_dtype": "bfloat16",
    "v_head_dim": 256,
}
_HYBRID_QUANTIZATION_FIELDS: Final[Mapping[str, object]] = {
    "bits": 8,
    "checkpoint_format": "gptq",
    "desc_act": False,
    "group_size": -1,
    "lm_head": True,
    "quant_method": "gptq",
    "sym": True,
}
_HYBRID_LAYER_SUFFIXES: Final = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.gate.weight",
    "mlp.shared_experts.down_proj.qweight",
    "mlp.shared_experts.down_proj.scales",
    "mlp.shared_experts.gate_proj.qweight",
    "mlp.shared_experts.gate_proj.scales",
    "mlp.shared_experts.up_proj.qweight",
    "mlp.shared_experts.up_proj.scales",
    "post_attention_layernorm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.qweight",
    "self_attn.indexer.wk.scales",
    "self_attn.indexer.wq_b.qweight",
    "self_attn.indexer.wq_b.scales",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.qweight",
    "self_attn.kv_a_proj_with_mqa.scales",
    "self_attn.kv_b_proj.kc_qweight",
    "self_attn.kv_b_proj.kc_scales",
    "self_attn.kv_b_proj.vc_qweight",
    "self_attn.kv_b_proj.vc_scales",
    "self_attn.o_proj.qweight",
    "self_attn.o_proj.scales",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_a_proj.qweight",
    "self_attn.q_a_proj.scales",
    "self_attn.q_b_proj.qweight",
    "self_attn.q_b_proj.scales",
    "shared_head.norm.weight",
)
_COMPACT_KV_B_ABI: Final[Mapping[str, tuple[str, tuple[int, ...]]]] = {
    "kc_qweight": ("I32", (64, 48, 512)),
    "kc_scales": ("BF16", (64, 1, 512)),
    "vc_qweight": ("I32", (64, 128, 256)),
    "vc_scales": ("BF16", (64, 1, 256)),
}
_SAFETENSORS_DTYPE_BYTES: Final[Mapping[str, int]] = {
    "BF16": 2,
    "I32": 4,
}


class KTMTPAdmissionError(ValueError):
    """Raised when a GLM-5.2 KT MTP request is not provably supported."""


@dataclass(frozen=True)
class KTMTPAdmission:
    enabled: bool = False
    physical_layer_index: int | None = None
    reason: str = ""
    compact_mla_kv_b_w8: bool = False


@dataclass(frozen=True)
class _ManifestFile:
    path: str
    size_bytes: int
    sha256: str


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


def _load_json_bytes(path: Path, description: str) -> tuple[Mapping[str, Any], bytes]:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise KTMTPAdmissionError(
                f"{description} at {path} must be a regular non-symlink file"
            )
        if before.st_size <= 0 or before.st_size > _MAXIMUM_JSON_BYTES:
            raise KTMTPAdmissionError(
                f"{description} at {path} has an invalid JSON file size"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise KTMTPAdmissionError(
                    f"{description} at {path} became truncated while reading"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if tuple(getattr(before, field) for field in identity_fields) != tuple(
            getattr(after, field) for field in identity_fields
        ):
            raise KTMTPAdmissionError(
                f"{description} at {path} changed while it was read"
            )
        raw = b"".join(chunks)
        value = json.loads(raw)
    except KTMTPAdmissionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KTMTPAdmissionError(
            f"Cannot read {description} at {path}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise KTMTPAdmissionError(f"{description} at {path} must be a JSON object")
    return value, raw


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    return _load_json_bytes(path, description)[0]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise KTMTPAdmissionError(
            f"Cannot hash required file {path}: {error}"
        ) from error
    return digest.hexdigest()


def _require_effectively_read_only(
    path: Path,
    *,
    description: str,
    directory: bool,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise KTMTPAdmissionError(
            f"Cannot stat {description} at {path}: {error}"
        ) from error
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if path.is_symlink() or not expected_type(metadata.st_mode):
        expected_description = "directory" if directory else "regular file"
        raise KTMTPAdmissionError(
            f"{description} must be a non-symlink {expected_description}: {path}"
        )
    mount_read_only = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    mode_read_only = metadata.st_mode & 0o222 == 0
    if not mount_read_only and not mode_read_only:
        raise KTMTPAdmissionError(f"{description} is not immutable: {path}")


def _parse_manifest_files(
    value: object,
    *,
    description: str,
) -> tuple[_ManifestFile, ...]:
    if not isinstance(value, list) or not value:
        raise KTMTPAdmissionError(f"{description} must contain a non-empty file table")
    files: list[_ManifestFile] = []
    for raw_entry in value:
        if not isinstance(raw_entry, dict):
            raise KTMTPAdmissionError(
                f"{description} file entries must be JSON objects"
            )
        raw_path = raw_entry.get("path")
        raw_size = raw_entry.get("size_bytes")
        raw_sha256 = raw_entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise KTMTPAdmissionError(
                f"{description} file paths must be non-empty strings"
            )
        relative_path = PurePosixPath(raw_path)
        if (
            relative_path.is_absolute()
            or "." in relative_path.parts
            or ".." in relative_path.parts
            or relative_path.as_posix() != raw_path
            or "\x00" in raw_path
        ):
            raise KTMTPAdmissionError(
                f"{description} has unsafe file path {raw_path!r}"
            )
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise KTMTPAdmissionError(f"{description} has invalid size for {raw_path}")
        if not isinstance(raw_sha256, str) or _SHA256.fullmatch(raw_sha256) is None:
            raise KTMTPAdmissionError(
                f"{description} has invalid SHA-256 for {raw_path}"
            )
        files.append(_ManifestFile(raw_path, raw_size, raw_sha256))
    paths = tuple(entry.path for entry in files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise KTMTPAdmissionError(
            f"{description} file paths must be unique and strictly sorted"
        )
    return tuple(files)


def _verify_manifest_file_metadata(
    root: Path,
    files: tuple[_ManifestFile, ...],
    *,
    description: str,
) -> None:
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as error:
        raise KTMTPAdmissionError(
            f"Cannot resolve {description} root {root}: {error}"
        ) from error
    for entry in files:
        candidate = root.joinpath(*PurePosixPath(entry.path).parts)
        try:
            metadata = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            _ = resolved.relative_to(canonical_root)
        except (OSError, ValueError) as error:
            raise KTMTPAdmissionError(
                f"{description} file is unavailable or escapes its root: {entry.path}"
            ) from error
        if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise KTMTPAdmissionError(
                f"{description} file must be regular and non-symlink: {entry.path}"
            )
        if metadata.st_size != entry.size_bytes:
            raise KTMTPAdmissionError(f"{description} file size changed: {entry.path}")
        mount_read_only = bool(os.statvfs(candidate).f_flag & os.ST_RDONLY)
        mode_read_only = metadata.st_mode & 0o222 == 0
        if not mount_read_only and not mode_read_only:
            raise KTMTPAdmissionError(
                f"{description} file is not immutable: {entry.path}"
            )


def _read_safetensors_entries(
    path: Path,
    names: set[str],
) -> Mapping[str, Mapping[str, Any]]:
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise KTMTPAdmissionError(
                    f"safetensors shard has no complete header length: {path}"
                )
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > _MAXIMUM_JSON_BYTES:
                raise KTMTPAdmissionError(
                    f"safetensors shard has invalid header length: {path}"
                )
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise KTMTPAdmissionError(
                    f"safetensors shard has a truncated header: {path}"
                )
            shard_size = os.fstat(stream.fileno()).st_size
        header = json.loads(raw_header)
    except KTMTPAdmissionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KTMTPAdmissionError(
            f"Cannot read safetensors header at {path}: {error}"
        ) from error
    if not isinstance(header, dict):
        raise KTMTPAdmissionError(f"safetensors header at {path} is not an object")
    entries: dict[str, Mapping[str, Any]] = {}
    for name in names:
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise KTMTPAdmissionError(
                f"safetensors shard {path} has no tensor header for {name}"
            )
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            dtype not in _SAFETENSORS_DTYPE_BYTES
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in offsets
            )
        ):
            raise KTMTPAdmissionError(
                f"safetensors shard {path} has an invalid tensor header for {name}"
            )
        start, end = offsets
        assert isinstance(start, int) and isinstance(end, int)
        element_count = 1
        for dimension in shape:
            assert isinstance(dimension, int)
            element_count *= dimension
        expected_extent = element_count * _SAFETENSORS_DTYPE_BYTES[dtype]
        if (
            start < 0
            or end <= start
            or end - start != expected_extent
            or 8 + header_length + end > shard_size
        ):
            raise KTMTPAdmissionError(
                f"safetensors shard {path} has invalid data offsets for {name}"
            )
        entries[name] = entry
    return entries


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


def _validate_hybrid_quantization_config(model_path: Path) -> bool:
    config = _load_json(model_path / "config.json", "target checkpoint config")
    hybrid_manifest_is_named = os.path.lexists(model_path / _HYBRID_MANIFEST_FILENAME)
    raw_quantization = config.get("quantization_config")
    if not isinstance(raw_quantization, dict):
        if hybrid_manifest_is_named:
            raise KTMTPAdmissionError(
                "Hybrid target manifest is present but quantization_config is absent"
            )
        return False
    if "exo_mla_kv_b_w8" not in raw_quantization:
        if hybrid_manifest_is_named:
            raise KTMTPAdmissionError(
                "Hybrid target manifest is present but exo_mla_kv_b_w8 "
                "metadata is absent"
            )
        return False

    expected_quantization = dict(_HYBRID_QUANTIZATION_FIELDS)
    expected_quantization["dynamic"] = {
        r"-:.*\.mlp\.experts$": {},
        r"-:.*\.self_attn\.indexer\.weights_proj$": {},
    }
    expected_quantization["exo_mla_kv_b_w8"] = dict(_MLA_KV_B_W8_FORMAT)
    if raw_quantization != expected_quantization:
        raise KTMTPAdmissionError(
            "Hybrid target quantization_config does not exactly match the "
            "admitted Exo GPTQ W8/MLA kv_b format"
        )

    standalone_quantization = _load_json(
        model_path / "quantize_config.json",
        "target quantization config",
    )
    if standalone_quantization != raw_quantization:
        raise KTMTPAdmissionError(
            "Hybrid target config.json and quantize_config.json disagree"
        )
    return True


def _validate_compact_kv_b_headers(
    model_path: Path,
    weight_map: Mapping[str, Any],
    *,
    physical_layer_index: int,
) -> None:
    stem = f"model.layers.{physical_layer_index}.self_attn.kv_b_proj."
    names = {stem + suffix for suffix in _COMPACT_KV_B_ABI}
    names_by_shard: dict[str, set[str]] = {}
    for name in names:
        shard = weight_map[name]
        assert isinstance(shard, str)
        names_by_shard.setdefault(shard, set()).add(name)

    entries: dict[str, Mapping[str, Any]] = {}
    for shard, shard_names in names_by_shard.items():
        entries.update(_read_safetensors_entries(model_path / shard, shard_names))
    for suffix, (expected_dtype, expected_shape) in _COMPACT_KV_B_ABI.items():
        name = stem + suffix
        entry = entries[name]
        actual_dtype = entry.get("dtype")
        actual_shape = entry.get("shape")
        if actual_dtype != expected_dtype or actual_shape != list(expected_shape):
            raise KTMTPAdmissionError(
                "Hybrid target compact MLA kv_b tensor ABI mismatch for "
                f"{name}: dtype={actual_dtype!r}, shape={actual_shape!r}, "
                f"expected dtype={expected_dtype}, shape={list(expected_shape)}"
            )


def _validate_source_checkpoint(
    model_path: Path,
    physical_layer_index: int,
) -> tuple[bool, set[str]]:
    index_path = model_path / "model.safetensors.index.json"
    index = _load_json(index_path, "target checkpoint index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise KTMTPAdmissionError(
            f"Target checkpoint index {index_path} has no object weight_map"
        )

    prefix = f"model.layers.{physical_layer_index}."
    common_required = (
        prefix + "eh_proj.weight",
        prefix + "enorm.weight",
        prefix + "hnorm.weight",
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
        prefix + "shared_head.norm.weight",
    )
    if not _validate_hybrid_quantization_config(model_path):
        legacy_required = (
            *common_required,
            prefix + "mlp.experts.0.gate_proj.weight",
            prefix + "mlp.experts.255.down_proj.weight",
            prefix + "self_attn.kv_b_proj.weight",
        )
        required_shards = _validate_index_entries(
            model_path,
            weight_map,
            legacy_required,
            "target checkpoint",
        )
        return False, required_shards

    expected_names = {prefix + suffix for suffix in _HYBRID_LAYER_SUFFIXES}
    actual_names = {name for name in weight_map if name.startswith(prefix)}
    routed_experts = sorted(
        name for name in actual_names if name.startswith(prefix + "mlp.experts.")
    )
    if routed_experts:
        raise KTMTPAdmissionError(
            "Hybrid target must omit routed MTP experts in favor of the "
            f"attested KT artifact; first unexpected: {routed_experts[0]}"
        )
    legacy_kv_b_name = prefix + "self_attn.kv_b_proj.weight"
    if legacy_kv_b_name in actual_names:
        raise KTMTPAdmissionError(
            "Hybrid target ambiguously contains legacy kv_b_proj.weight "
            "alongside compact W8 metadata"
        )
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise KTMTPAdmissionError(
            "Hybrid target layer-78 tensor set differs from the admitted "
            f"35-tensor contract: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    required_shards = _validate_index_entries(
        model_path,
        weight_map,
        sorted(expected_names),
        "hybrid target checkpoint",
        require_flat_shards=True,
    )
    _validate_compact_kv_b_headers(
        model_path,
        weight_map,
        physical_layer_index=physical_layer_index,
    )
    return True, required_shards


def _validate_kt_artifact(
    weight_path: Path,
    *,
    physical_layer_index: int,
    numa_slot_count: int,
) -> set[str]:
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
    return _validate_index_entries(weight_path, weight_map, required, "KT artifact")


def _validate_index_entries(
    root: Path,
    weight_map: Mapping[str, Any],
    required: tuple[str, ...] | list[str],
    description: str,
    *,
    require_flat_shards: bool = False,
) -> set[str]:
    missing = [name for name in required if name not in weight_map]
    if missing:
        preview = ", ".join(missing[:3])
        raise KTMTPAdmissionError(
            f"{description} is missing {len(missing)} required layer-78 "
            f"entries; first missing: {preview}"
        )
    invalid_shards = [
        name for name in required if not isinstance(weight_map[name], str)
    ]
    if invalid_shards:
        raise KTMTPAdmissionError(
            f"{description} has non-string shard values for "
            + ", ".join(invalid_shards[:3])
        )
    required_shards = {str(weight_map[name]) for name in required}
    unsafe_shards = sorted(
        shard
        for shard in required_shards
        if (
            not shard
            or (path := PurePosixPath(shard)).is_absolute()
            or "." in path.parts
            or ".." in path.parts
            or path.as_posix() != shard
            or (require_flat_shards and len(path.parts) != 1)
            or "\x00" in shard
        )
    )
    if unsafe_shards:
        raise KTMTPAdmissionError(
            f"{description} has unsafe shard paths: " + ", ".join(unsafe_shards[:3])
        )
    missing_shards = sorted(
        shard for shard in required_shards if not (root / shard).is_file()
    )
    if missing_shards:
        raise KTMTPAdmissionError(
            f"{description} references missing shard files: "
            + ", ".join(missing_shards[:3])
        )
    return required_shards


def _files_by_path(
    files: tuple[_ManifestFile, ...],
) -> Mapping[str, _ManifestFile]:
    return {entry.path: entry for entry in files}


def _require_manifest_file_hashes(
    root: Path,
    files: tuple[_ManifestFile, ...],
    required_paths: set[str],
    *,
    description: str,
) -> None:
    by_path = _files_by_path(files)
    missing = sorted(required_paths - set(by_path))
    if missing:
        raise KTMTPAdmissionError(
            f"{description} does not attest required files: {missing[:3]}"
        )
    for relative_path in sorted(required_paths):
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        if _sha256_file(path) != by_path[relative_path].sha256:
            raise KTMTPAdmissionError(
                f"{description} SHA-256 differs for {relative_path}"
            )


def _validate_hybrid_checkpoint_manifest(
    model_path: Path,
    *,
    required_shards: set[str],
) -> Mapping[str, Any]:
    manifest_path = model_path / _HYBRID_MANIFEST_FILENAME
    _require_effectively_read_only(
        model_path,
        description="hybrid checkpoint root",
        directory=True,
    )
    _require_effectively_read_only(
        manifest_path,
        description="hybrid checkpoint manifest",
        directory=False,
    )
    manifest = _load_json(manifest_path, "hybrid checkpoint manifest")
    if (
        manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != _HYBRID_MANIFEST_KIND
    ):
        raise KTMTPAdmissionError(
            "Hybrid target has an unsupported checkpoint manifest schema"
        )
    if manifest.get("immutability") != {
        "directory_mode": "0555",
        "file_mode": "0444",
        "materialized_offline": True,
    }:
        raise KTMTPAdmissionError(
            "Hybrid target manifest has an invalid immutability contract"
        )
    if manifest.get("output") != _EXPECTED_HYBRID_OUTPUT:
        raise KTMTPAdmissionError(
            "Hybrid target manifest has an unexpected output contract"
        )
    raw_quantization = manifest.get("quantization")
    if not isinstance(raw_quantization, dict) or any(
        raw_quantization.get(name) != expected
        for name, expected in _EXPECTED_HYBRID_QUANTIZATION.items()
    ):
        raise KTMTPAdmissionError(
            "Hybrid target manifest does not prove direct compact W8 consumption"
        )
    content_id = manifest.get("content_id")
    if not isinstance(content_id, str) or _SHA256.fullmatch(content_id) is None:
        raise KTMTPAdmissionError(
            "Hybrid target manifest content_id is not a SHA-256 digest"
        )
    content_contract = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_id", "created_at_utc", "immutability", "tensors"}
    }
    if _sha256_bytes(_canonical_json_bytes(content_contract)) != content_id:
        raise KTMTPAdmissionError(
            "Hybrid target manifest content_id does not match its content contract"
        )

    files = _parse_manifest_files(
        manifest.get("files"),
        description="hybrid checkpoint manifest",
    )
    missing_shards = sorted(required_shards - set(_files_by_path(files)))
    if missing_shards:
        raise KTMTPAdmissionError(
            "Hybrid checkpoint manifest does not attest target layer-78 shards: "
            + ", ".join(missing_shards[:3])
        )
    _verify_manifest_file_metadata(
        model_path,
        files,
        description="hybrid checkpoint",
    )
    _require_manifest_file_hashes(
        model_path,
        files,
        {
            "config.json",
            "model.safetensors.index.json",
            "quantize_config.json",
        },
        description="hybrid checkpoint manifest",
    )

    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise KTMTPAdmissionError("Hybrid target manifest has no policy object")
    omit_expert = policy.get("omit_expert")
    if (
        not isinstance(omit_expert, dict)
        or omit_expert.get("source_tensor_count") != _EXPECTED_OMITTED_EXPERT_COUNT
        or policy.get("omitted_expert_names_sha256")
        != _EXPECTED_OMITTED_EXPERT_NAMES_SHA256
    ):
        raise KTMTPAdmissionError(
            "Hybrid target manifest does not prove the exact routed-expert "
            "omission policy"
        )

    expert_checkpoint = manifest.get("expert_checkpoint")
    if not isinstance(expert_checkpoint, dict):
        raise KTMTPAdmissionError(
            "Hybrid target manifest has no expert_checkpoint contract"
        )
    return expert_checkpoint


def _validate_shared_weight_environment(
    *,
    manifest_path: Path,
    content_id: str,
) -> None:
    enabled = os.getenv("KT_SHARED_HOST_WEIGHTS", "")
    if enabled not in {"", "0", "1"}:
        raise KTMTPAdmissionError("KT_SHARED_HOST_WEIGHTS must be exactly 0 or 1")
    configured_manifest = os.getenv("KT_SHARED_HOST_WEIGHTS_MANIFEST")
    configured_content_id = os.getenv("KT_SHARED_HOST_WEIGHTS_CONTENT_ID")
    if enabled == "1":
        if not configured_manifest or not configured_content_id:
            raise KTMTPAdmissionError(
                "Shared KT weights require manifest and content-ID environment bindings"
            )
        if not _same_local_path(configured_manifest, str(manifest_path)):
            raise KTMTPAdmissionError(
                "KT_SHARED_HOST_WEIGHTS_MANIFEST differs from the hybrid "
                "expert contract"
            )
        if configured_content_id != content_id:
            raise KTMTPAdmissionError(
                "KT_SHARED_HOST_WEIGHTS_CONTENT_ID differs from the hybrid "
                "expert contract"
            )
        if not os.getenv("KT_SHARED_HOST_WEIGHTS_STATE_DIR"):
            raise KTMTPAdmissionError(
                "Shared KT weights require KT_SHARED_HOST_WEIGHTS_STATE_DIR"
            )
        return
    if configured_manifest and not _same_local_path(
        configured_manifest, str(manifest_path)
    ):
        raise KTMTPAdmissionError(
            "Dormant KT shared-weight manifest differs from the hybrid contract"
        )
    if configured_content_id and configured_content_id != content_id:
        raise KTMTPAdmissionError(
            "Dormant KT shared-weight content ID differs from the hybrid contract"
        )


def _validate_hybrid_expert_contract(
    *,
    model_path: Path,
    weight_path: Path,
    target_required_shards: set[str],
    required_shards: set[str],
    numa_nodes: tuple[int, ...],
) -> None:
    expert_checkpoint = _validate_hybrid_checkpoint_manifest(
        model_path,
        required_shards=target_required_shards,
    )
    if expert_checkpoint.get("method") != "AMXINT4":
        raise KTMTPAdmissionError("Hybrid target expert contract must use AMXINT4")
    configured_weight_path = expert_checkpoint.get("weight_path")
    if not isinstance(configured_weight_path, str) or not _same_local_path(
        configured_weight_path,
        str(weight_path),
    ):
        raise KTMTPAdmissionError(
            "Hybrid target expert weight_path differs from --kt-weight-path"
        )

    raw_manifest_path = expert_checkpoint.get("manifest_path")
    expected_manifest_sha256 = expert_checkpoint.get("manifest_sha256")
    expected_content_id = expert_checkpoint.get("content_id")
    if (
        not isinstance(raw_manifest_path, str)
        or not Path(raw_manifest_path).is_absolute()
    ):
        raise KTMTPAdmissionError("Hybrid target expert manifest_path must be absolute")
    if (
        not isinstance(expected_manifest_sha256, str)
        or _SHA256.fullmatch(expected_manifest_sha256) is None
        or not isinstance(expected_content_id, str)
        or _SHA256.fullmatch(expected_content_id) is None
    ):
        raise KTMTPAdmissionError("Hybrid target expert manifest identity is invalid")

    manifest_path = Path(raw_manifest_path)
    manifest, raw_manifest = _load_json_bytes(
        manifest_path,
        "KT shared-host-weight manifest",
    )
    if _sha256_bytes(raw_manifest) != expected_manifest_sha256:
        raise KTMTPAdmissionError(
            "KT shared-host-weight manifest SHA-256 differs from the hybrid "
            "expert contract"
        )
    if (
        manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != _SHARED_WEIGHT_MANIFEST_KIND
    ):
        raise KTMTPAdmissionError(
            "KT shared-host-weight manifest has an unsupported schema"
        )
    if manifest.get("content_id") != expected_content_id:
        raise KTMTPAdmissionError(
            "KT shared-host-weight content ID differs from the hybrid expert contract"
        )
    raw_numa_nodes = manifest.get("numa_nodes")
    if (
        not isinstance(raw_numa_nodes, list)
        or any(
            isinstance(node, bool) or not isinstance(node, int) or node < 0
            for node in raw_numa_nodes
        )
        or len(raw_numa_nodes) != len(set(raw_numa_nodes))
        or tuple(raw_numa_nodes) != numa_nodes
    ):
        raise KTMTPAdmissionError(
            "KT shared-host-weight manifest NUMA order differs from kt_numa_nodes"
        )

    files = _parse_manifest_files(
        manifest.get("files"),
        description="KT shared-host-weight manifest",
    )
    descriptor = {
        "files": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in files
        ],
        "kind": _SHARED_WEIGHT_CONTENT_KIND,
        "schema_version": _MANIFEST_SCHEMA_VERSION,
    }
    if _sha256_bytes(_canonical_json_bytes(descriptor)) != expected_content_id:
        raise KTMTPAdmissionError(
            "KT shared-host-weight content ID does not match its file table"
        )
    _verify_manifest_file_metadata(
        weight_path,
        files,
        description="KT shared-host-weight checkpoint",
    )
    _require_manifest_file_hashes(
        weight_path,
        files,
        {"config.json", "model.safetensors.index.json"},
        description="KT shared-host-weight manifest",
    )
    manifest_paths = set(_files_by_path(files))
    missing_shards = sorted(required_shards - manifest_paths)
    if missing_shards:
        raise KTMTPAdmissionError(
            "KT shared-host-weight manifest does not attest MTP expert shards: "
            + ", ".join(missing_shards[:3])
        )
    _validate_shared_weight_environment(
        manifest_path=manifest_path,
        content_id=expected_content_id,
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
        "kt_lora_path": (None,),
        "speculative_draft_model_quantization": (None,),
        "init_expert_location": (None, "trivial"),
    }
    for name, admitted_values in optional_disabled.items():
        actual = _get(server_args, name)
        if actual not in admitted_values:
            mismatches.append(
                f"{name}={actual!r} (expected one of {admitted_values!r})"
            )
    for name in ("load_format", "speculative_draft_load_format"):
        actual = _get(server_args, name)
        normalized = getattr(actual, "value", actual)
        if normalized != "safetensors":
            mismatches.append(f"{name}={actual!r} (expected 'safetensors')")

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
    if (
        isinstance(threadpool_count, bool)
        or not isinstance(threadpool_count, int)
        or threadpool_count <= 0
    ):
        mismatches.append("kt_threadpool_count must be a positive integer")
    if (
        not isinstance(numa_nodes, (list, tuple))
        or any(
            isinstance(node, bool) or not isinstance(node, int) or node < 0
            for node in numa_nodes
        )
        or not isinstance(threadpool_count, int)
        or isinstance(threadpool_count, bool)
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
    compact_mla_kv_b_w8 = False
    if validate_artifacts:
        assert isinstance(threadpool_count, int)
        model_path_object = Path(model_path)
        weight_path_object = Path(_get(server_args, "kt_weight_path"))
        compact_mla_kv_b_w8, target_required_shards = _validate_source_checkpoint(
            model_path_object,
            physical_layer_index,
        )
        required_expert_shards = _validate_kt_artifact(
            weight_path_object,
            physical_layer_index=physical_layer_index,
            numa_slot_count=threadpool_count,
        )
        if compact_mla_kv_b_w8:
            assert isinstance(numa_nodes, (list, tuple))
            _validate_hybrid_expert_contract(
                model_path=model_path_object,
                weight_path=weight_path_object,
                target_required_shards=target_required_shards,
                required_shards=required_expert_shards,
                numa_nodes=tuple(numa_nodes),
            )

    return KTMTPAdmission(
        enabled=True,
        physical_layer_index=physical_layer_index,
        reason=(
            "admitted GLM-5.2 layer-78 NEXTN with persistent AMXINT4 experts"
            + (" and direct compact MLA kv_b W8" if compact_mla_kv_b_w8 else "")
        ),
        compact_mla_kv_b_w8=compact_mla_kv_b_w8,
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
    if admission.compact_mla_kv_b_w8:
        try:
            kv_b_projection = decoder.self_attn.kv_b_proj
            kv_b_quant_method = kv_b_projection.quant_method
        except AttributeError:
            errors.append("compact MLA kv_b projection is absent")
        else:
            if getattr(kv_b_quant_method, "is_mla_kv_b_w8", False) is not True:
                errors.append("kv_b projection does not use the compact W8 method")
            missing_compact_parameters = [
                name
                for name in ("kc_qweight", "kc_scales", "vc_qweight", "vc_scales")
                if not hasattr(kv_b_projection, name)
            ]
            if missing_compact_parameters:
                errors.append(
                    "kv_b projection lacks compact parameters "
                    + ", ".join(missing_compact_parameters)
                )
            if hasattr(kv_b_projection, "weight"):
                errors.append("kv_b projection materialized a legacy weight")
    if errors:
        raise KTMTPAdmissionError(
            "Loaded GLM-5.2 KT MTP proof failed: " + "; ".join(errors)
        )
    receipt = {
        "enabled": True,
        "physical_layer_index": physical_layer_index,
        "gpu_expert_count": gpu_expert_count,
        "wrapper_id": wrapper_id,
        "shared_embed_and_head_at_construction": shared_at_construction,
    }
    if admission.compact_mla_kv_b_w8:
        receipt["compact_mla_kv_b_w8"] = True
    return receipt
