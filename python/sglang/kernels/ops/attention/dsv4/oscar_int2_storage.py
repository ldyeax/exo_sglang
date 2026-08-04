"""OSCAR INT2 cache-storage prototype for DeepSeek V4 on SM86.

This is an OSCAR format, not the symmetric INT4 format in
``int4_storage.py``.  A caller must supply a numerically validated,
model-calibrated 448 x 448 orthogonal rotation, seven groupwise clip ratios
and indices, and seven diagnostic offline thresholds.  The no-PE latent is
rotated before clipping and asymmetric INT2 quantization.  The rotation is fit
from OSCAR's attention-weighted value/SST covariance and is admitted only if
the joint held-out QQT/SST metric also improves.  The 64 RoPE values remain
exact BF16.  Genuine OSCAR mode computes a per-row absolute order
statistic; the no-sort offline-threshold candidate is a separate explicit mode.

The fused writer adapts OSCAR's rotate -> clip -> asymmetric INT2 pack path to
DeepSeek V4's non-power-of-two 448-wide latent.  Seven independent 64-column
``tl.dot`` programs avoid materialising a rotated tensor and avoid trying to
fit a 448 x 448 rotation plus all outputs in one CTA.  Scales and zeros are
stored as BF16 and quantization uses those persisted values, so encoding and
decoding agree at bin boundaries.

Per-token byte layout (``oscar-int2-asym-g64-v1``)::

    [0, 112)     448 asymmetric INT2 codes, four consecutive values per byte
    [112, 240)    64 exact BF16 RoPE values
    [240, 268)     7 interleaved BF16 (scale, zero) pairs
    [268, 272)     4 alignment bytes, never written by a kernel

All CUDA hot-path APIs require caller-owned storage/output tensors and do not
allocate.  Negative locations are padding and leave destinations untouched.
The decoder returns the *rotated* no-PE coordinates: an attention consumer
must apply the same OSCAR rotation to queries.  It intentionally does not
apply an inverse rotation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

FORMAT_NAME = "oscar-int2-asym-g64-v1"
MASKED_WRITER_EXECUTION = "device-uniform-live-mask-row-v1"
CALIBRATION_DOMAIN = "attention_shared_latent"
ARTIFACT_FORMAT = "dsv4-oscar-int2-calibration"
ARTIFACT_VERSION = 2
ARTIFACT_ALGORITHM = (
    "oscar-dsv4-compressed-history-shared-v-sst-u-pbr-h64x7-c4-k-qqt-u-h128-pbr"
)
SHARED_ROTATION_COMPOSITION = "u-pbr-h64x7"
C4_ROTATION_COMPOSITION = "u-h128-pbr"
SHARED_ROTATION_OBJECTIVE = "attention_weighted_value_sst"
C4_ROTATION_OBJECTIVE = "query_qqt"
SHARED_LATENT_SOURCE = "compressed_history_only"
CONSUMER_SCOPE = "target_compressed_history_layers"
CLIP_PER_ROW_QUANTILE = "per_row_quantile"
CLIP_OFFLINE_GROUP_THRESHOLDS = "offline_group_thresholds"
ARTIFACT_CLIP_SEMANTICS = "per_row_per_group_abs_order_stat_then_affine_u2_v1"
OSCAR_SOURCE_COMMIT = "797e39c7ccf442c5c6789f87c5a692ee6e98b263"
OSCAR_ROTATION_SOURCE_SHA256 = (
    "5f89868fe3fd80cecb7202b802744a83eb0e2d273c6a9960df458b8436b10b2f"
)
OSCAR_CLIP_SOURCE_SHA256 = (
    "c1d7fd911c688cf29df9b98ce19fb48c6e7147ea6fcc81761e33cbf5f38b4157"
)
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
GROUP_SIZE = 64
NUM_GROUPS = NOPE_DIM // GROUP_SIZE
INT2_VALUES_PER_BYTE = 4
PACKED_GROUP_BYTES = GROUP_SIZE // INT2_VALUES_PER_BYTE
PACKED_NOPE_BYTES = NOPE_DIM // INT2_VALUES_PER_BYTE
ROPE_BYTES = ROPE_DIM * torch.bfloat16.itemsize
SCALE_ZERO_VALUES = NUM_GROUPS * 2
SCALE_ZERO_BYTES = SCALE_ZERO_VALUES * torch.bfloat16.itemsize
ROPE_OFFSET_BYTES = PACKED_NOPE_BYTES
SCALE_ZERO_OFFSET_BYTES = ROPE_OFFSET_BYTES + ROPE_BYTES
LOGICAL_BYTES_PER_TOKEN = SCALE_ZERO_OFFSET_BYTES + SCALE_ZERO_BYTES
STORAGE_BYTES_PER_TOKEN = 272
OSCAR_INT2_STORAGE_BYTES_PER_TOKEN = STORAGE_BYTES_PER_TOKEN
PADDING_BYTES_PER_TOKEN = STORAGE_BYTES_PER_TOKEN - LOGICAL_BYTES_PER_TOKEN
INT2_MIN = 0
INT2_MAX = 3
_ROTATE_BLOCK_TOKENS = 16
_ROTATION_INPUT_TILE = 512
_IDENTITY_REJECTION_ATOL = 1.0e-3
_EXPECTED_COMPRESSION_RATIOS = (
    0,
    0,
    *(value for _ in range(20) for value in (4, 128)),
    4,
)
_REQUIRED_LAYER_IDS = frozenset(
    layer_id
    for layer_id, ratio in enumerate(_EXPECTED_COMPRESSION_RATIOS)
    if ratio != 0
)


@dataclass(frozen=True)
class OscarInt2Calibration:
    """Validated, explicit OSCAR calibration tensors.

    Instances are produced only by :func:`validate_oscar_int2_calibration`.
    Validation is a setup-time operation and must happen outside CUDA graph
    capture.  ``rotation`` is BF16 ``[448, 448]``; ``clip_ratios`` and
    diagnostic ``clip_thresholds`` are FP32 ``[7]``; ``clip_indices`` is
    INT16 ``[7]`` with ``floor(ratio * 64)`` semantics.
    """

    rotation: torch.Tensor
    clip_thresholds: torch.Tensor
    clip_ratios: torch.Tensor
    clip_indices: torch.Tensor
    layer_id: int
    clip_mode: str
    clip_provenance: str
    domain: str
    max_orthogonality_error: float


def oscar_int2_page_bytes(page_size: int) -> int:
    """Return the exact byte count of one token-major OSCAR INT2 page."""

    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError(f"page_size must be a positive integer, got {page_size!r}")
    return page_size * STORAGE_BYTES_PER_TOKEN


def validate_oscar_int2_calibration(
    rotation: torch.Tensor,
    clip_thresholds: torch.Tensor,
    clip_ratios: torch.Tensor,
    clip_indices: torch.Tensor,
    *,
    layer_id: int,
    clip_mode: str,
    clip_provenance: str,
    orthogonality_atol: float = 2.0e-2,
) -> OscarInt2Calibration:
    """Validate explicit OSCAR calibration tensors and bind them together.

    This performs a full FP32 ``R.T @ R`` check.  It is deliberately separate
    from the hot writer so graph replay never hides calibration work or a
    device allocation.  No identity/default rotation or implicit clipping is
    substituted when calibration is absent.
    """

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("OSCAR calibration validation is forbidden in CUDA capture")
    if rotation.shape != (NOPE_DIM, NOPE_DIM):
        raise ValueError(
            f"rotation must have shape ({NOPE_DIM}, {NOPE_DIM}), "
            f"got {tuple(rotation.shape)}"
        )
    if rotation.dtype != torch.bfloat16:
        raise ValueError(f"rotation must be BF16, got {rotation.dtype}")
    if not rotation.is_contiguous():
        raise ValueError("rotation must be contiguous")
    if clip_thresholds.shape != (NUM_GROUPS,):
        raise ValueError(
            f"clip_thresholds must have shape ({NUM_GROUPS},), "
            f"got {tuple(clip_thresholds.shape)}"
        )
    if clip_thresholds.dtype != torch.float32:
        raise ValueError(f"clip_thresholds must be FP32, got {clip_thresholds.dtype}")
    if not clip_thresholds.is_contiguous():
        raise ValueError("clip_thresholds must be contiguous")
    if clip_thresholds.device != rotation.device:
        raise ValueError("rotation and clip_thresholds must share one device")
    if clip_ratios.shape != (NUM_GROUPS,) or clip_ratios.dtype != torch.float32:
        raise ValueError(f"clip_ratios must be FP32 with shape ({NUM_GROUPS},)")
    if clip_indices.shape != (NUM_GROUPS,) or clip_indices.dtype != torch.int16:
        raise ValueError(f"clip_indices must be INT16 with shape ({NUM_GROUPS},)")
    if not clip_ratios.is_contiguous() or not clip_indices.is_contiguous():
        raise ValueError("clip_ratios and clip_indices must be contiguous")
    if clip_ratios.device != rotation.device or clip_indices.device != rotation.device:
        raise ValueError("all OSCAR calibration tensors must share one device")
    if not torch.isfinite(rotation).all().item():
        raise ValueError("rotation contains non-finite values")
    if not torch.isfinite(clip_thresholds).all().item():
        raise ValueError("clip_thresholds contains non-finite values")
    if not torch.all(clip_thresholds > 0).item():
        raise ValueError("every OSCAR clip threshold must be positive")
    if not torch.isfinite(clip_ratios).all().item():
        raise ValueError("clip_ratios contains non-finite values")
    if not torch.all((clip_ratios > 0.0) & (clip_ratios <= 1.0)).item():
        raise ValueError("every OSCAR clip ratio must be in (0, 1]")
    if not isinstance(orthogonality_atol, float) or orthogonality_atol <= 0.0:
        raise ValueError(
            f"orthogonality_atol must be a positive float, got {orthogonality_atol!r}"
        )
    if not isinstance(layer_id, int) or not 0 <= layer_id <= 42:
        raise ValueError(f"layer_id must be an integer in [0, 42], got {layer_id!r}")
    expected_clip_indices = torch.minimum(
        torch.floor(clip_ratios.to(torch.float64) * GROUP_SIZE).to(torch.int16),
        torch.full_like(clip_indices, GROUP_SIZE - 1),
    )
    if not torch.equal(clip_indices, expected_clip_indices):
        raise ValueError("clip_indices do not match floor(clip_ratios * group_size)")
    if not isinstance(clip_provenance, str) or not clip_provenance.strip():
        raise ValueError("clip_provenance must be a non-empty calibration identifier")
    if clip_mode not in (
        CLIP_PER_ROW_QUANTILE,
        CLIP_OFFLINE_GROUP_THRESHOLDS,
    ):
        raise ValueError(
            "clip_mode must explicitly select per_row_quantile or "
            "offline_group_thresholds"
        )

    rotation_fp32 = rotation.float()
    gram = rotation_fp32.T @ rotation_fp32
    identity = torch.eye(NOPE_DIM, dtype=torch.float32, device=rotation.device)
    max_error = (gram - identity).abs().amax().item()
    if max_error > orthogonality_atol:
        raise ValueError(
            "OSCAR rotation is not orthogonal within tolerance: "
            f"max_error={max_error:.8f}, atol={orthogonality_atol:.8f}"
        )
    max_identity_deviation = (rotation_fp32 - identity).abs().amax().item()
    if max_identity_deviation <= _IDENTITY_REJECTION_ATOL:
        raise ValueError(
            "identity is not a calibrated OSCAR rotation; no fallback is allowed"
        )
    return OscarInt2Calibration(
        rotation=rotation,
        clip_thresholds=clip_thresholds,
        clip_ratios=clip_ratios,
        clip_indices=clip_indices,
        layer_id=layer_id,
        clip_mode=clip_mode,
        clip_provenance=clip_provenance,
        domain=CALIBRATION_DOMAIN,
        max_orthogonality_error=max_error,
    )


def _require_mapping(value: Any, *, field: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"OSCAR artifact field {field!r} must be a mapping")
    return value


def _require_tensor(value: Any, *, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"OSCAR artifact field {field!r} must be a tensor")
    return value


def _require_exact_keys(
    value: Mapping[Any, Any], *, field: str, expected: set[str]
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"OSCAR artifact field {field!r} has wrong keys: "
            f"missing={missing}, extra={extra}"
        )


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"OSCAR artifact field {field!r} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"OSCAR artifact field {field!r} must be a SHA-256 hex digest"
        ) from error
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    contiguous_cpu = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(contiguous_cpu.dtype).encode("ascii"))
    digest.update(_canonical_json(list(contiguous_cpu.shape)))
    digest.update(contiguous_cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _update_tree_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        digest.update(_tensor_content_digest(value).encode("ascii"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(_canonical_json(str(key)))
            _update_tree_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_tree_digest(digest, item)
        digest.update(b"]")
        return
    digest.update(b"scalar:")
    digest.update(_canonical_json(value))


def tree_sha256(value: Any) -> str:
    """Return the canonical OSCAR artifact tree digest."""

    digest = hashlib.sha256()
    _update_tree_digest(digest, value)
    return digest.hexdigest()


def _validate_c4_scorer_artifact_payload(
    value: Any,
    *,
    layer_id: int,
) -> None:
    field = f"layers[{layer_id}].c4_scorer"
    payload = _require_mapping(value, field=field)
    _require_exact_keys(
        payload,
        field=field,
        expected={
            "domain",
            "rotation_objective",
            "rotation_composition",
            "rotation",
            "eigenvalues",
            "clip_thresholds",
            "clip_mode",
            "clip_ratios",
            "clip_indices",
            "clip_semantics",
            "clip_calibration",
            "provenance_sha256",
            "orthogonality_max_abs",
            "heldout_metrics",
        },
    )
    if payload.get("domain") != "c4_scorer":
        raise ValueError(f"{field}.domain must be 'c4_scorer'")
    if payload.get("rotation_objective") != C4_ROTATION_OBJECTIVE:
        raise ValueError(f"{field}.rotation_objective is unsupported")
    if payload.get("rotation_composition") != C4_ROTATION_COMPOSITION:
        raise ValueError(f"{field}.rotation_composition is unsupported")
    rotation = _require_tensor(payload.get("rotation"), field=f"{field}.rotation")
    eigenvalues = _require_tensor(
        payload.get("eigenvalues"), field=f"{field}.eigenvalues"
    )
    thresholds = _require_tensor(
        payload.get("clip_thresholds"), field=f"{field}.clip_thresholds"
    )
    ratios = _require_tensor(payload.get("clip_ratios"), field=f"{field}.clip_ratios")
    indices = _require_tensor(
        payload.get("clip_indices"), field=f"{field}.clip_indices"
    )
    if rotation.dtype != torch.float32 or rotation.shape != (128, 128):
        raise ValueError(f"{field}.rotation must be FP32 [128, 128]")
    if eigenvalues.dtype != torch.float32 or eigenvalues.shape != (128,):
        raise ValueError(f"{field}.eigenvalues must be FP32 [128]")
    if thresholds.dtype != torch.float32 or thresholds.shape != (1,):
        raise ValueError(f"{field}.clip_thresholds must be FP32 [1]")
    if ratios.dtype != torch.float32 or ratios.shape != (1,):
        raise ValueError(f"{field}.clip_ratios must be FP32 [1]")
    if indices.dtype != torch.int16 or indices.shape != (1,):
        raise ValueError(f"{field}.clip_indices must be INT16 [1]")
    expected_indices = torch.minimum(
        torch.floor(ratios.to(torch.float64) * 128).to(torch.int16),
        torch.full_like(indices, 127),
    )
    if not torch.equal(indices, expected_indices):
        raise ValueError(f"{field}.clip_indices do not match clip_ratios")
    if payload.get("clip_mode") != CLIP_PER_ROW_QUANTILE:
        raise ValueError(f"{field}.clip_mode must be per_row_quantile")
    if payload.get("clip_semantics") != ARTIFACT_CLIP_SEMANTICS:
        raise ValueError(f"{field}.clip_semantics is unsupported")
    _require_mapping(payload.get("clip_calibration"), field=f"{field}.clip_calibration")
    _validate_sha256(
        payload.get("provenance_sha256"), field=f"{field}.provenance_sha256"
    )
    declared_orthogonality = payload.get("orthogonality_max_abs")
    if not isinstance(declared_orthogonality, float):
        raise TypeError(f"{field}.orthogonality_max_abs must be a float")
    # The offline proof is computed from the serialized FP32 matrix promoted
    # to FP64.  Recompute in that exact arithmetic here: an FP32 GEMM can add
    # several ulps and falsely reject a valid dense OSCAR rotation even though
    # the tensor and recorded proof are identical.
    rotation_fp64 = rotation.to(torch.float64)
    gram = rotation_fp64.T @ rotation_fp64
    actual_orthogonality = (
        (gram - torch.eye(128, dtype=torch.float64, device=rotation.device))
        .abs()
        .amax()
        .item()
    )
    if actual_orthogonality > 2.0e-5:
        raise ValueError(
            f"{field}.rotation is not orthogonal: max_error={actual_orthogonality:.8f}"
        )
    if abs(actual_orthogonality - declared_orthogonality) > 1.0e-7:
        raise ValueError(f"{field}.orthogonality_max_abs does not match its tensor")
    _validate_heldout_metrics(
        payload.get("heldout_metrics"), field=f"{field}.heldout_metrics"
    )


def _validate_heldout_metrics(value: Any, *, field: str) -> None:
    metrics = _require_mapping(value, field=field)
    required = {
        "query_relative_error",
        "value_relative_error",
        "joint_relative_error",
        "unrotated_query_relative_error",
        "unrotated_value_relative_error",
        "unrotated_joint_relative_error",
        "improvement_vs_unrotated",
        "row_count",
    }
    _require_exact_keys(metrics, field=field, expected=required)
    for key in required:
        metric = metrics.get(key)
        if not isinstance(metric, (float, int)) or not math.isfinite(float(metric)):
            raise ValueError(f"{field}.{key} must be a finite number")
    row_count = metrics["row_count"]
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 16:
        raise ValueError(f"{field}.row_count must be an integer >= 16")
    if float(metrics["joint_relative_error"]) > 0.55:
        raise ValueError(f"{field}.joint_relative_error fails admission gate")
    if float(metrics["improvement_vs_unrotated"]) < 0.0:
        raise ValueError(f"{field}.improvement_vs_unrotated fails admission gate")


def load_dsv4_oscar_int2_calibrations(
    artifact_path: str | Path,
    *,
    device: str | torch.device,
    expected_metadata: Mapping[str, str] | None = None,
) -> dict[int, OscarInt2Calibration]:
    """Load and fail-closed validate every compressed-layer calibration.

    ``artifact_path`` must be absolute.  The safe ``weights_only`` loader is
    used, the frozen schema/domain are exact, layer coverage must be exactly
    compressed layers 2..42, and every runtime BF16 rotation receives a numerical orthogonality
    check.  When ``expected_metadata`` is supplied, keys are resolved only
    against the frozen ``model`` / ``provenance`` / top-level hash fields and
    must match exactly.  Missing expected keys fail rather than falling back.

    Artifact schema::

        {
          "format": "dsv4_oscar_int2_calibration_v1",
          "format_version": 2,
          "algorithm": {...}, "model": {...}, "quantization": {...},
          "provenance": {...}, "artifact_provenance_sha256": "...",
          "layers": {
            2: {"layer_id": 2, "compress_ratio": int,
                "attention_shared_latent": {
              "domain": "attention_shared_latent",
              "rotation_composition": "u-pbr-h64x7",
              "rotation": Tensor[448,448], "clip_thresholds": Tensor[7],
              "clip_mode": "per_row_quantile",
              "clip_ratios": Tensor[7], "clip_indices": Tensor[7],
              "clip_semantics": "static_per_group_median_row_abs_order_stat_v1",
              "provenance_sha256": str, ...
            }}, ... 42: {...}
          },
        }

    Extra layer payloads such as a separately calibrated ``c4_scorer`` are
    preserved in the artifact but ignored here; they can never substitute for
    ``attention_shared_latent``.
    """

    path = Path(artifact_path)
    if not path.is_absolute():
        raise ValueError("SGLANG_DSV4_OSCAR_CALIBRATION_PATH must be absolute")
    if not path.is_file():
        raise FileNotFoundError(f"OSCAR calibration artifact does not exist: {path}")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("OSCAR artifacts must be loaded before CUDA graph capture")

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    artifact = _require_mapping(loaded, field="root")
    _require_exact_keys(
        artifact,
        field="root",
        expected={
            "format",
            "format_version",
            "algorithm",
            "model",
            "quantization",
            "provenance",
            "layers",
            "artifact_provenance_sha256",
        },
    )
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise ValueError(
            f"OSCAR artifact format must be {ARTIFACT_FORMAT!r}, "
            f"got {artifact.get('format')!r}"
        )
    if artifact.get("format_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"OSCAR artifact version must be {ARTIFACT_VERSION}, "
            f"got {artifact.get('format_version')!r}"
        )
    if artifact.get("algorithm") != ARTIFACT_ALGORITHM:
        raise ValueError("artifact is not the calibrated shared-latent OSCAR algorithm")
    declared_artifact_digest = _validate_sha256(
        artifact.get("artifact_provenance_sha256"),
        field="artifact_provenance_sha256",
    )
    digest_payload = {
        key: value
        for key, value in artifact.items()
        if key != "artifact_provenance_sha256"
    }
    actual_artifact_digest = tree_sha256(digest_payload)
    if actual_artifact_digest != declared_artifact_digest:
        raise ValueError(
            "OSCAR artifact_provenance_sha256 does not match the artifact content"
        )

    model = _require_mapping(artifact.get("model"), field="model")
    _require_exact_keys(
        model,
        field="model",
        expected={
            "model_id",
            "model_type",
            "checkpoint_sha256",
            "config_sha256",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "latent_dim",
            "rope_dim",
            "index_head_dim",
            "index_n_heads",
            "compression_ratios",
        },
    )
    if model.get("num_hidden_layers") != 43:
        raise ValueError("OSCAR artifact model must have exactly 43 layers")
    expected_model_geometry = {
        "model_type": "deepseek_v4",
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "latent_dim": NOPE_DIM,
        "rope_dim": ROPE_DIM,
        "index_head_dim": 128,
        "index_n_heads": 64,
    }
    if any(model.get(key) != value for key, value in expected_model_geometry.items()):
        raise ValueError("OSCAR artifact model has incompatible shared-latent geometry")
    for hash_key in ("checkpoint_sha256", "config_sha256"):
        _validate_sha256(model.get(hash_key), field=f"model.{hash_key}")
    compression_ratios = _require_tensor(
        model.get("compression_ratios"), field="model.compression_ratios"
    )
    if compression_ratios.dtype != torch.int16 or compression_ratios.shape != (43,):
        raise ValueError(
            "model.compression_ratios must be an INT16 tensor of shape (43,)"
        )
    if tuple(int(value) for value in compression_ratios.tolist()) != tuple(
        _EXPECTED_COMPRESSION_RATIOS
    ):
        raise ValueError("model.compression_ratios do not match DSV4 Flash")

    quantization = _require_mapping(artifact.get("quantization"), field="quantization")
    expected_quantization: dict[str, str | int] = {
        "bits": 2,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "storage_layout": FORMAT_NAME,
        "codes_bytes_per_token": PACKED_NOPE_BYTES,
        "scale_zero_bytes_per_token": SCALE_ZERO_BYTES,
        "rope_bytes_per_token": ROPE_BYTES,
        "logical_bytes_per_token": LOGICAL_BYTES_PER_TOKEN,
        "padded_bytes_per_token": STORAGE_BYTES_PER_TOKEN,
    }
    _require_exact_keys(
        quantization,
        field="quantization",
        expected=set(expected_quantization),
    )
    for key, expected_value in expected_quantization.items():
        if quantization.get(key) != expected_value:
            raise ValueError(
                f"OSCAR artifact quantization mismatch for {key!r}: "
                f"expected {expected_value!r}, got {quantization.get(key)!r}"
            )

    provenance = _require_mapping(artifact.get("provenance"), field="provenance")
    _require_exact_keys(
        provenance,
        field="provenance",
        expected={
            "prompt_manifest_sha256",
            "capture_manifest_sha256",
            "checkpoint_fingerprint_sha256",
            "capture_input_set_sha256",
            "calibration_prompt_tokens",
            "train_latent_rows",
            "heldout_latent_rows",
            "shared_latent_source",
            "shared_rotation_objective",
            "c4_rotation_objective",
            "oscar_source_commit",
            "oscar_rotation_source_sha256",
            "oscar_clip_source_sha256",
            "statistics_sha256",
            "train_sample_rows",
            "heldout_sample_rows",
            "consumer_scope",
            "shared_rotation_composition",
            "c4_rotation_composition",
            "calibrator_source_sha256",
        },
    )
    for key, value in provenance.items():
        if not isinstance(key, str):
            raise TypeError("OSCAR artifact provenance keys must be strings")
        if key.endswith("sha256"):
            _validate_sha256(value, field=f"provenance.{key}")
        elif key.endswith("count") and (not isinstance(value, int) or value < 0):
            raise ValueError(f"provenance.{key} must be a non-negative integer")
    if provenance.get("oscar_source_commit") != OSCAR_SOURCE_COMMIT:
        raise ValueError("artifact OSCAR source commit mismatch")
    if provenance.get("oscar_rotation_source_sha256") != OSCAR_ROTATION_SOURCE_SHA256:
        raise ValueError("artifact OSCAR rotation source hash mismatch")
    if provenance.get("oscar_clip_source_sha256") != OSCAR_CLIP_SOURCE_SHA256:
        raise ValueError("artifact OSCAR clip source hash mismatch")
    if provenance.get("shared_latent_source") != SHARED_LATENT_SOURCE:
        raise ValueError("artifact shared latent source is not compressed history")
    if provenance.get("shared_rotation_objective") != SHARED_ROTATION_OBJECTIVE:
        raise ValueError("artifact shared rotation objective is not V/SST")
    if provenance.get("c4_rotation_objective") != C4_ROTATION_OBJECTIVE:
        raise ValueError("artifact C4 rotation objective is not K/QQT")
    if provenance.get("consumer_scope") != CONSUMER_SCOPE:
        raise ValueError("artifact OSCAR consumer scope mismatch")
    if provenance.get("shared_rotation_composition") != SHARED_ROTATION_COMPOSITION:
        raise ValueError("artifact shared rotation composition mismatch")
    if provenance.get("c4_rotation_composition") != C4_ROTATION_COMPOSITION:
        raise ValueError("artifact C4 rotation composition mismatch")
    for count_key in (
        "calibration_prompt_tokens",
        "train_latent_rows",
        "heldout_latent_rows",
        "train_sample_rows",
        "heldout_sample_rows",
    ):
        count = provenance.get(count_key)
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"provenance.{count_key} must be a positive integer")

    if expected_metadata is not None:
        metadata_sources = (model, provenance, artifact)
        for key, expected_value in expected_metadata.items():
            if not isinstance(key, str) or not isinstance(expected_value, str):
                raise TypeError("expected_metadata keys/values must be strings")
            matches = [source[key] for source in metadata_sources if key in source]
            if len(matches) != 1 or matches[0] != expected_value:
                actual_value = matches[0] if len(matches) == 1 else None
                raise ValueError(
                    f"OSCAR artifact metadata mismatch for {key!r}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )

    layers = _require_mapping(artifact.get("layers"), field="layers")
    if set(layers) != _REQUIRED_LAYER_IDS:
        raise ValueError("OSCAR artifact must cover exactly compressed layers 2..42")

    target_device = torch.device(device)
    calibrations: dict[int, OscarInt2Calibration] = {}
    for layer_id in sorted(_REQUIRED_LAYER_IDS):
        layer = _require_mapping(layers[layer_id], field=f"layers[{layer_id}]")
        compress_ratio = compression_ratios[layer_id].item()
        expected_layer_keys = {
            "layer_id",
            "compress_ratio",
            CALIBRATION_DOMAIN,
        }
        if compress_ratio == 4:
            expected_layer_keys.add("c4_scorer")
        _require_exact_keys(
            layer,
            field=f"layers[{layer_id}]",
            expected=expected_layer_keys,
        )
        if layer.get("layer_id") != layer_id:
            raise ValueError(
                f"OSCAR artifact layer {layer_id} has a mismatched layer_id"
            )
        if layer.get("compress_ratio") != compress_ratio:
            raise ValueError(
                f"OSCAR artifact layer {layer_id} has a mismatched compress_ratio"
            )
        if compress_ratio == 4:
            _validate_c4_scorer_artifact_payload(
                layer.get("c4_scorer"), layer_id=layer_id
            )
        shared = _require_mapping(
            layer.get(CALIBRATION_DOMAIN),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}",
        )
        _require_exact_keys(
            shared,
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}",
            expected={
                "domain",
                "rotation_objective",
                "rotation_composition",
                "rotation",
                "eigenvalues",
                "clip_thresholds",
                "clip_mode",
                "clip_ratios",
                "clip_indices",
                "clip_semantics",
                "clip_calibration",
                "provenance_sha256",
                "orthogonality_max_abs",
                "heldout_metrics",
            },
        )
        if shared.get("domain") != CALIBRATION_DOMAIN:
            raise ValueError(
                f"layer {layer_id} OSCAR payload has the wrong consumer domain"
            )
        if shared.get("rotation_objective") != SHARED_ROTATION_OBJECTIVE:
            raise ValueError(
                f"layer {layer_id} has unsupported shared rotation objective"
            )
        if shared.get("rotation_composition") != SHARED_ROTATION_COMPOSITION:
            raise ValueError(
                f"layer {layer_id} has unsupported shared rotation composition"
            )
        rotation = _require_tensor(
            shared.get("rotation"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.rotation",
        )
        thresholds = _require_tensor(
            shared.get("clip_thresholds"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.clip_thresholds",
        )
        eigenvalues = _require_tensor(
            shared.get("eigenvalues"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.eigenvalues",
        )
        if rotation.dtype != torch.float32 or rotation.shape != (NOPE_DIM, NOPE_DIM):
            raise ValueError(f"layer {layer_id} rotation must be FP32 [448, 448]")
        if eigenvalues.dtype != torch.float32 or eigenvalues.shape != (NOPE_DIM,):
            raise ValueError(f"layer {layer_id} eigenvalues must be FP32 [448]")
        if thresholds.dtype != torch.float32 or thresholds.shape != (NUM_GROUPS,):
            raise ValueError(f"layer {layer_id} clip_thresholds must be FP32 [7]")
        clip_ratios = _require_tensor(
            shared.get("clip_ratios"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.clip_ratios",
        )
        clip_indices = _require_tensor(
            shared.get("clip_indices"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.clip_indices",
        )
        if clip_ratios.dtype != torch.float32 or clip_ratios.shape != (NUM_GROUPS,):
            raise ValueError(f"layer {layer_id} clip_ratios must be FP32 [7]")
        if clip_indices.dtype != torch.int16 or clip_indices.shape != (NUM_GROUPS,):
            raise ValueError(f"layer {layer_id} clip_indices must be INT16 [7]")
        runtime_rotation = rotation.to(
            device=target_device, dtype=torch.bfloat16
        ).contiguous()
        runtime_thresholds = thresholds.to(
            device=target_device, dtype=torch.float32
        ).contiguous()
        runtime_clip_ratios = clip_ratios.to(
            device=target_device, dtype=torch.float32
        ).contiguous()
        runtime_clip_indices = clip_indices.to(
            device=target_device, dtype=torch.int16
        ).contiguous()
        clip_mode = shared.get("clip_mode")
        clip_semantics = shared.get("clip_semantics")
        clip_provenance = shared.get("provenance_sha256")
        if clip_mode != CLIP_PER_ROW_QUANTILE:
            raise ValueError(
                f"layer {layer_id} artifact clip_mode must be per_row_quantile"
            )
        if not isinstance(clip_provenance, str):
            raise TypeError(f"layer {layer_id} clip_provenance must be a string")
        _validate_sha256(
            clip_provenance,
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.provenance_sha256",
        )
        if clip_semantics != ARTIFACT_CLIP_SEMANTICS:
            raise ValueError(
                f"layer {layer_id} has unsupported artifact clip semantics "
                f"{clip_semantics!r}"
            )
        declared_orthogonality = shared.get("orthogonality_max_abs")
        if not isinstance(declared_orthogonality, float):
            raise TypeError(f"layer {layer_id} orthogonality_max_abs must be a float")
        rotation_fp64 = rotation.to(torch.float64)
        actual_orthogonality = (
            (
                rotation_fp64.T @ rotation_fp64
                - torch.eye(NOPE_DIM, dtype=torch.float64, device=rotation.device)
            )
            .abs()
            .amax()
            .item()
        )
        if actual_orthogonality > 2.0e-5:
            raise ValueError(
                f"layer {layer_id} artifact rotation is not orthogonal: "
                f"max_error={actual_orthogonality:.8f}"
            )
        if abs(actual_orthogonality - declared_orthogonality) > 1.0e-7:
            raise ValueError(
                f"layer {layer_id} orthogonality_max_abs does not match its tensor"
            )
        _require_mapping(
            shared.get("clip_calibration"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.clip_calibration",
        )
        _validate_heldout_metrics(
            shared.get("heldout_metrics"),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}.heldout_metrics",
        )
        calibrations[layer_id] = validate_oscar_int2_calibration(
            runtime_rotation,
            runtime_thresholds,
            runtime_clip_ratios,
            runtime_clip_indices,
            layer_id=layer_id,
            clip_mode=CLIP_PER_ROW_QUANTILE,
            clip_provenance=clip_provenance,
        )
    return calibrations


def _validate_reference_values(values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(
            f"values must have shape (num_tokens, {HEAD_DIM}), got {values.shape}"
        )
    if values.dtype != torch.bfloat16:
        raise ValueError(f"values must be BF16, got {values.dtype}")


def _validate_reference_storage(storage: torch.Tensor) -> None:
    if storage.ndim != 2 or storage.shape[1] != STORAGE_BYTES_PER_TOKEN:
        raise ValueError(
            "storage must have shape "
            f"(num_tokens, {STORAGE_BYTES_PER_TOKEN}), got {storage.shape}"
        )
    if storage.dtype != torch.uint8:
        raise ValueError(f"storage must be uint8, got {storage.dtype}")


def quantize_dsv4_oscar_int2_reference(
    values: torch.Tensor,
    calibration: OscarInt2Calibration,
) -> torch.Tensor:
    """Portable Torch OSCAR encoder used for calibration and parity tests."""

    _validate_reference_values(values)
    if values.device != calibration.rotation.device:
        raise ValueError("values and calibration must share one device")

    num_tokens = values.shape[0]
    rotated = values[:, :NOPE_DIM].float() @ calibration.rotation.float()
    grouped = rotated.reshape(num_tokens, NUM_GROUPS, GROUP_SIZE)
    if calibration.clip_mode == CLIP_PER_ROW_QUANTILE:
        sorted_absolute = grouped.abs().sort(dim=-1).values
        gather_indices = calibration.clip_indices.to(torch.int64).reshape(
            1, NUM_GROUPS, 1
        )
        thresholds = torch.gather(
            sorted_absolute,
            dim=-1,
            index=gather_indices.expand(num_tokens, -1, -1),
        )
    elif calibration.clip_mode == CLIP_OFFLINE_GROUP_THRESHOLDS:
        thresholds = calibration.clip_thresholds.reshape(1, NUM_GROUPS, 1)
    else:
        raise ValueError(f"unsupported OSCAR clip mode {calibration.clip_mode!r}")
    clipped = torch.minimum(torch.maximum(grouped, -thresholds), thresholds)

    minimum = clipped.amin(dim=-1)
    maximum = clipped.amax(dim=-1)
    scale_fp32 = torch.where(
        maximum == minimum,
        torch.ones_like(maximum),
        (maximum - minimum) / float(INT2_MAX),
    )
    scales = scale_fp32.to(torch.bfloat16)
    zeros = (-minimum / scales.float()).to(torch.bfloat16)
    codes = torch.floor(
        clipped / scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1) + 0.5
    ).clamp(INT2_MIN, INT2_MAX)
    codes = codes.to(torch.uint8)
    packed = (
        codes[..., 0::4]
        | (codes[..., 1::4] << 2)
        | (codes[..., 2::4] << 4)
        | (codes[..., 3::4] << 6)
    ).reshape(num_tokens, PACKED_NOPE_BYTES)

    storage = torch.zeros(
        (num_tokens, STORAGE_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=values.device,
    )
    storage[:, :PACKED_NOPE_BYTES].copy_(packed)
    storage[:, ROPE_OFFSET_BYTES:SCALE_ZERO_OFFSET_BYTES].view(torch.bfloat16).copy_(
        values[:, NOPE_DIM:]
    )
    metadata = storage[:, SCALE_ZERO_OFFSET_BYTES:LOGICAL_BYTES_PER_TOKEN].view(
        torch.bfloat16
    )
    metadata[:, 0::2].copy_(scales)
    metadata[:, 1::2].copy_(zeros)
    return storage


def dequantize_dsv4_oscar_int2_reference(storage: torch.Tensor) -> torch.Tensor:
    """Portable Torch decoder returning rotated no-PE plus exact RoPE."""

    _validate_reference_storage(storage)
    num_tokens = storage.shape[0]
    packed = storage[:, :PACKED_NOPE_BYTES].reshape(
        num_tokens, NUM_GROUPS, PACKED_GROUP_BYTES
    )
    codes = torch.empty(
        (num_tokens, NUM_GROUPS, GROUP_SIZE),
        dtype=torch.uint8,
        device=storage.device,
    )
    codes[..., 0::4] = packed & 0x03
    codes[..., 1::4] = (packed >> 2) & 0x03
    codes[..., 2::4] = (packed >> 4) & 0x03
    codes[..., 3::4] = (packed >> 6) & 0x03
    metadata = storage[:, SCALE_ZERO_OFFSET_BYTES:LOGICAL_BYTES_PER_TOKEN].view(
        torch.bfloat16
    )
    scales = metadata[:, 0::2].float()
    zeros = metadata[:, 1::2].float()

    output = torch.empty(
        (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=storage.device
    )
    output[:, :NOPE_DIM] = (
        ((codes.float() - zeros.unsqueeze(-1)) * scales.unsqueeze(-1))
        .reshape(num_tokens, NOPE_DIM)
        .to(torch.bfloat16)
    )
    output[:, NOPE_DIM:] = storage[:, ROPE_OFFSET_BYTES:SCALE_ZERO_OFFSET_BYTES].view(
        torch.bfloat16
    )
    return output


def scatter_dsv4_oscar_int2_reference_paged(
    values: torch.Tensor,
    calibration: OscarInt2Calibration,
    storage: torch.Tensor,
    locations: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """CPU-oriented reference scatter with production page addressing."""

    _validate_reference_values(values)
    _validate_reference_paged_storage(storage, page_size)
    if locations.ndim != 1 or locations.numel() != values.shape[0]:
        raise ValueError("locations must contain one entry per input row")
    if locations.dtype not in (torch.int32, torch.int64):
        raise ValueError("locations must be int32 or int64")
    if values.device != storage.device or locations.device != storage.device:
        raise ValueError("values, storage, and locations must share one device")

    encoded = quantize_dsv4_oscar_int2_reference(values, calibration)
    capacity = storage.shape[0] * page_size
    for input_row, location in enumerate(locations.tolist()):
        if location < 0:
            continue
        if location >= capacity:
            raise ValueError(f"location {location} exceeds cache capacity {capacity}")
        page, in_page = divmod(location, page_size)
        start = in_page * STORAGE_BYTES_PER_TOKEN
        storage[page, start : start + LOGICAL_BYTES_PER_TOKEN].copy_(
            encoded[input_row, :LOGICAL_BYTES_PER_TOKEN]
        )


def _validate_cuda_values(values: torch.Tensor) -> None:
    _validate_reference_values(values)
    if not values.is_cuda:
        raise ValueError("the OSCAR INT2 fused writer requires a CUDA tensor")
    if values.stride(1) != 1:
        raise ValueError("values must be contiguous along the head dimension")
    capability = torch.cuda.get_device_capability(values.device)
    if capability != (8, 6):
        raise RuntimeError(
            "the prototype is fail-closed to NVIDIA SM86, "
            f"got compute capability {capability}"
        )


def _validate_reference_paged_storage(storage: torch.Tensor, page_size: int) -> None:
    expected_page_bytes = oscar_int2_page_bytes(page_size)
    if storage.ndim != 2 or storage.shape[1] != expected_page_bytes:
        raise ValueError(
            "paged storage must have shape "
            f"(num_pages, {expected_page_bytes}), got {tuple(storage.shape)}"
        )
    if storage.dtype != torch.uint8:
        raise ValueError(f"paged storage must use uint8 bytes, got {storage.dtype}")
    if storage.stride(1) != 1 or storage.stride(0) < expected_page_bytes:
        raise ValueError("paged storage must have contiguous byte rows")
    if storage.stride(0) % 2 or storage.storage_offset() % 2:
        raise ValueError("paged storage must be BF16 aligned")


def _validate_cuda_paged_storage(storage: torch.Tensor, page_size: int) -> None:
    _validate_reference_paged_storage(storage, page_size)
    if not storage.is_cuda:
        raise ValueError("OSCAR INT2 paged storage must be a CUDA tensor")


def _validate_cuda_locations(locations: torch.Tensor, num_tokens: int) -> None:
    if not locations.is_cuda or locations.ndim != 1:
        raise ValueError("locations must be a one-dimensional CUDA tensor")
    if locations.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"locations must be int32 or int64, got {locations.dtype}")
    if locations.numel() != num_tokens:
        raise ValueError(f"expected {num_tokens} locations, got {locations.numel()}")


def _validate_cuda_output(output: torch.Tensor, num_tokens: int) -> None:
    if not output.is_cuda:
        raise ValueError("decoded output must be a CUDA tensor")
    if output.shape != (num_tokens, HEAD_DIM):
        raise ValueError(
            f"output must have shape ({num_tokens}, {HEAD_DIM}), got {output.shape}"
        )
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output must be BF16, got {output.dtype}")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")


def _validate_cuda_calibration(
    calibration: OscarInt2Calibration, device: torch.device
) -> None:
    calibration_devices = {
        calibration.rotation.device,
        calibration.clip_thresholds.device,
        calibration.clip_ratios.device,
        calibration.clip_indices.device,
    }
    if calibration_devices != {device}:
        raise ValueError("values/storage and OSCAR calibration must share one device")
    # Full numerical validation has already happened in the setup-only factory.
    # These metadata-only checks are safe during graph capture.
    if calibration.rotation.shape != (NOPE_DIM, NOPE_DIM):
        raise ValueError("invalid OSCAR rotation shape")
    if calibration.rotation.dtype != torch.bfloat16:
        raise ValueError("invalid OSCAR rotation dtype")
    if not calibration.rotation.is_contiguous():
        raise ValueError("OSCAR rotation must remain contiguous")
    if calibration.clip_thresholds.shape != (NUM_GROUPS,):
        raise ValueError("invalid OSCAR clip-threshold shape")
    if calibration.clip_thresholds.dtype != torch.float32:
        raise ValueError("invalid OSCAR clip-threshold dtype")
    if not calibration.clip_thresholds.is_contiguous():
        raise ValueError("OSCAR clip thresholds must remain contiguous")
    if calibration.domain != CALIBRATION_DOMAIN:
        raise ValueError(
            f"OSCAR calibration domain must be {CALIBRATION_DOMAIN!r}, "
            f"got {calibration.domain!r}"
        )
    if not 0 <= calibration.layer_id <= 42:
        raise ValueError("invalid OSCAR calibration layer_id")
    if calibration.clip_ratios.shape != (NUM_GROUPS,):
        raise ValueError("invalid OSCAR calibration clip-ratio shape")
    if calibration.clip_ratios.dtype != torch.float32:
        raise ValueError("invalid OSCAR calibration clip-ratio dtype")
    if calibration.clip_indices.shape != (NUM_GROUPS,):
        raise ValueError("invalid OSCAR calibration clip-index shape")
    if calibration.clip_indices.dtype != torch.int16:
        raise ValueError("invalid OSCAR calibration clip-index dtype")
    if (
        not calibration.clip_ratios.is_contiguous()
        or not calibration.clip_indices.is_contiguous()
    ):
        raise ValueError("OSCAR clip ratios/indices must remain contiguous")
    if not calibration.clip_provenance:
        raise ValueError("OSCAR calibration has no clip provenance")
    if calibration.clip_mode not in (
        CLIP_PER_ROW_QUANTILE,
        CLIP_OFFLINE_GROUP_THRESHOLDS,
    ):
        raise ValueError("invalid OSCAR calibration clip_mode")


def _validate_cuda_write_mask(
    write_mask: torch.Tensor | None,
    *,
    num_tokens: int,
    device: torch.device,
) -> None:
    if write_mask is None:
        return
    if not write_mask.is_cuda or write_mask.device != device:
        raise ValueError("write_mask must be on the same CUDA device as values")
    if write_mask.ndim != 1 or write_mask.numel() != num_tokens:
        raise ValueError(f"write_mask must have shape ({num_tokens},)")
    if write_mask.dtype not in (torch.bool, torch.uint8):
        raise ValueError("write_mask must use bool or uint8 elements")
    if not write_mask.is_contiguous():
        raise ValueError("write_mask must be contiguous")


def _validate_shared_latent_transform(
    values: torch.Tensor,
    output: torch.Tensor,
    calibration: OscarInt2Calibration,
) -> None:
    if not values.is_cuda or not output.is_cuda:
        raise ValueError("OSCAR shared-latent transforms require CUDA tensors")
    if values.ndim != 2 or values.shape[1] != NOPE_DIM:
        raise ValueError(
            f"shared latent must have shape (num_rows, {NOPE_DIM}), got {values.shape}"
        )
    if output.shape != values.shape:
        raise ValueError(
            f"transform output must have shape {tuple(values.shape)}, got {output.shape}"
        )
    if values.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("OSCAR shared-latent transforms require BF16 input/output")
    if not values.is_contiguous() or not output.is_contiguous():
        raise ValueError("OSCAR shared-latent transform tensors must be contiguous")
    if values.device != output.device:
        raise ValueError("OSCAR shared-latent input/output must share one device")
    _validate_cuda_calibration(calibration, values.device)
    capability = torch.cuda.get_device_capability(values.device)
    if capability != (8, 6):
        raise RuntimeError(
            "the prototype is fail-closed to NVIDIA SM86, "
            f"got compute capability {capability}"
        )


def rotate_dsv4_oscar_query_shared_latent(
    query_nope: torch.Tensor,
    calibration: OscarInt2Calibration,
    output: torch.Tensor,
) -> None:
    """Apply the cache's exact ``q_nope @ R`` transform into ``output``.

    ``output`` is mandatory so the operation is safe after its BF16 GEMM has
    been warmed and captured in a CUDA graph.  Applying this same rotation to
    queries is required to preserve no-PE attention dot products.
    """

    _validate_shared_latent_transform(query_nope, output, calibration)
    torch.mm(query_nope, calibration.rotation, out=output)


def restore_dsv4_oscar_attention_output_shared_latent(
    rotated_attention_output: torch.Tensor,
    calibration: OscarInt2Calibration,
    output: torch.Tensor,
) -> None:
    """Apply ``o_rot @ R.T`` after attention's weighted latent sum.

    This inverse is mandatory for DeepSeek V4's shared K/value latent before
    the normal output projection.  This prototype intentionally has no flag
    that silently assumes the inverse was absorbed into model weights.
    """

    _validate_shared_latent_transform(rotated_attention_output, output, calibration)
    torch.mm(rotated_attention_output, calibration.rotation.T, out=output)


def _validate_full_head_transform(
    values: torch.Tensor,
    output: torch.Tensor,
    calibration: OscarInt2Calibration,
) -> None:
    if not values.is_cuda or not output.is_cuda:
        raise ValueError("OSCAR full-head transforms require CUDA tensors")
    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(
            f"full heads must have shape (num_rows, {HEAD_DIM}), got {values.shape}"
        )
    if output.shape != values.shape:
        raise ValueError(
            f"full-head output must have shape {tuple(values.shape)}, got {output.shape}"
        )
    if values.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("OSCAR full-head transforms require BF16 input/output")
    if not values.is_contiguous() or not output.is_contiguous():
        raise ValueError("OSCAR full-head transform tensors must be contiguous")
    if values.device != output.device:
        raise ValueError("OSCAR full-head input/output must share one device")
    if values.data_ptr() == output.data_ptr():
        raise ValueError("OSCAR full-head transforms do not support in-place output")
    _validate_cuda_calibration(calibration, values.device)
    capability = torch.cuda.get_device_capability(values.device)
    if capability != (8, 6):
        raise RuntimeError(
            "the prototype is fail-closed to NVIDIA SM86, "
            f"got compute capability {capability}"
        )


def rotate_dsv4_oscar_full_head_shared_latent(
    values: torch.Tensor,
    calibration: OscarInt2Calibration,
    output: torch.Tensor,
) -> None:
    """Apply ``x_nope @ R`` and copy exact RoPE for contiguous BF16 heads."""

    _validate_full_head_transform(values, output, calibration)
    if values.shape[0] == 0:
        return
    grid = (triton.cdiv(values.shape[0], _ROTATE_BLOCK_TOKENS), NUM_GROUPS)
    _transform_dsv4_oscar_full_head_kernel[grid](
        values,
        calibration.rotation,
        output,
        values.shape[0],
        values.stride(0),
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        output.stride(0),
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        group_size=GROUP_SIZE,
        rotation_input_tile=_ROTATION_INPUT_TILE,
        block_tokens=_ROTATE_BLOCK_TOKENS,
        inverse=False,
        num_warps=8,
        num_stages=1,
    )


def restore_dsv4_oscar_full_head_attention_output(
    rotated_values: torch.Tensor,
    calibration: OscarInt2Calibration,
    output: torch.Tensor,
) -> None:
    """Apply ``o_nope @ R.T`` and copy exact RoPE for contiguous BF16 heads."""

    _validate_full_head_transform(rotated_values, output, calibration)
    if rotated_values.shape[0] == 0:
        return
    grid = (
        triton.cdiv(rotated_values.shape[0], _ROTATE_BLOCK_TOKENS),
        NUM_GROUPS,
    )
    _transform_dsv4_oscar_full_head_kernel[grid](
        rotated_values,
        calibration.rotation,
        output,
        rotated_values.shape[0],
        rotated_values.stride(0),
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        output.stride(0),
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        group_size=GROUP_SIZE,
        rotation_input_tile=_ROTATION_INPUT_TILE,
        block_tokens=_ROTATE_BLOCK_TOKENS,
        inverse=True,
        num_warps=8,
        num_stages=1,
    )


def quantize_dsv4_oscar_int2_cache_paged(
    values: torch.Tensor,
    calibration: OscarInt2Calibration,
    storage: torch.Tensor,
    locations: torch.Tensor,
    *,
    page_size: int,
    write_mask: torch.Tensor | None = None,
) -> None:
    """Fused rotate, clip, asymmetric-INT2 pack, and paged scatter on SM86.

    The operation is allocation-free.  ``storage`` and ``locations`` must be
    caller-owned, and the specialization should be warmed before capture.
    Negative or out-of-capacity locations do not modify the cache.
    """

    _validate_cuda_values(values)
    _validate_cuda_paged_storage(storage, page_size)
    _validate_cuda_locations(locations, values.shape[0])
    _validate_cuda_calibration(calibration, values.device)
    _validate_cuda_write_mask(
        write_mask,
        num_tokens=values.shape[0],
        device=values.device,
    )
    if values.device != storage.device or locations.device != storage.device:
        raise ValueError("values, storage, and locations must share one CUDA device")
    if values.shape[0] == 0:
        return

    storage_bf16 = storage.view(torch.bfloat16)
    write_mask_pointer = locations if write_mask is None else write_mask
    # Decode and speculative-verification graph buckets contain at most a
    # handful of rows.  Specializing those calls to one row per program lets
    # the device-side write mask bypass every non-boundary row independently.
    # Larger prefill calls retain the wider tile.
    block_tokens = (
        1
        if write_mask is not None and values.shape[0] <= _ROTATE_BLOCK_TOKENS
        else _ROTATE_BLOCK_TOKENS
    )
    grid = (triton.cdiv(values.shape[0], block_tokens), NUM_GROUPS)
    _quantize_dsv4_oscar_int2_cache_paged_kernel[grid](
        values,
        calibration.rotation,
        calibration.clip_thresholds,
        calibration.clip_indices,
        storage,
        storage_bf16,
        locations,
        write_mask_pointer,
        values.shape[0],
        storage.shape[0] * page_size,
        values.stride(0),
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        storage.stride(0),
        storage_bf16.stride(0),
        page_size=page_size,
        token_bytes=STORAGE_BYTES_PER_TOKEN,
        nope_dim=NOPE_DIM,
        rotation_input_tile=_ROTATION_INPUT_TILE,
        rope_dim=ROPE_DIM,
        group_size=GROUP_SIZE,
        packed_group_bytes=PACKED_GROUP_BYTES,
        rope_offset_bytes=ROPE_OFFSET_BYTES,
        scale_zero_offset_bytes=SCALE_ZERO_OFFSET_BYTES,
        block_tokens=block_tokens,
        int2_max=INT2_MAX,
        offline_group_thresholds=(
            calibration.clip_mode == CLIP_OFFLINE_GROUP_THRESHOLDS
        ),
        use_write_mask=write_mask is not None,
        num_warps=8,
        num_stages=1,
    )


def dequantize_dsv4_oscar_int2_cache_paged(
    storage: torch.Tensor,
    locations: torch.Tensor,
    output: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Gather/dequantize rotated OSCAR INT2 rows into caller-owned BF16 output."""

    _validate_cuda_paged_storage(storage, page_size)
    _validate_cuda_locations(locations, locations.numel())
    _validate_cuda_output(output, locations.numel())
    if storage.device != locations.device or storage.device != output.device:
        raise ValueError("storage, locations, and output must share one CUDA device")
    capability = torch.cuda.get_device_capability(storage.device)
    if capability != (8, 6):
        raise RuntimeError(
            "the prototype is fail-closed to NVIDIA SM86, "
            f"got compute capability {capability}"
        )
    if locations.numel() == 0:
        return

    storage_bf16 = storage.view(torch.bfloat16)
    _dequantize_dsv4_oscar_int2_cache_paged_kernel[(locations.numel(), NUM_GROUPS)](
        storage,
        storage_bf16,
        locations,
        output,
        storage.shape[0] * page_size,
        storage.stride(0),
        storage_bf16.stride(0),
        output.stride(0),
        page_size=page_size,
        token_bytes=STORAGE_BYTES_PER_TOKEN,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        group_size=GROUP_SIZE,
        packed_group_bytes=PACKED_GROUP_BYTES,
        rope_offset_bytes=ROPE_OFFSET_BYTES,
        scale_zero_offset_bytes=SCALE_ZERO_OFFSET_BYTES,
        num_warps=2,
        num_stages=1,
    )


@triton.jit
def _transform_dsv4_oscar_full_head_kernel(
    values_ptr,
    rotation_ptr,
    output_ptr,
    num_rows,
    values_stride,
    rotation_stride_in,
    rotation_stride_out,
    output_stride,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    group_size: tl.constexpr,
    rotation_input_tile: tl.constexpr,
    block_tokens: tl.constexpr,
    inverse: tl.constexpr,
):
    row_block = tl.program_id(0)
    group_id = tl.program_id(1)
    row_offsets = row_block * block_tokens + tl.arange(0, block_tokens)
    row_mask = row_offsets < num_rows
    input_offsets = tl.arange(0, rotation_input_tile)
    group_offsets = tl.arange(0, group_size)
    output_offsets = group_id * group_size + group_offsets
    input_tile = tl.load(
        values_ptr + row_offsets[:, None] * values_stride + input_offsets[None, :],
        mask=row_mask[:, None] & (input_offsets[None, :] < nope_dim),
        other=0.0,
    )
    if inverse:
        rotation_offsets = (
            output_offsets[None, :] * rotation_stride_in
            + input_offsets[:, None] * rotation_stride_out
        )
    else:
        rotation_offsets = (
            input_offsets[:, None] * rotation_stride_in
            + output_offsets[None, :] * rotation_stride_out
        )
    rotation_tile = tl.load(
        rotation_ptr + rotation_offsets,
        mask=input_offsets[:, None] < nope_dim,
        other=0.0,
    )
    transformed = tl.dot(input_tile, rotation_tile, out_dtype=tl.float32)
    tl.store(
        output_ptr + row_offsets[:, None] * output_stride + output_offsets[None, :],
        transformed,
        mask=row_mask[:, None],
    )

    rope_offsets = tl.arange(0, rope_dim)
    rope = tl.load(
        values_ptr
        + row_offsets[:, None] * values_stride
        + nope_dim
        + rope_offsets[None, :],
        mask=row_mask[:, None] & (group_id == 0),
        other=0.0,
    )
    tl.store(
        output_ptr
        + row_offsets[:, None] * output_stride
        + nope_dim
        + rope_offsets[None, :],
        rope,
        mask=row_mask[:, None] & (group_id == 0),
    )


@triton.jit
def _quantize_dsv4_oscar_int2_cache_paged_kernel(
    values_ptr,
    rotation_ptr,
    clip_thresholds_ptr,
    clip_indices_ptr,
    storage_u8_ptr,
    storage_bf16_ptr,
    locations_ptr,
    write_mask_ptr,
    num_tokens,
    capacity,
    values_stride,
    rotation_stride_in,
    rotation_stride_out,
    storage_u8_page_stride,
    storage_bf16_page_stride,
    page_size: tl.constexpr,
    token_bytes: tl.constexpr,
    nope_dim: tl.constexpr,
    rotation_input_tile: tl.constexpr,
    rope_dim: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    rope_offset_bytes: tl.constexpr,
    scale_zero_offset_bytes: tl.constexpr,
    block_tokens: tl.constexpr,
    int2_max: tl.constexpr,
    offline_group_thresholds: tl.constexpr,
    use_write_mask: tl.constexpr,
):
    token_block = tl.program_id(0)
    group_id = tl.program_id(1)
    token_offsets = token_block * block_tokens + tl.arange(0, block_tokens)
    token_mask = token_offsets < num_tokens
    locations = tl.load(locations_ptr + token_offsets, mask=token_mask, other=-1).to(
        tl.int64
    )
    if use_write_mask:
        requested = tl.load(
            write_mask_ptr + token_offsets,
            mask=token_mask,
            other=0,
        ).to(tl.int1)
    else:
        requested = token_mask
    active = token_mask & requested & (locations >= 0) & (locations < capacity)
    safe_locations = tl.where(active, locations, 0)
    program_has_active_row = tl.sum(active.to(tl.int32), axis=0) != 0

    # This is a uniform, device-side branch.  CUDA graph replay re-reads the
    # live mask and locations, while a fully masked program avoids all value/R
    # loads, HMMA, sorting, packing, and RoPE traffic.
    if program_has_active_row:
        input_offsets = tl.arange(0, rotation_input_tile)
        group_offsets = tl.arange(0, group_size)
        input_tile = tl.load(
            values_ptr
            + token_offsets[:, None] * values_stride
            + input_offsets[None, :],
            mask=active[:, None] & (input_offsets[None, :] < nope_dim),
            other=0.0,
        )
        rotation_tile = tl.load(
            rotation_ptr
            + input_offsets[:, None] * rotation_stride_in
            + (group_id * group_size + group_offsets[None, :]) * rotation_stride_out,
            mask=input_offsets[:, None] < nope_dim,
            other=0.0,
        )
        rotated = tl.dot(input_tile, rotation_tile, out_dtype=tl.float32)
        if offline_group_thresholds:
            threshold = tl.load(clip_thresholds_ptr + group_id)
            clipped = tl.minimum(tl.maximum(rotated, -threshold), threshold)
        else:
            sorted_absolute = tl.sort(tl.abs(rotated), dim=1)
            clip_index = tl.load(clip_indices_ptr + group_id).to(tl.int32)
            selected = group_offsets[None, :] == clip_index
            threshold = tl.sum(
                tl.where(selected, sorted_absolute, 0.0),
                axis=1,
            )
            clipped = tl.minimum(
                tl.maximum(rotated, -threshold[:, None]), threshold[:, None]
            )

        minimum = tl.min(clipped, axis=1)
        maximum = tl.max(clipped, axis=1)
        scale_fp32 = tl.where(maximum == minimum, 1.0, (maximum - minimum) / int2_max)
        scale_bf16 = scale_fp32.to(tl.bfloat16)
        persisted_scale = scale_bf16.to(tl.float32)
        zero_bf16 = (-minimum / persisted_scale).to(tl.bfloat16)
        persisted_zero = zero_bf16.to(tl.float32)
        codes = tl.floor(
            clipped / persisted_scale[:, None] + persisted_zero[:, None] + 0.5
        )
        codes = tl.maximum(tl.minimum(codes, int2_max), 0.0).to(tl.uint8)

        shaped = tl.reshape(codes, (block_tokens, packed_group_bytes, 2, 2))
        even, odd = tl.split(shaped)
        code0, code2 = tl.split(even)
        code1, code3 = tl.split(odd)
        packed = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)

        page = safe_locations // page_size
        in_page = safe_locations - page * page_size
        storage_u8_base = page * storage_u8_page_stride + in_page * token_bytes
        storage_bf16_base = page * storage_bf16_page_stride + in_page * (
            token_bytes // 2
        )
        byte_offsets = tl.arange(0, packed_group_bytes)
        tl.store(
            storage_u8_ptr
            + storage_u8_base[:, None]
            + group_id * packed_group_bytes
            + byte_offsets[None, :],
            packed,
            mask=active[:, None],
        )
        metadata_base = scale_zero_offset_bytes // 2 + group_id * 2
        tl.store(
            storage_bf16_ptr + storage_bf16_base + metadata_base,
            scale_bf16,
            mask=active,
        )
        tl.store(
            storage_bf16_ptr + storage_bf16_base + metadata_base + 1,
            zero_bf16,
            mask=active,
        )

        # One group program copies the exact RoPE tail; all other groups skip it.
        rope_offsets = tl.arange(0, rope_dim)
        rope = tl.load(
            values_ptr
            + token_offsets[:, None] * values_stride
            + nope_dim
            + rope_offsets[None, :],
            mask=active[:, None] & (group_id == 0),
            other=0.0,
        )
        tl.store(
            storage_bf16_ptr
            + storage_bf16_base[:, None]
            + rope_offset_bytes // 2
            + rope_offsets[None, :],
            rope,
            mask=active[:, None] & (group_id == 0),
        )


@triton.jit
def _dequantize_dsv4_oscar_int2_cache_paged_kernel(
    storage_u8_ptr,
    storage_bf16_ptr,
    locations_ptr,
    output_ptr,
    capacity,
    storage_u8_page_stride,
    storage_bf16_page_stride,
    output_stride,
    page_size: tl.constexpr,
    token_bytes: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    rope_offset_bytes: tl.constexpr,
    scale_zero_offset_bytes: tl.constexpr,
):
    output_row = tl.program_id(0)
    group_id = tl.program_id(1)
    location = tl.load(locations_ptr + output_row).to(tl.int64)
    active = (location >= 0) & (location < capacity)
    safe_location = tl.where(active, location, 0)
    page = safe_location // page_size
    in_page = safe_location - page * page_size
    storage_u8_base = page * storage_u8_page_stride + in_page * token_bytes
    storage_bf16_base = page * storage_bf16_page_stride + in_page * (token_bytes // 2)

    byte_offsets = tl.arange(0, packed_group_bytes)
    packed = tl.load(
        storage_u8_ptr + storage_u8_base + group_id * packed_group_bytes + byte_offsets,
        mask=active,
        other=0,
    ).to(tl.uint8)
    code0 = packed & 0x03
    code1 = (packed >> 2) & 0x03
    code2 = (packed >> 4) & 0x03
    code3 = (packed >> 6) & 0x03
    metadata_base = scale_zero_offset_bytes // 2 + group_id * 2
    scale = tl.load(
        storage_bf16_ptr + storage_bf16_base + metadata_base,
        mask=active,
        other=1.0,
    ).to(tl.float32)
    zero = tl.load(
        storage_bf16_ptr + storage_bf16_base + metadata_base + 1,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    value0 = (code0.to(tl.float32) - zero) * scale
    value1 = (code1.to(tl.float32) - zero) * scale
    value2 = (code2.to(tl.float32) - zero) * scale
    value3 = (code3.to(tl.float32) - zero) * scale

    output_group_base = output_row * output_stride + group_id * group_size
    tl.store(
        output_ptr + output_group_base + byte_offsets * 4,
        value0,
        mask=active,
    )
    tl.store(
        output_ptr + output_group_base + byte_offsets * 4 + 1,
        value1,
        mask=active,
    )
    tl.store(
        output_ptr + output_group_base + byte_offsets * 4 + 2,
        value2,
        mask=active,
    )
    tl.store(
        output_ptr + output_group_base + byte_offsets * 4 + 3,
        value3,
        mask=active,
    )

    rope_offsets = tl.arange(0, rope_dim)
    rope = tl.load(
        storage_bf16_ptr + storage_bf16_base + rope_offset_bytes // 2 + rope_offsets,
        mask=active & (group_id == 0),
        other=0.0,
    )
    tl.store(
        output_ptr + output_row * output_stride + nope_dim + rope_offsets,
        rope,
        mask=active & (group_id == 0),
    )
