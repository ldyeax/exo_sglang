"""Calibrated OSCAR-INT2 storage and scorer for the DSV4 C4 indexer.

This is the admitted exact-SM86 C4 implementation selected by the Oscar-only
runtime pool.  It never substitutes an identity transform or a generic cache
format when calibration is absent: every writer and scorer call requires a
validated, layer-bound 128 x 128 OSCAR rotation.

The frozen page format is ``oscar-int2-c4-asym-c128-fp32-adjacent4-v1``.  A
64-token production page is laid out as follows::

    [0, 2048)       64 tokens x 32 code bytes
    [2048, 2560)    64 tokens x one FP32 (scale, zero) pair

Codes are unsigned affine INT2 values in ``[0, 3]``.  Each byte stores four
adjacent dimensions: ``d + 0`` in bits 0..1, ``d + 1`` in bits 2..3,
``d + 2`` in bits 4..5, and ``d + 3`` in bits 6..7.  Metadata is interleaved
``scale, zero`` for the complete rotated C128 row.  FP32 metadata is
intentional for this first admitted scorer format: it removes metadata-rounding
ambiguity while costing only eight bytes per token.

The writer fuses ``key @ R``, calibrated clipping, affine quantization,
adjacent-4 packing, and paged scatter.  The scorer computes ``query @ R`` with
the same calibration, decodes each key page in registers, and performs the C4
ReLU/head-weight reduction.  It never materializes a cache-sized dequantized
tensor.  CUDA APIs require caller-owned cache/output tensors and are safe to
replay after their Triton specializations have been warmed before capture.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    ARTIFACT_ALGORITHM,
    ARTIFACT_CLIP_SEMANTICS,
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    C4_ROTATION_COMPOSITION,
    C4_ROTATION_OBJECTIVE,
    CLIP_PER_ROW_QUANTILE,
    CONSUMER_SCOPE,
    OSCAR_CLIP_SOURCE_SHA256,
    OSCAR_ROTATION_SOURCE_SHA256,
    OSCAR_SOURCE_COMMIT,
    SHARED_LATENT_SOURCE,
    SHARED_ROTATION_COMPOSITION,
    SHARED_ROTATION_OBJECTIVE,
    _validate_heldout_metrics,
    tree_sha256,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    CALIBRATION_DOMAIN as SHARED_LATENT_CALIBRATION_DOMAIN,
)

FORMAT_NAME = "oscar-int2-c4-asym-c128-fp32-adjacent4-v1"
MASKED_WRITER_EXECUTION = "device-uniform-live-mask-row-v1"
QUERY_ROTATION_EXECUTION = "once-per-query-stable-workspace-v1"
CALIBRATION_DOMAIN = "c4_scorer"
CLIP_MODE = CLIP_PER_ROW_QUANTILE
CLIP_SEMANTICS = ARTIFACT_CLIP_SEMANTICS

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
GROUP_SIZE = HEAD_DIM
NUM_GROUPS = HEAD_DIM // GROUP_SIZE
INT2_VALUES_PER_BYTE = 4
INT2_MIN = 0
INT2_MAX = 3
PACKED_GROUP_BYTES = GROUP_SIZE // INT2_VALUES_PER_BYTE
CODES_BYTES_PER_TOKEN = HEAD_DIM // INT2_VALUES_PER_BYTE
METADATA_VALUES_PER_TOKEN = NUM_GROUPS * 2
METADATA_BYTES_PER_TOKEN = METADATA_VALUES_PER_TOKEN * torch.float32.itemsize
STORAGE_BYTES_PER_TOKEN = CODES_BYTES_PER_TOKEN + METADATA_BYTES_PER_TOKEN
CODES_BYTES_PER_PAGE = PAGE_SIZE * CODES_BYTES_PER_TOKEN
METADATA_OFFSET_BYTES = CODES_BYTES_PER_PAGE
PAGE_BYTES = PAGE_SIZE * STORAGE_BYTES_PER_TOKEN

_ROTATE_BLOCK_TOKENS = 16
_TARGET_PROGRAMS = 2048
_SINGLE_QUERY_PROGRAMS = 256
_SMALL_BATCH_PROGRAMS_PER_QUERY = 64
_SMALL_BATCH_LIMIT = 8
_C4_COMPRESSION_RATIO = 4
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
class OscarInt2C4Calibration:
    """Setup-validated, layer-bound calibration for the C4 consumer domain."""

    rotation: torch.Tensor
    clip_thresholds: torch.Tensor
    clip_ratios: torch.Tensor
    clip_indices: torch.Tensor
    layer_id: int
    clip_ratio: float
    clip_index: int
    clip_mode: str
    clip_semantics: str
    clip_provenance: str
    domain: str
    storage_format: str
    max_orthogonality_error: float
    max_identity_deviation: float


def oscar_int2_c4_page_bytes(page_size: int) -> int:
    """Return the exact byte width of one plane-separated C4 cache page."""

    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        raise ValueError(f"page_size must be a positive integer, got {page_size!r}")
    return page_size * STORAGE_BYTES_PER_TOKEN


def validate_oscar_int2_c4_calibration(
    rotation: torch.Tensor,
    clip_thresholds: torch.Tensor,
    clip_ratios: torch.Tensor,
    clip_indices: torch.Tensor,
    *,
    layer_id: int,
    clip_mode: str,
    clip_provenance: str,
    orthogonality_atol: float = 2.0e-2,
) -> OscarInt2C4Calibration:
    """Validate one explicit C4 OSCAR calibration outside CUDA capture.

    Runtime rotations are BF16 because both fused transforms use Ampere BF16
    tensor cores.  The full FP32 Gram check is intentionally setup-only.  An
    exact or near-identity matrix is rejected so a placeholder cannot masquerade
    as a calibrated OSCAR artifact.
    """

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("OSCAR C4 calibration validation is forbidden in capture")
    if rotation.shape != (HEAD_DIM, HEAD_DIM):
        raise ValueError(
            f"rotation must have shape ({HEAD_DIM}, {HEAD_DIM}), "
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
    if clip_ratios.shape != (NUM_GROUPS,) or clip_ratios.dtype != torch.float32:
        raise ValueError(f"clip_ratios must be FP32 with shape ({NUM_GROUPS},)")
    if clip_indices.shape != (NUM_GROUPS,) or clip_indices.dtype != torch.int16:
        raise ValueError(f"clip_indices must be INT16 with shape ({NUM_GROUPS},)")
    if not clip_ratios.is_contiguous() or not clip_indices.is_contiguous():
        raise ValueError("clip_ratios and clip_indices must be contiguous")
    calibration_tensors = (
        rotation,
        clip_thresholds,
        clip_ratios,
        clip_indices,
    )
    if any(tensor.device != rotation.device for tensor in calibration_tensors):
        raise ValueError("all C4 calibration tensors must share one device")
    if not torch.isfinite(rotation).all().item():
        raise ValueError("rotation contains non-finite values")
    if not torch.isfinite(clip_thresholds).all().item():
        raise ValueError("clip_thresholds contains non-finite values")
    if not torch.all(clip_thresholds > 0).item():
        raise ValueError("every C4 clip threshold must be positive")
    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or not 0 <= layer_id <= 42
    ):
        raise ValueError(f"layer_id must be an integer in [0, 42], got {layer_id!r}")
    if not torch.isfinite(clip_ratios).all().item():
        raise ValueError("clip_ratios contains non-finite values")
    if not torch.all((clip_ratios > 0.0) & (clip_ratios <= 1.0)).item():
        raise ValueError("every C4 clip ratio must be in (0, 1]")
    expected_clip_indices = torch.minimum(
        torch.floor(clip_ratios.to(torch.float64) * GROUP_SIZE).to(torch.int16),
        torch.full_like(clip_indices, GROUP_SIZE - 1),
    )
    if not torch.equal(clip_indices, expected_clip_indices):
        raise ValueError("clip_indices do not match floor(clip_ratios * 128)")
    if clip_mode != CLIP_MODE:
        raise ValueError(f"clip_mode must be {CLIP_MODE!r}")
    if not isinstance(clip_provenance, str) or not clip_provenance.strip():
        raise ValueError("clip_provenance must be a non-empty calibration identifier")
    if not isinstance(orthogonality_atol, float) or orthogonality_atol <= 0.0:
        raise ValueError("orthogonality_atol must be a positive float")

    rotation_fp32 = rotation.float()
    identity = torch.eye(HEAD_DIM, dtype=torch.float32, device=rotation.device)
    max_orthogonality_error = (
        (rotation_fp32.T @ rotation_fp32 - identity).abs().amax().item()
    )
    if max_orthogonality_error > orthogonality_atol:
        raise ValueError(
            "OSCAR C4 rotation is not orthogonal within tolerance: "
            f"max_error={max_orthogonality_error:.8f}, "
            f"atol={orthogonality_atol:.8f}"
        )
    max_identity_deviation = (rotation_fp32 - identity).abs().amax().item()
    if max_identity_deviation <= _IDENTITY_REJECTION_ATOL:
        raise ValueError(
            "identity is not a calibrated OSCAR C4 rotation; no fallback is allowed"
        )
    clip_ratio = float(clip_ratios[0].item())
    clip_index = int(clip_indices[0].item())
    return OscarInt2C4Calibration(
        rotation=rotation,
        clip_thresholds=clip_thresholds,
        clip_ratios=clip_ratios,
        clip_indices=clip_indices,
        layer_id=layer_id,
        clip_ratio=clip_ratio,
        clip_index=clip_index,
        clip_mode=clip_mode,
        clip_semantics=CLIP_SEMANTICS,
        clip_provenance=clip_provenance,
        domain=CALIBRATION_DOMAIN,
        storage_format=FORMAT_NAME,
        max_orthogonality_error=max_orthogonality_error,
        max_identity_deviation=max_identity_deviation,
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
        raise ValueError(f"OSCAR artifact field {field!r} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"OSCAR artifact field {field!r} must be a SHA-256 digest"
        ) from error
    return value


def load_dsv4_oscar_int2_c4_calibrations(
    artifact_path: str | Path,
    *,
    device: str | torch.device,
    expected_metadata: Mapping[str, str] | None = None,
) -> dict[int, OscarInt2C4Calibration]:
    """Load C4 payloads from the DSV4 OSCAR v2 artifact envelope.

    The shared calibration module reserves ``layers[i]["c4_scorer"]`` exactly
    when ``compression_ratios[i] == 4``.  This loader consumes that payload and
    deliberately rejects missing, extra, wrong-domain, or identity entries.
    The payload mirrors the shared-latent calibration fields::

        "c4_scorer": {
          "domain": "c4_scorer",
          "rotation_composition": "u-h128-pbr",
          "rotation": Tensor[128,128] FP32,
          "eigenvalues": Tensor[128] FP32,
          "clip_thresholds": Tensor[1] FP32,
          "clip_mode": "per_row_quantile",
          "clip_ratios": Tensor[1] FP32,
          "clip_indices": Tensor[1] INT16,
          "clip_semantics":
              "per_row_per_group_abs_order_stat_then_affine_u2_v1",
          "clip_calibration": {...},
          "provenance_sha256": str,
          "orthogonality_max_abs": float,
          "heldout_metrics": {...},
        }
    """

    path = Path(artifact_path)
    if not path.is_absolute():
        raise ValueError("OSCAR calibration artifact path must be absolute")
    if not path.is_file():
        raise FileNotFoundError(f"OSCAR calibration artifact does not exist: {path}")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("OSCAR artifacts must be loaded before CUDA capture")

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
        raise ValueError("artifact is not the calibrated OSCAR algorithm")
    _require_mapping(artifact.get("quantization"), field="quantization")
    declared_artifact_digest = _validate_sha256(
        artifact.get("artifact_provenance_sha256"),
        field="artifact_provenance_sha256",
    )
    digest_payload = {
        key: value
        for key, value in artifact.items()
        if key != "artifact_provenance_sha256"
    }
    if tree_sha256(digest_payload) != declared_artifact_digest:
        raise ValueError("artifact_provenance_sha256 does not match artifact content")

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
        "head_dim": 512,
        "latent_dim": 448,
        "rope_dim": 64,
        "index_head_dim": HEAD_DIM,
        "index_n_heads": NUM_HEADS,
    }
    if any(model.get(key) != value for key, value in expected_model_geometry.items()):
        raise ValueError("OSCAR artifact model has incompatible DSV4 geometry")
    for hash_key in ("checkpoint_sha256", "config_sha256"):
        _validate_sha256(model.get(hash_key), field=f"model.{hash_key}")
    compression_ratios = _require_tensor(
        model.get("compression_ratios"), field="model.compression_ratios"
    )
    if compression_ratios.dtype != torch.int16 or compression_ratios.shape != (43,):
        raise ValueError("model.compression_ratios must be INT16 with shape (43,)")
    if tuple(int(value) for value in compression_ratios.tolist()) != tuple(
        _EXPECTED_COMPRESSION_RATIOS
    ):
        raise ValueError("model.compression_ratios do not match DSV4 Flash")

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
    if provenance.get("shared_latent_source") != SHARED_LATENT_SOURCE:
        raise ValueError("artifact shared latent source is not compressed history")
    if provenance.get("shared_rotation_objective") != SHARED_ROTATION_OBJECTIVE:
        raise ValueError("artifact shared rotation objective is not V/SST")
    if provenance.get("c4_rotation_objective") != C4_ROTATION_OBJECTIVE:
        raise ValueError("artifact C4 rotation objective is not K/QQT")
    if provenance.get("oscar_source_commit") != OSCAR_SOURCE_COMMIT:
        raise ValueError("artifact OSCAR source commit mismatch")
    if provenance.get("oscar_rotation_source_sha256") != OSCAR_ROTATION_SOURCE_SHA256:
        raise ValueError("artifact OSCAR rotation source hash mismatch")
    if provenance.get("oscar_clip_source_sha256") != OSCAR_CLIP_SOURCE_SHA256:
        raise ValueError("artifact OSCAR clip source hash mismatch")
    if provenance.get("consumer_scope") != CONSUMER_SCOPE:
        raise ValueError("artifact OSCAR consumer scope mismatch")
    if provenance.get("shared_rotation_composition") != SHARED_ROTATION_COMPOSITION:
        raise ValueError("artifact shared rotation composition mismatch")
    if provenance.get("c4_rotation_composition") != C4_ROTATION_COMPOSITION:
        raise ValueError("artifact C4 rotation composition mismatch")

    if expected_metadata is not None:
        metadata_sources = (model, provenance, artifact)
        for key, expected_value in expected_metadata.items():
            if not isinstance(key, str) or not isinstance(expected_value, str):
                raise TypeError("expected_metadata keys and values must be strings")
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
    calibrations: dict[int, OscarInt2C4Calibration] = {}
    c4_fields = {
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
    }
    for layer_id in sorted(_REQUIRED_LAYER_IDS):
        layer = _require_mapping(layers[layer_id], field=f"layers[{layer_id}]")
        compress_ratio = int(compression_ratios[layer_id].item())
        expected_layer_keys = {
            "layer_id",
            "compress_ratio",
            SHARED_LATENT_CALIBRATION_DOMAIN,
        }
        if compress_ratio == _C4_COMPRESSION_RATIO:
            expected_layer_keys.add(CALIBRATION_DOMAIN)
        _require_exact_keys(
            layer,
            field=f"layers[{layer_id}]",
            expected=expected_layer_keys,
        )
        if layer.get("layer_id") != layer_id:
            raise ValueError(f"OSCAR artifact layer {layer_id} has wrong layer_id")
        if layer.get("compress_ratio") != compress_ratio:
            raise ValueError(
                f"OSCAR artifact layer {layer_id} has wrong compress_ratio"
            )
        shared = _require_mapping(
            layer.get(SHARED_LATENT_CALIBRATION_DOMAIN),
            field=f"layers[{layer_id}].{SHARED_LATENT_CALIBRATION_DOMAIN}",
        )
        if (
            shared.get("domain") != SHARED_LATENT_CALIBRATION_DOMAIN
            or shared.get("rotation_objective") != SHARED_ROTATION_OBJECTIVE
            or shared.get("rotation_composition") != SHARED_ROTATION_COMPOSITION
        ):
            raise ValueError(
                f"layer {layer_id} shared calibration contract is incompatible"
            )
        if compress_ratio != _C4_COMPRESSION_RATIO:
            continue

        c4 = _require_mapping(
            layer.get(CALIBRATION_DOMAIN),
            field=f"layers[{layer_id}].{CALIBRATION_DOMAIN}",
        )
        _require_exact_keys(
            c4, field=f"layers[{layer_id}].c4_scorer", expected=c4_fields
        )
        if c4.get("domain") != CALIBRATION_DOMAIN:
            raise ValueError(f"layer {layer_id} C4 payload has the wrong domain")
        if c4.get("rotation_objective") != C4_ROTATION_OBJECTIVE:
            raise ValueError(
                f"layer {layer_id} C4 payload has unsupported rotation objective"
            )
        if c4.get("rotation_composition") != C4_ROTATION_COMPOSITION:
            raise ValueError(
                f"layer {layer_id} C4 payload has unsupported rotation composition"
            )
        rotation = _require_tensor(
            c4.get("rotation"), field=f"layers[{layer_id}].c4_scorer.rotation"
        )
        eigenvalues = _require_tensor(
            c4.get("eigenvalues"), field=f"layers[{layer_id}].c4_scorer.eigenvalues"
        )
        thresholds = _require_tensor(
            c4.get("clip_thresholds"),
            field=f"layers[{layer_id}].c4_scorer.clip_thresholds",
        )
        clip_ratios = _require_tensor(
            c4.get("clip_ratios"),
            field=f"layers[{layer_id}].c4_scorer.clip_ratios",
        )
        clip_indices = _require_tensor(
            c4.get("clip_indices"),
            field=f"layers[{layer_id}].c4_scorer.clip_indices",
        )
        if rotation.dtype != torch.float32 or rotation.shape != (HEAD_DIM, HEAD_DIM):
            raise ValueError(f"layer {layer_id} C4 rotation must be FP32 [128, 128]")
        if eigenvalues.dtype != torch.float32 or eigenvalues.shape != (HEAD_DIM,):
            raise ValueError(f"layer {layer_id} C4 eigenvalues must be FP32 [128]")
        if thresholds.dtype != torch.float32 or thresholds.shape != (NUM_GROUPS,):
            raise ValueError(f"layer {layer_id} C4 thresholds must be FP32 [1]")
        if clip_ratios.dtype != torch.float32 or clip_ratios.shape != (NUM_GROUPS,):
            raise ValueError(f"layer {layer_id} C4 clip_ratios must be FP32 [1]")
        if clip_indices.dtype != torch.int16 or clip_indices.shape != (NUM_GROUPS,):
            raise ValueError(f"layer {layer_id} C4 clip_indices must be INT16 [1]")
        clip_mode = c4.get("clip_mode")
        clip_semantics = c4.get("clip_semantics")
        clip_provenance = c4.get("provenance_sha256")
        if clip_mode != CLIP_MODE:
            raise ValueError(f"layer {layer_id} clip_mode must be per_row_quantile")
        if clip_semantics != CLIP_SEMANTICS:
            raise ValueError(
                f"layer {layer_id} has unsupported clip semantics {clip_semantics!r}"
            )
        _validate_sha256(
            clip_provenance,
            field=f"layers[{layer_id}].c4_scorer.provenance_sha256",
        )
        declared_orthogonality = c4.get("orthogonality_max_abs")
        if not isinstance(declared_orthogonality, float):
            raise TypeError(f"layer {layer_id} orthogonality_max_abs must be a float")
        rotation_fp64 = rotation.to(torch.float64)
        actual_orthogonality = (
            (
                rotation_fp64.T @ rotation_fp64
                - torch.eye(HEAD_DIM, dtype=torch.float64, device=rotation.device)
            )
            .abs()
            .amax()
            .item()
        )
        if actual_orthogonality > 2.0e-5:
            raise ValueError(f"layer {layer_id} C4 rotation is not orthogonal")
        if abs(actual_orthogonality - declared_orthogonality) > 1.0e-7:
            raise ValueError(
                f"layer {layer_id} orthogonality_max_abs does not match rotation"
            )
        _require_mapping(
            c4.get("clip_calibration"),
            field=f"layers[{layer_id}].c4_scorer.clip_calibration",
        )
        _validate_heldout_metrics(
            c4.get("heldout_metrics"),
            field=f"layers[{layer_id}].c4_scorer.heldout_metrics",
        )
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
        calibrations[layer_id] = validate_oscar_int2_c4_calibration(
            runtime_rotation,
            runtime_thresholds,
            runtime_clip_ratios,
            runtime_clip_indices,
            layer_id=layer_id,
            clip_mode=clip_mode,
            clip_provenance=clip_provenance,
        )
    return calibrations


def _validate_reference_keys(keys: torch.Tensor) -> None:
    if keys.ndim != 2 or keys.shape[1] != HEAD_DIM:
        raise ValueError(f"keys must have shape (num_tokens, {HEAD_DIM})")
    if keys.dtype != torch.bfloat16:
        raise ValueError(f"keys must be BF16, got {keys.dtype}")


def _validate_runtime_calibration(
    calibration: OscarInt2C4Calibration, device: torch.device
) -> None:
    calibration_tensors = (
        calibration.rotation,
        calibration.clip_thresholds,
        calibration.clip_ratios,
        calibration.clip_indices,
    )
    if any(tensor.device != device for tensor in calibration_tensors):
        raise ValueError("scorer/writer tensors and calibration must share one device")
    if calibration.rotation.shape != (HEAD_DIM, HEAD_DIM):
        raise ValueError("invalid OSCAR C4 rotation shape")
    if (
        calibration.rotation.dtype != torch.bfloat16
        or not calibration.rotation.is_contiguous()
    ):
        raise ValueError("OSCAR C4 rotation must remain contiguous BF16")
    if calibration.clip_thresholds.shape != (NUM_GROUPS,):
        raise ValueError("invalid OSCAR C4 threshold shape")
    if (
        calibration.clip_thresholds.dtype != torch.float32
        or not calibration.clip_thresholds.is_contiguous()
    ):
        raise ValueError("OSCAR C4 thresholds must remain contiguous FP32")
    if (
        calibration.clip_ratios.shape != (NUM_GROUPS,)
        or calibration.clip_ratios.dtype != torch.float32
        or not calibration.clip_ratios.is_contiguous()
    ):
        raise ValueError("OSCAR C4 clip_ratios must remain contiguous FP32 [1]")
    if (
        calibration.clip_indices.shape != (NUM_GROUPS,)
        or calibration.clip_indices.dtype != torch.int16
        or not calibration.clip_indices.is_contiguous()
    ):
        raise ValueError("OSCAR C4 clip_indices must remain contiguous INT16 [1]")
    if calibration.domain != CALIBRATION_DOMAIN:
        raise ValueError(f"calibration domain must be {CALIBRATION_DOMAIN!r}")
    if calibration.storage_format != FORMAT_NAME:
        raise ValueError(f"calibration storage format must be {FORMAT_NAME!r}")
    if not 0 <= calibration.layer_id <= 42:
        raise ValueError("invalid OSCAR C4 layer_id")
    if calibration.clip_semantics != CLIP_SEMANTICS:
        raise ValueError("invalid OSCAR C4 clip semantics")
    if calibration.clip_mode != CLIP_MODE:
        raise ValueError("invalid OSCAR C4 clip mode")
    if not calibration.clip_provenance:
        raise ValueError("OSCAR C4 calibration has no provenance")


def quantize_oscar_int2_c4_reference(
    keys: torch.Tensor,
    calibration: OscarInt2C4Calibration,
) -> torch.Tensor:
    """Allocation-heavy CPU/Torch encoder returning token-major records."""

    _validate_reference_keys(keys)
    _validate_runtime_calibration(calibration, keys.device)
    num_tokens = keys.shape[0]
    rotated = keys.float() @ calibration.rotation.float()
    grouped = rotated.reshape(num_tokens, NUM_GROUPS, GROUP_SIZE)
    thresholds = (
        grouped.abs()
        .sort(dim=-1)
        .values[..., calibration.clip_index : calibration.clip_index + 1]
    )
    clipped = torch.minimum(torch.maximum(grouped, -thresholds), thresholds)
    minimum = clipped.amin(dim=-1)
    maximum = clipped.amax(dim=-1)
    scales = torch.where(
        maximum == minimum,
        torch.ones_like(maximum),
        (maximum - minimum) / float(INT2_MAX),
    )
    zeros = -minimum / scales
    codes = (
        torch.floor(clipped / scales.unsqueeze(-1) + zeros.unsqueeze(-1) + 0.5)
        .clamp(INT2_MIN, INT2_MAX)
        .to(torch.uint8)
    )
    packed = (
        codes[..., 0::4]
        | (codes[..., 1::4] << 2)
        | (codes[..., 2::4] << 4)
        | (codes[..., 3::4] << 6)
    ).reshape(num_tokens, CODES_BYTES_PER_TOKEN)

    storage = torch.empty(
        (num_tokens, STORAGE_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=keys.device,
    )
    storage[:, :CODES_BYTES_PER_TOKEN].copy_(packed)
    metadata = storage[:, CODES_BYTES_PER_TOKEN:].view(torch.float32)
    metadata[:, 0::2].copy_(scales)
    metadata[:, 1::2].copy_(zeros)
    return storage


def dequantize_oscar_int2_c4_reference(storage: torch.Tensor) -> torch.Tensor:
    """Decode token-major records to the BF16 rotated key coordinates."""

    if storage.dtype != torch.uint8 or storage.ndim != 2:
        raise ValueError("storage must be a rank-2 uint8 tensor")
    if storage.shape[1] != STORAGE_BYTES_PER_TOKEN or storage.stride(1) != 1:
        raise ValueError(
            f"storage must have contiguous {STORAGE_BYTES_PER_TOKEN}-byte rows"
        )
    num_tokens = storage.shape[0]
    packed = storage[:, :CODES_BYTES_PER_TOKEN].reshape(
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
    metadata = storage[:, CODES_BYTES_PER_TOKEN:].view(torch.float32)
    scales = metadata[:, 0::2]
    zeros = metadata[:, 1::2]
    decoded = (codes.float() - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    return decoded.reshape(num_tokens, HEAD_DIM).to(torch.bfloat16)


def _validate_reference_pages(storage: torch.Tensor, page_size: int) -> None:
    expected_page_bytes = oscar_int2_c4_page_bytes(page_size)
    if storage.dtype != torch.uint8 or storage.ndim != 2:
        raise ValueError("paged storage must be a rank-2 uint8 tensor")
    if storage.shape[1] != expected_page_bytes:
        raise ValueError(
            f"paged storage rows must contain {expected_page_bytes} bytes, "
            f"got {tuple(storage.shape)}"
        )
    if storage.stride(1) != 1 or storage.stride(0) < expected_page_bytes:
        raise ValueError("paged storage must have contiguous byte rows")
    if storage.stride(0) % 4 or storage.storage_offset() % 4:
        raise ValueError("paged storage must be FP32 aligned")


def pack_oscar_int2_c4_pages_reference(
    keys: torch.Tensor,
    calibration: OscarInt2C4Calibration,
) -> torch.Tensor:
    """Encode ``[pages, page_size, 128]`` keys into the physical page layout."""

    if keys.ndim != 3 or keys.shape[2] != HEAD_DIM:
        raise ValueError(f"keys must have shape [num_pages, page_size, {HEAD_DIM}]")
    if keys.dtype != torch.bfloat16:
        raise ValueError(f"keys must be BF16, got {keys.dtype}")
    page_size = keys.shape[1]
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    num_pages = keys.shape[0]
    records = quantize_oscar_int2_c4_reference(
        keys.reshape(-1, HEAD_DIM), calibration
    ).reshape(num_pages, page_size, STORAGE_BYTES_PER_TOKEN)
    page_bytes = oscar_int2_c4_page_bytes(page_size)
    storage = torch.empty(
        (num_pages, page_bytes), dtype=torch.uint8, device=keys.device
    )
    code_bytes = page_size * CODES_BYTES_PER_TOKEN
    storage[:, :code_bytes].copy_(
        records[:, :, :CODES_BYTES_PER_TOKEN].reshape(num_pages, code_bytes)
    )
    storage[:, code_bytes:].copy_(
        records[:, :, CODES_BYTES_PER_TOKEN:].reshape(num_pages, -1)
    )
    return storage


def unpack_oscar_int2_c4_pages_reference(
    storage: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Decode physical pages to BF16 rotated keys without applying ``R.T``."""

    _validate_reference_pages(storage, page_size)
    num_pages = storage.shape[0]
    code_bytes = page_size * CODES_BYTES_PER_TOKEN
    records = torch.empty(
        (num_pages, page_size, STORAGE_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=storage.device,
    )
    records[:, :, :CODES_BYTES_PER_TOKEN].copy_(
        storage[:, :code_bytes].reshape(num_pages, page_size, CODES_BYTES_PER_TOKEN)
    )
    records[:, :, CODES_BYTES_PER_TOKEN:].copy_(
        storage[:, code_bytes:].reshape(num_pages, page_size, METADATA_BYTES_PER_TOKEN)
    )
    return dequantize_oscar_int2_c4_reference(
        records.reshape(-1, STORAGE_BYTES_PER_TOKEN)
    ).reshape(num_pages, page_size, HEAD_DIM)


def scatter_oscar_int2_c4_reference_paged(
    keys: torch.Tensor,
    calibration: OscarInt2C4Calibration,
    storage: torch.Tensor,
    locations: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
) -> None:
    """Reference scatter using the same plane-separated physical addressing."""

    _validate_reference_keys(keys)
    _validate_reference_pages(storage, page_size)
    if locations.ndim != 1 or locations.numel() != keys.shape[0]:
        raise ValueError("locations must contain one entry per key")
    if locations.dtype not in (torch.int32, torch.int64):
        raise ValueError("locations must be int32 or int64")
    if keys.device != storage.device or locations.device != storage.device:
        raise ValueError("keys, storage, and locations must share one device")
    records = quantize_oscar_int2_c4_reference(keys, calibration)
    capacity = storage.shape[0] * page_size
    code_page_bytes = page_size * CODES_BYTES_PER_TOKEN
    for input_row, location in enumerate(locations.tolist()):
        if location < 0:
            continue
        if location >= capacity:
            raise ValueError(f"location {location} exceeds cache capacity {capacity}")
        page, in_page = divmod(location, page_size)
        code_start = in_page * CODES_BYTES_PER_TOKEN
        metadata_start = code_page_bytes + in_page * METADATA_BYTES_PER_TOKEN
        storage[page, code_start : code_start + CODES_BYTES_PER_TOKEN].copy_(
            records[input_row, :CODES_BYTES_PER_TOKEN]
        )
        storage[page, metadata_start : metadata_start + METADATA_BYTES_PER_TOKEN].copy_(
            records[input_row, CODES_BYTES_PER_TOKEN:]
        )


def oscar_int2_c4_paged_mqa_logits_reference(
    query: torch.Tensor,
    storage: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    calibration: OscarInt2C4Calibration,
    max_seq_len: int,
    *,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Allocation-heavy CPU/Torch oracle for the fused paged scorer."""

    _validate_reference_pages(storage, page_size)
    if query.ndim != 4:
        raise ValueError("query must be rank 4")
    batch_size = query.shape[0]
    expected_query_shape = (batch_size, 1, NUM_HEADS, HEAD_DIM)
    if query.dtype != torch.bfloat16 or tuple(query.shape) != expected_query_shape:
        raise ValueError(f"query must be BF16 with shape {expected_query_shape}")
    if tuple(weight.shape) != (batch_size, NUM_HEADS):
        raise ValueError(f"weight must have shape ({batch_size}, {NUM_HEADS})")
    if tuple(seq_lens.shape) not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("seq_lens has an invalid shape")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table has an invalid shape")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if max_seq_len > page_table.shape[1] * page_size:
        raise ValueError("max_seq_len exceeds page-table capacity")
    tensors = (storage, weight, seq_lens, page_table)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all reference tensors must share one device")
    _validate_runtime_calibration(calibration, query.device)
    if storage.shape[0] == 0:
        raise ValueError("storage must contain at least one physical page")

    decoded_pages = unpack_oscar_int2_c4_pages_reference(
        storage, page_size=page_size
    ).float()
    page_ids = page_table.to(torch.int64)
    valid_pages = (page_ids >= 0) & (page_ids < storage.shape[0])
    safe_page_ids = page_ids.clamp(0, storage.shape[0] - 1)
    gathered_keys = decoded_pages[safe_page_ids].reshape(batch_size, -1, HEAD_DIM)[
        :, :max_seq_len
    ]
    valid_tokens = valid_pages.repeat_interleave(page_size, dim=1)[:, :max_seq_len]
    rotated_query = (
        (query[:, 0].float() @ calibration.rotation.float()).to(torch.bfloat16).float()
    )
    logits = torch.bmm(gathered_keys, rotated_query.transpose(1, 2))
    reduced = (torch.relu(logits) * weight.float()[:, None, :]).sum(dim=2)
    positions = torch.arange(max_seq_len, device=query.device)[None, :]
    bounded_lengths = seq_lens.reshape(batch_size, 1).clamp(0, max_seq_len)
    return reduced.masked_fill(~valid_tokens | (positions >= bounded_lengths), 0.0)


def _require_exact_sm86(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("the OSCAR C4 indexer requires CUDA tensors")
    if torch.version.hip is not None:
        raise RuntimeError("the OSCAR C4 indexer targets NVIDIA SM86 only")
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 6):
        raise RuntimeError(
            "the OSCAR C4 indexer is fail-closed to exact NVIDIA SM86, "
            f"got compute capability {capability}"
        )


def store_oscar_int2_c4_indexer_cache(
    keys: torch.Tensor,
    storage: torch.Tensor,
    locations: torch.Tensor,
    *,
    calibration: OscarInt2C4Calibration,
    page_size: int = PAGE_SIZE,
    write_mask: torch.Tensor | None = None,
) -> None:
    """Fused ``key @ R`` + calibrated affine-INT2 paged cache write."""

    _validate_reference_keys(keys)
    if not keys.is_cuda or keys.stride(1) != 1:
        raise ValueError("keys must be CUDA BF16 with contiguous 128-value rows")
    _require_exact_sm86(keys.device)
    _validate_reference_pages(storage, page_size)
    if not storage.is_cuda:
        raise ValueError("paged storage must be a CUDA tensor")
    if (
        not locations.is_cuda
        or locations.ndim != 1
        or locations.dtype not in (torch.int32, torch.int64)
        or locations.numel() != keys.shape[0]
    ):
        raise ValueError("locations must be CUDA int32/int64 with one entry per key")
    if keys.device != storage.device or locations.device != storage.device:
        raise ValueError("keys, storage, and locations must share one CUDA device")
    _validate_runtime_calibration(calibration, keys.device)
    if write_mask is not None and (
        not write_mask.is_cuda
        or write_mask.device != keys.device
        or write_mask.ndim != 1
        or write_mask.numel() != keys.shape[0]
        or write_mask.dtype not in (torch.bool, torch.uint8)
        or not write_mask.is_contiguous()
    ):
        raise ValueError("write_mask must be contiguous CUDA bool/uint8 per key")
    if keys.shape[0] == 0:
        return

    storage_f32 = storage.view(torch.float32)
    write_mask_pointer = locations if write_mask is None else write_mask
    # Small decode/verification graph buckets use one row per program so the
    # live device mask can bypass every non-boundary C4 write independently.
    # Prefill keeps the wider tile to avoid multiplying active launch count.
    block_tokens = (
        1
        if write_mask is not None and keys.shape[0] <= _ROTATE_BLOCK_TOKENS
        else _ROTATE_BLOCK_TOKENS
    )
    grid = (triton.cdiv(keys.shape[0], block_tokens), NUM_GROUPS)
    _store_oscar_int2_c4_indexer_cache_kernel[grid](
        keys,
        calibration.rotation,
        storage,
        storage_f32,
        locations,
        write_mask_pointer,
        keys.shape[0],
        storage.shape[0] * page_size,
        keys.stride(0),
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        storage.stride(0),
        storage_f32.stride(0),
        page_size=page_size,
        head_dim=HEAD_DIM,
        group_size=GROUP_SIZE,
        packed_group_bytes=PACKED_GROUP_BYTES,
        codes_bytes_per_token=CODES_BYTES_PER_TOKEN,
        metadata_values_per_token=METADATA_VALUES_PER_TOKEN,
        block_tokens=block_tokens,
        int2_max=INT2_MAX,
        clip_index=calibration.clip_index,
        use_write_mask=write_mask is not None,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _store_oscar_int2_c4_indexer_cache_kernel(
    keys_ptr,
    rotation_ptr,
    storage_u8_ptr,
    storage_f32_ptr,
    locations_ptr,
    write_mask_ptr,
    num_tokens,
    capacity,
    keys_stride,
    rotation_stride_in,
    rotation_stride_out,
    storage_u8_page_stride,
    storage_f32_page_stride,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    codes_bytes_per_token: tl.constexpr,
    metadata_values_per_token: tl.constexpr,
    block_tokens: tl.constexpr,
    int2_max: tl.constexpr,
    clip_index: tl.constexpr,
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
            write_mask_ptr + token_offsets, mask=token_mask, other=0
        ).to(tl.int1)
    else:
        requested = token_mask
    active = token_mask & requested & (locations >= 0) & (locations < capacity)
    safe_locations = tl.where(active, locations, 0)
    program_has_active_row = tl.sum(active.to(tl.int32), axis=0) != 0

    # Keep this predicate on the device: graph replay must observe updated
    # boundary masks without a host sync.  A fully masked program performs no
    # key/R loads, HMMA, order-statistic clipping, or packing.
    if program_has_active_row:
        input_offsets = tl.arange(0, head_dim)
        group_offsets = tl.arange(0, group_size)
        input_tile = tl.load(
            keys_ptr + token_offsets[:, None] * keys_stride + input_offsets[None, :],
            mask=active[:, None],
            other=0.0,
        )
        rotation_tile = tl.load(
            rotation_ptr
            + input_offsets[:, None] * rotation_stride_in
            + (group_id * group_size + group_offsets[None, :]) * rotation_stride_out
        )
        rotated = tl.dot(input_tile, rotation_tile, out_dtype=tl.float32)
        sorted_absolute = tl.sort(tl.abs(rotated), dim=1)
        selected = group_offsets[None, :] == clip_index
        threshold = tl.sum(tl.where(selected, sorted_absolute, 0.0), axis=1)
        clipped = tl.minimum(
            tl.maximum(rotated, -threshold[:, None]), threshold[:, None]
        )
        minimum = tl.min(clipped, axis=1)
        maximum = tl.max(clipped, axis=1)
        scale = tl.where(maximum == minimum, 1.0, (maximum - minimum) / int2_max)
        zero = -minimum / scale
        codes = tl.floor(clipped / scale[:, None] + zero[:, None] + 0.5)
        codes = tl.maximum(tl.minimum(codes, int2_max), 0.0).to(tl.uint8)

        shaped = tl.reshape(codes, (block_tokens, packed_group_bytes, 2, 2))
        even, odd = tl.split(shaped)
        code0, code2 = tl.split(even)
        code1, code3 = tl.split(odd)
        packed = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)

        page = safe_locations // page_size
        in_page = safe_locations - page * page_size
        code_base = (
            page * storage_u8_page_stride
            + in_page * codes_bytes_per_token
            + group_id * packed_group_bytes
        )
        byte_offsets = tl.arange(0, packed_group_bytes)
        tl.store(
            storage_u8_ptr + code_base[:, None] + byte_offsets[None, :],
            packed,
            mask=active[:, None],
        )
        metadata_base = (
            page * storage_f32_page_stride
            + (page_size * codes_bytes_per_token) // 4
            + in_page * metadata_values_per_token
            + group_id * 2
        )
        tl.store(storage_f32_ptr + metadata_base, scale, mask=active)
        tl.store(storage_f32_ptr + metadata_base + 1, zero, mask=active)


def _programs_per_query(batch_size: int, max_pages: int) -> int:
    if batch_size == 1:
        target = _SINGLE_QUERY_PROGRAMS
    elif batch_size <= _SMALL_BATCH_LIMIT:
        target = _SMALL_BATCH_PROGRAMS_PER_QUERY
    else:
        target = triton.cdiv(_TARGET_PROGRAMS, batch_size)
    return max(1, min(max_pages, target))


@triton.jit
def _rotate_oscar_int2_c4_query_kernel(
    query_ptr,
    rotation_ptr,
    output_ptr,
    rotation_stride_in,
    rotation_stride_out,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Rotate one query row exactly once before page-parallel scoring."""

    batch_idx = tl.program_id(0)
    head_offsets = tl.arange(0, num_heads)
    dim_offsets = tl.arange(0, head_dim)
    query_offsets = (
        batch_idx * num_heads * head_dim
        + head_offsets[:, None] * head_dim
        + dim_offsets[None, :]
    )
    query = tl.load(query_ptr + query_offsets)
    rotation = tl.load(
        rotation_ptr
        + dim_offsets[:, None] * rotation_stride_in
        + dim_offsets[None, :] * rotation_stride_out
    )
    rotated_query = tl.dot(query, rotation, out_dtype=tl.float32).to(tl.bfloat16)
    tl.store(output_ptr + query_offsets, rotated_query)


def oscar_int2_c4_paged_mqa_logits_triton(
    query: torch.Tensor,
    storage: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
    *,
    calibration: OscarInt2C4Calibration,
    out: torch.Tensor,
    rotated_query_out: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Rotate once, then fuse OSCAR page decode and C4 scoring.

    ``out`` and ``rotated_query_out`` are mandatory caller-owned workspaces.
    Warm this exact shape/specialization eagerly before CUDA graph capture;
    thereafter the wrapper performs no device allocation.  The scorer's page
    programs consume the one rotated query instead of redundantly evaluating
    ``query @ R`` once per program.
    """

    del deep_gemm_metadata

    if query.ndim != 4:
        raise ValueError("query must be rank 4")
    batch_size = query.shape[0]
    if batch_size == 0:
        raise ValueError("query batch must not be empty")
    expected_query_shape = (batch_size, 1, NUM_HEADS, HEAD_DIM)
    if query.dtype != torch.bfloat16 or tuple(query.shape) != expected_query_shape:
        raise ValueError(f"query must be BF16 with shape {expected_query_shape}")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous")
    _require_exact_sm86(query.device)
    _validate_reference_pages(storage, page_size)
    if not storage.is_cuda or storage.shape[0] == 0:
        raise ValueError("storage must contain at least one CUDA physical page")
    if weight.dtype != torch.float32 or tuple(weight.shape) != (batch_size, NUM_HEADS):
        raise ValueError(f"weight must be FP32 with shape ({batch_size}, {NUM_HEADS})")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if (
        seq_lens.dtype != torch.int32
        or tuple(seq_lens.shape) not in {(batch_size,), (batch_size, 1)}
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous INT32 [B] or [B, 1]")
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or page_table.shape[0] != batch_size
        or page_table.stride(1) != 1
    ):
        raise ValueError("page_table must be INT32 [B, max_pages] with contiguous rows")
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len <= 0
    ):
        raise ValueError("max_seq_len must be a positive integer")
    if max_seq_len > page_table.shape[1] * page_size:
        raise ValueError("max_seq_len exceeds page-table capacity")
    if (
        out.dtype != torch.float32
        or tuple(out.shape) != (batch_size, max_seq_len)
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous FP32 [{batch_size}, {max_seq_len}]")
    expected_rotated_shape = (batch_size, NUM_HEADS, HEAD_DIM)
    if (
        rotated_query_out.dtype != torch.bfloat16
        or tuple(rotated_query_out.shape) != expected_rotated_shape
        or not rotated_query_out.is_contiguous()
    ):
        raise ValueError(
            "rotated_query_out must be contiguous BF16 with shape "
            f"{expected_rotated_shape}"
        )
    if rotated_query_out.data_ptr() == query.data_ptr():
        raise ValueError("rotated_query_out must not alias query")
    tensors = (storage, weight, seq_lens, page_table, out, rotated_query_out)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all scorer tensors must share one CUDA device")
    _validate_runtime_calibration(calibration, query.device)
    if clean_logits:
        out.zero_()

    _rotate_oscar_int2_c4_query_kernel[(batch_size,)](
        query,
        calibration.rotation,
        rotated_query_out,
        calibration.rotation.stride(0),
        calibration.rotation.stride(1),
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_warps=8,
        num_stages=1,
    )

    storage_f32 = storage.view(torch.float32)
    max_pages = triton.cdiv(max_seq_len, page_size)
    programs_per_query = _programs_per_query(batch_size, max_pages)
    _oscar_int2_c4_paged_mqa_logits_kernel[(batch_size, programs_per_query)](
        rotated_query_out,
        storage,
        storage_f32,
        weight,
        seq_lens.view(batch_size),
        page_table,
        out,
        storage.stride(0),
        storage_f32.stride(0),
        page_table.stride(0),
        max_seq_len=max_seq_len,
        page_table_width=page_table.shape[1],
        num_cache_pages=storage.shape[0],
        programs_per_query=programs_per_query,
        page_size=page_size,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        group_size=GROUP_SIZE,
        packed_bytes_per_token=CODES_BYTES_PER_TOKEN,
        metadata_values_per_token=METADATA_VALUES_PER_TOKEN,
        num_warps=8,
        num_stages=1,
    )
    return out


@triton.jit
def _oscar_int2_c4_paged_mqa_logits_kernel(
    rotated_query_ptr,
    storage_u8_ptr,
    storage_f32_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    output_ptr,
    storage_u8_page_stride,
    storage_f32_page_stride,
    page_table_stride,
    max_seq_len: tl.constexpr,
    page_table_width: tl.constexpr,
    num_cache_pages: tl.constexpr,
    programs_per_query: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    packed_bytes_per_token: tl.constexpr,
    metadata_values_per_token: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    program_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), max_seq_len)
    active_pages = tl.minimum(tl.cdiv(bounded_seq_len, page_size), page_table_width)

    if program_idx < active_pages:
        head_offsets = tl.arange(0, num_heads)
        dim_offsets = tl.arange(0, head_dim)
        token_offsets = tl.arange(0, page_size)
        packed_offsets = tl.arange(0, packed_bytes_per_token)
        query_offsets = (
            batch_idx * num_heads * head_dim
            + head_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        rotated_query = tl.load(rotated_query_ptr + query_offsets)
        weights = tl.load(weight_ptr + batch_idx * num_heads + head_offsets).to(
            tl.float32
        )

        for page_slot in tl.range(program_idx, active_pages, programs_per_query):
            page_id = tl.load(
                page_table_ptr + batch_idx * page_table_stride + page_slot
            )
            page_is_valid = (page_id >= 0) & (page_id < num_cache_pages)
            safe_page_id = tl.where(page_is_valid, page_id, 0)
            packed = tl.load(
                storage_u8_ptr
                + safe_page_id * storage_u8_page_stride
                + token_offsets[:, None] * packed_bytes_per_token
                + packed_offsets[None, :],
                mask=page_is_valid,
                other=0,
            ).to(tl.uint8)
            code0 = packed & 0x03
            code1 = (packed >> 2) & 0x03
            code2 = (packed >> 4) & 0x03
            code3 = (packed >> 6) & 0x03
            even_codes = tl.interleave(code0, code2)
            odd_codes = tl.interleave(code1, code3)
            codes = tl.interleave(even_codes, odd_codes)

            metadata_base = (
                safe_page_id * storage_f32_page_stride
                + (page_size * packed_bytes_per_token) // 4
                + token_offsets * metadata_values_per_token
            )
            scale0 = tl.load(
                storage_f32_ptr + metadata_base, mask=page_is_valid, other=1.0
            )
            zero0 = tl.load(
                storage_f32_ptr + metadata_base + 1, mask=page_is_valid, other=0.0
            )
            keys = ((codes.to(tl.float32) - zero0[:, None]) * scale0[:, None]).to(
                tl.bfloat16
            )
            logits = tl.dot(keys, tl.trans(rotated_query), out_dtype=tl.float32)
            reduced = tl.sum(tl.maximum(logits, 0.0) * weights[None, :], axis=1)
            reduced = tl.where(page_is_valid, reduced, 0.0)
            output_positions = page_slot * page_size + token_offsets
            tl.store(
                output_ptr + batch_idx * max_seq_len + output_positions,
                reduced,
                mask=output_positions < bounded_seq_len,
            )


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "CALIBRATION_DOMAIN",
    "CLIP_MODE",
    "CLIP_SEMANTICS",
    "CODES_BYTES_PER_PAGE",
    "CODES_BYTES_PER_TOKEN",
    "FORMAT_NAME",
    "GROUP_SIZE",
    "HEAD_DIM",
    "INT2_MAX",
    "INT2_MIN",
    "MASKED_WRITER_EXECUTION",
    "METADATA_BYTES_PER_TOKEN",
    "METADATA_OFFSET_BYTES",
    "METADATA_VALUES_PER_TOKEN",
    "NUM_GROUPS",
    "NUM_HEADS",
    "PACKED_GROUP_BYTES",
    "PAGE_BYTES",
    "PAGE_SIZE",
    "QUERY_ROTATION_EXECUTION",
    "STORAGE_BYTES_PER_TOKEN",
    "OscarInt2C4Calibration",
    "dequantize_oscar_int2_c4_reference",
    "load_dsv4_oscar_int2_c4_calibrations",
    "oscar_int2_c4_page_bytes",
    "oscar_int2_c4_paged_mqa_logits_reference",
    "oscar_int2_c4_paged_mqa_logits_triton",
    "pack_oscar_int2_c4_pages_reference",
    "scatter_oscar_int2_c4_reference_paged",
    "store_oscar_int2_c4_indexer_cache",
    "unpack_oscar_int2_c4_pages_reference",
    "validate_oscar_int2_c4_calibration",
]
