from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    _load_oscar_admission_receipt,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_receipt(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    artifact_path = tmp_path / "calibration.pt"
    artifact_bytes = b"model-bound OSCAR artifact"
    artifact_path.write_bytes(artifact_bytes)
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    fingerprint_path = tmp_path / "checkpoint-fingerprint.json"
    fingerprint_bytes = b'{"checkpoint":"fingerprint"}\n'
    fingerprint_path.write_bytes(fingerprint_bytes)
    config_sha256 = "2" * 64
    payload: dict[str, object] = {
        "format": "dsv4-oscar-int2-admission",
        "format_version": 1,
        "admitted": True,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "artifact_path": str(artifact_path.resolve()),
        "artifact_file_sha256": _sha256(artifact_bytes),
        "artifact_provenance_sha256": "3" * 64,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": "4" * 64,
        "config_sha256": config_sha256,
        "checkpoint_fingerprint_path": str(fingerprint_path.resolve()),
        "checkpoint_fingerprint_sha256": _sha256(fingerprint_bytes),
        "validation_policy": "rehash-config-index-and-all-referenced-shards-v1",
    }
    payload["admission_sha256"] = _sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )
    receipt_path = tmp_path / "admission.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    return (
        receipt_path,
        artifact_path,
        checkpoint_path,
        config_sha256,
        _sha256(artifact_bytes),
    )


def test_oscar_admission_receipt_binds_artifact_checkpoint_and_config(
    tmp_path: Path,
) -> None:
    receipt_path, artifact_path, checkpoint_path, config_sha256, artifact_sha256 = (
        _write_receipt(tmp_path)
    )

    receipt, receipt_sha256 = _load_oscar_admission_receipt(
        receipt_path,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        checkpoint_path=checkpoint_path,
        config_sha256=config_sha256,
    )

    assert receipt["admitted"] is True
    assert receipt_sha256 == _sha256(receipt_path.read_bytes())


def test_oscar_admission_receipt_rejects_artifact_replacement(tmp_path: Path) -> None:
    receipt_path, artifact_path, checkpoint_path, config_sha256, _ = _write_receipt(
        tmp_path
    )

    with pytest.raises(ValueError, match="artifact changed"):
        _load_oscar_admission_receipt(
            receipt_path,
            artifact_path=artifact_path,
            artifact_sha256="9" * 64,
            checkpoint_path=checkpoint_path,
            config_sha256=config_sha256,
        )


def test_oscar_admission_receipt_rejects_content_tampering(tmp_path: Path) -> None:
    receipt_path, artifact_path, checkpoint_path, config_sha256, artifact_sha256 = (
        _write_receipt(tmp_path)
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["model_id"] = "tampered/model"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="admission_sha256"):
        _load_oscar_admission_receipt(
            receipt_path,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            checkpoint_path=checkpoint_path,
            config_sha256=config_sha256,
        )
