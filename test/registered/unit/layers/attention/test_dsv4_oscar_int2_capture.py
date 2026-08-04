from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.dsv4 import oscar_int2_capture as capture


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_config(tmp_path: Path) -> Path:
    control = tmp_path / "capture_control.json"
    _write_json(
        control,
        {
            "format": capture.CONTROL_FORMAT,
            "format_version": 1,
            "generation": 1,
            "state": "armed",
            "prompt_id": "train-a",
            "split": "train",
        },
    )
    limits = {kind: {"train": 2, "heldout": 1} for kind in capture._KINDS}
    config = tmp_path / "runtime_capture_config.json"
    _write_json(
        config,
        {
            "format": capture.CONFIG_FORMAT,
            "format_version": 1,
            "session_dir": str(tmp_path),
            "control_path": str(control),
            "session_id": "a" * 64,
            "expected_tp_size": 2,
            "prompt_splits": {"train-a": "train", "heldout-a": "heldout"},
            "maximum_rows_per_prompt": limits,
        },
    )
    return config


@pytest.fixture(autouse=True)
def _reset_global_capture(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SGLANG_DSV4_OSCAR_INT2_KV_STORAGE", raising=False)
    monkeypatch.delenv(capture.CAPTURE_CONFIG_ENV, raising=False)
    monkeypatch.setattr(capture, "_CAPTURE_CONFIG_PATH", "")
    monkeypatch.setattr(capture, "_CAPTURER", None)
    monkeypatch.setattr(capture, "_CAPTURER_PATH", None)


def test_disabled_capture_does_not_initialize() -> None:
    assert capture.capture_configured() is False
    assert capture._get_capturer() is None


def test_attention_tp2_shards_use_identical_bounded_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _make_config(tmp_path)
    config = capture._parse_config(config_path)
    capturer = capture.Dsv4OscarRuntimeCapturer(config)
    monkeypatch.setattr(
        capture,
        "_eligible_forward",
        lambda _forward_batch, *, target_model: target_model,
    )
    forward_batch = SimpleNamespace()
    for rank in (0, 1):
        rows = torch.arange(7 * 32 * 448, dtype=torch.float32).view(7, 32, 448)
        rows = rows + rank * 0.25
        capturer.record(
            kind="attention_query_nope",
            layer_id=3,
            tensor=rows,
            forward_batch=forward_batch,
            target_model=True,
            tp_rank=rank,
            tp_size=2,
            head_start=rank * 32,
        )
    states = [
        torch.load(
            tmp_path
            / "raw"
            / f"rank_{rank:02d}"
            / "layer_03"
            / "train"
            / "attention_query_nope.pt",
            map_location="cpu",
            weights_only=True,
        )
        for rank in (0, 1)
    ]
    assert states[0]["tensor"].shape == (2, 32, 448)
    assert states[1]["tensor"].shape == (2, 32, 448)
    assert torch.equal(states[0]["priorities"], states[1]["priorities"])
    assert states[0]["row_prompt_ids"] == states[1]["row_prompt_ids"]
    assert states[0]["seen_rows_by_prompt"] == {"train-a": 7}
    assert states[1]["head_start"] == 32


def test_c4_scorer_query_rejects_tp_local_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capturer = capture.Dsv4OscarRuntimeCapturer(
        capture._parse_config(_make_config(tmp_path))
    )
    monkeypatch.setattr(capture, "_eligible_forward", lambda *_args, **_kwargs: True)
    with pytest.raises(ValueError, match=r"shape \[rows,64,128\]"):
        capturer.record(
            kind="c4_scorer_query",
            layer_id=2,
            tensor=torch.zeros(3, 32, 128),
            forward_batch=SimpleNamespace(),
            target_model=True,
            tp_rank=0,
            tp_size=2,
            head_start=0,
        )


def test_oscar_enabled_capture_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_DSV4_OSCAR_INT2_KV_STORAGE", "1")
    with pytest.raises(ValueError, match="unrotated baseline"):
        capture.Dsv4OscarRuntimeCapturer(capture._parse_config(_make_config(tmp_path)))


def test_c4_capture_query_applies_learned_head_weight_and_scale() -> None:
    query = torch.ones(2, 64, 128, dtype=torch.bfloat16)
    weight = torch.arange(128, dtype=torch.bfloat16).view(2, 64)
    result = capture._apply_c4_scorer_weight(query, weight, 0.125)
    expected = weight.unsqueeze(-1).expand_as(result) * 0.125
    assert torch.equal(result, expected)


def test_config_requires_exact_tp2(tmp_path: Path) -> None:
    config_path = _make_config(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["expected_tp_size"] = 1
    _write_json(config_path, document)
    with pytest.raises(ValueError, match="exact TP2"):
        capture._parse_config(config_path)
