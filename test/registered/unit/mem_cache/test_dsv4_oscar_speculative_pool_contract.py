from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.attention.dsv4 import (
    oscar_int2_c4_indexer,
    oscar_int2_storage,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache import deepseek_v4_memory_pool as pool_module
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    _load_dsv4_oscar_runtime_calibrations,
    _resolve_dsv4_oscar_kv_pool_contract,
)
from sglang.srt.mem_cache.kv_cache_configurator import (
    _resolve_dsv4_worker_compression_ratios,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _target_contract():
    return _resolve_dsv4_oscar_kv_pool_contract(
        is_draft_worker=False,
        compression_ratios=[0, 4, 128, 4],
        swa_size=1024,
        c4_size=256,
        c128_size=8,
        c4_state_pool_size=64,
        c128_state_pool_size=4,
    )


def _draft_contract():
    return _resolve_dsv4_oscar_kv_pool_contract(
        is_draft_worker=True,
        compression_ratios=[0],
        swa_size=1024,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
    )


def test_dsv4_worker_ratio_contract_preserves_target_and_rewrites_draft() -> None:
    target_ratios = [4, 128, 0]

    resolved_target = _resolve_dsv4_worker_compression_ratios(
        is_draft_worker=False,
        model_compression_ratios=target_ratios,
        num_effective_layers=3,
    )
    resolved_draft = _resolve_dsv4_worker_compression_ratios(
        is_draft_worker=True,
        model_compression_ratios=target_ratios,
        num_effective_layers=2,
    )

    assert resolved_target == target_ratios
    assert resolved_target is not target_ratios
    assert resolved_draft == [0, 0]


def test_oscar_target_and_draft_contracts_have_disjoint_cache_ownership() -> None:
    target = _target_contract()
    draft = _draft_contract()

    assert target.consumer_role == "target_compressed"
    assert target.compressed_layer_ids == {1, 2, 3}
    assert target.c4_layer_ids == {1, 3}
    assert target.retain_runtime_calibrations is True

    assert draft.consumer_role == "draft_swa_only"
    assert draft.compressed_layer_ids == set()
    assert draft.c4_layer_ids == set()
    assert draft.retain_runtime_calibrations is False


@pytest.mark.parametrize(
    ("compression_ratios", "capacities", "message"),
    [
        ([4], (1, 0, 1, 0), "must be SWA-only"),
        ([0], (1, 0, 0, 0), "cannot allocate compressed"),
    ],
)
def test_oscar_draft_rejects_any_compressed_layer_or_capacity(
    compression_ratios: list[int],
    capacities: tuple[int, int, int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_dsv4_oscar_kv_pool_contract(
            is_draft_worker=True,
            compression_ratios=compression_ratios,
            swa_size=1024,
            c4_size=capacities[0],
            c128_size=capacities[1],
            c4_state_pool_size=capacities[2],
            c128_state_pool_size=capacities[3],
        )


def test_oscar_target_rejects_missing_compressed_topology_or_capacity() -> None:
    with pytest.raises(ValueError, match="lost its calibrated"):
        _resolve_dsv4_oscar_kv_pool_contract(
            is_draft_worker=False,
            compression_ratios=[0],
            swa_size=1024,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
        )

    with pytest.raises(ValueError, match="no capacity"):
        _resolve_dsv4_oscar_kv_pool_contract(
            is_draft_worker=False,
            compression_ratios=[4, 128],
            swa_size=1024,
            c4_size=0,
            c128_size=8,
            c4_state_pool_size=64,
            c128_state_pool_size=4,
        )


def test_target_retains_full_calibrations_and_enforces_exact_c4_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    shared = {1: object(), 2: object(), 3: object()}
    c4 = {1: object(), 3: object()}

    def load_shared(*_args, device, **_kwargs):
        calls.append(("shared", str(device)))
        return shared

    def load_c4(*_args, device, **_kwargs):
        calls.append(("c4", str(device)))
        return c4

    monkeypatch.setattr(
        oscar_int2_storage, "load_dsv4_oscar_int2_calibrations", load_shared
    )
    monkeypatch.setattr(
        oscar_int2_c4_indexer,
        "load_dsv4_oscar_int2_c4_calibrations",
        load_c4,
    )

    loaded_shared, loaded_c4 = _load_dsv4_oscar_runtime_calibrations(
        artifact_path=Path("/absolute/calibration.pt"),
        device="cuda:1",
        config_sha256="a" * 64,
        contract=_target_contract(),
    )

    assert loaded_shared is shared
    assert loaded_c4 is c4
    assert calls == [("shared", "cuda:1"), ("c4", "cuda:1")]

    monkeypatch.setattr(
        oscar_int2_storage,
        "load_dsv4_oscar_int2_calibrations",
        lambda *_args, **_kwargs: {1: object(), 2: object(), 4: object()},
    )
    with pytest.raises(ValueError, match=r"missing=\[3\], extra=\[4\]"):
        _load_dsv4_oscar_runtime_calibrations(
            artifact_path=Path("/absolute/calibration.pt"),
            device="cuda:1",
            config_sha256="a" * 64,
            contract=_target_contract(),
        )

    monkeypatch.setattr(
        oscar_int2_storage, "load_dsv4_oscar_int2_calibrations", load_shared
    )
    monkeypatch.setattr(
        oscar_int2_c4_indexer,
        "load_dsv4_oscar_int2_c4_calibrations",
        lambda *_args, **_kwargs: {1: object()},
    )
    with pytest.raises(ValueError, match=r"missing=\[3\], extra=\[\]"):
        _load_dsv4_oscar_runtime_calibrations(
            artifact_path=Path("/absolute/calibration.pt"),
            device="cuda:1",
            config_sha256="a" * 64,
            contract=_target_contract(),
        )


def test_draft_validates_artifact_on_cpu_without_loading_c4_or_retaining_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def load_shared(*_args, device, **_kwargs):
        calls.append(("shared", str(device)))
        return {layer_id: object() for layer_id in range(43)}

    def reject_c4(*_args, **_kwargs):
        raise AssertionError("an SWA-only draft must not load target C4 tensors")

    monkeypatch.setattr(
        oscar_int2_storage, "load_dsv4_oscar_int2_calibrations", load_shared
    )
    monkeypatch.setattr(
        oscar_int2_c4_indexer,
        "load_dsv4_oscar_int2_c4_calibrations",
        reject_c4,
    )

    shared, c4 = _load_dsv4_oscar_runtime_calibrations(
        artifact_path=Path("/absolute/calibration.pt"),
        device="cuda:0",
        config_sha256="b" * 64,
        contract=_draft_contract(),
    )

    assert shared == {}
    assert c4 == {}
    assert calls == [("shared", "cpu")]


def test_oscar_draft_still_requires_model_bound_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    artifact = tmp_path / "calibration.pt"
    artifact.write_bytes(b"not reached: receipt admission must fail first")
    monkeypatch.setattr(
        pool_module, "get_dsv4_device_capability", lambda _device: (8, 6)
    )
    monkeypatch.setattr(
        pool_module,
        "get_server_args",
        lambda: SimpleNamespace(model_path=str(checkpoint)),
    )

    with ExitStack() as overrides:
        overrides.enter_context(envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.override(True))
        overrides.enter_context(envs.SGLANG_DSV4_INT4_KV_STORAGE.override(False))
        overrides.enter_context(
            envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(False)
        )
        overrides.enter_context(envs.SGLANG_DSV4_SM86_C128_BF16_STORAGE.override(False))
        overrides.enter_context(
            envs.SGLANG_DSV4_OSCAR_CALIBRATION_PATH.override(str(artifact))
        )
        overrides.enter_context(
            envs.SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH.override("")
        )

        with pytest.raises(ValueError, match="model-bound.*ADMISSION_RECEIPT_PATH"):
            DeepSeekV4TokenToKVPool(
                max_num_reqs=1,
                num_req_slots=2,
                swa_size=256,
                c4_size=0,
                c128_size=0,
                c4_state_pool_size=0,
                c128_state_pool_size=0,
                page_size=256,
                swa_page_size=256,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.float32,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
                compression_ratios=[0],
                is_draft_worker=True,
            )


def test_draft_calibration_accessors_fail_closed() -> None:
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool.use_oscar_int2_storage = True
    pool.oscar_consumer_role = "draft_swa_only"

    with pytest.raises(RuntimeError, match="SWA-only"):
        pool.get_oscar_calibration(0)
    with pytest.raises(RuntimeError, match="SWA-only"):
        pool.get_oscar_c4_calibration(0)
