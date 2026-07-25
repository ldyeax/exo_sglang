from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sglang.srt.disaggregation.glm52_intra_node_runtime import (
    GLM52PDDeviceContract,
    GLM52PDGPUObservation,
    GLM52PDKVHandoffContract,
    GLM52PDLaunchConfig,
    GLM52PDRuntimeContractError,
    GLM52PDSharedWeightContract,
    build_glm52_pd_launch_plan,
    live_policy_contracts,
    model_metadata_identity,
    validate_gpu_contracts,
    verify_shared_generation_leases,
    verify_shared_weight_contract,
)

_MIB = 1024 * 1024
_GIB = 1024**3
_PREFILL_UUID = "GPU-11111111-1111-1111-1111-111111111111"
_DECODE_UUID = "GPU-22222222-2222-2222-2222-222222222222"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _shared_weight_contract(tmp_path: Path) -> GLM52PDSharedWeightContract:
    checkpoint = tmp_path / "amxint4"
    checkpoint.mkdir()
    weight = checkpoint / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"immutable-amxint4")
    weight.chmod(0o444)
    files = [
        {
            "path": weight.name,
            "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "size_bytes": weight.stat().st_size,
        }
    ]
    content_id = hashlib.sha256(
        _canonical_json(
            {
                "files": files,
                "kind": "kt_shared_host_weights_content",
                "schema_version": 1,
            }
        )
    ).hexdigest()
    manifest_value = {
        "content_id": content_id,
        "files": files,
        "kind": "kt_shared_host_weights_manifest",
        "numa_nodes": [0, 1],
        "schema_version": 1,
    }
    manifest_raw = _canonical_json(manifest_value) + b"\n"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_raw)
    return GLM52PDSharedWeightContract(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=tmp_path / "state",
        content_id=content_id,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        weight_bytes=weight.stat().st_size,
        host_safety_margin_bytes=64 * _MIB,
    )


def _model_path(tmp_path: Path) -> tuple[Path, str]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures":["GlmMoeDsaForCausalLM"]}\n')
    (model / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":1},"weight_map":{"x":"model.safetensors"}}\n'
    )
    return model, model_metadata_identity(model)


def _config(tmp_path: Path) -> GLM52PDLaunchConfig:
    model, model_identity = _model_path(tmp_path)
    return GLM52PDLaunchConfig(
        python_executable=Path(sys.executable).resolve(),
        model_path=model,
        model_identity=model_identity,
        served_model_name="glm-5.2",
        architecture="GlmMoeDsaForCausalLM",
        runtime_root=tmp_path / "runtime",
        shared_weights=_shared_weight_contract(tmp_path),
        prefill_device=GLM52PDDeviceContract(
            gpu_uuid=_PREFILL_UUID,
            numa_node=0,
            required_free_bytes=10 * _GIB,
        ),
        decode_device=GLM52PDDeviceContract(
            gpu_uuid=_DECODE_UUID,
            numa_node=1,
            required_free_bytes=11 * _GIB,
        ),
        kv_handoff=GLM52PDKVHandoffContract(
            transfer_backend="mooncake",
            bootstrap_port=8998,
            context_length=16384,
            maximum_prefill_tokens=16384,
            maximum_total_tokens=32768,
        ),
        prefill_host="127.0.0.1",
        prefill_port=31001,
        decode_host="127.0.0.1",
        decode_port=31002,
        router_host="127.0.0.1",
        router_port=31000,
        cpu_infer_threads=96,
        threadpool_numa_nodes=(0, 1),
        chunked_prefill_size=2048,
        stream_prefill_token_threshold=4096,
    )


def _observations() -> tuple[GLM52PDGPUObservation, ...]:
    return (
        GLM52PDGPUObservation(
            gpu_uuid=_PREFILL_UUID,
            index=0,
            pci_bus_id="0000:41:00.0",
            numa_node=0,
            total_bytes=24 * _GIB,
            free_bytes=20 * _GIB,
        ),
        GLM52PDGPUObservation(
            gpu_uuid=_DECODE_UUID,
            index=1,
            pci_bus_id="0000:81:00.0",
            numa_node=1,
            total_bytes=24 * _GIB,
            free_bytes=19 * _GIB,
        ),
    )


def _argv_value(argv: tuple[str, ...], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_launch_plan_is_executable_native_pd_with_private_process_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    plan = build_glm52_pd_launch_plan(
        config,
        observations=_observations(),
        host_capacity_bytes=128 * _GIB,
    )

    assert _argv_value(plan.prefill.argv, "--tp-size") == "1"
    assert _argv_value(plan.decode.argv, "--tp-size") == "1"
    assert _argv_value(plan.prefill.argv, "--disaggregation-mode") == "prefill"
    assert _argv_value(plan.decode.argv, "--disaggregation-mode") == "decode"
    assert _argv_value(plan.prefill.argv, "--disaggregation-decode-tp") == "1"
    assert "--disaggregation-decode-tp" not in plan.decode.argv
    assert "--kt-stream-prefill" in plan.prefill.argv
    assert "--kt-stream-prefill" in plan.decode.argv

    prefill_environment = plan.prefill.environment_dict()
    decode_environment = plan.decode.environment_dict()
    assert prefill_environment["CUDA_VISIBLE_DEVICES"] == _PREFILL_UUID
    assert decode_environment["CUDA_VISIBLE_DEVICES"] == _DECODE_UUID
    for name in (
        "KT_SHARED_HOST_WEIGHTS_MANIFEST",
        "KT_SHARED_HOST_WEIGHTS_CONTENT_ID",
        "KT_SHARED_HOST_WEIGHTS_STATE_DIR",
    ):
        assert prefill_environment[name] == decode_environment[name]
    assert prefill_environment["TMPDIR"] != decode_environment["TMPDIR"]
    assert prefill_environment["XDG_CACHE_HOME"] != decode_environment["XDG_CACHE_HOME"]

    assert "--pd-disaggregation" in plan.router.argv
    assert (
        "http://127.0.0.1:31001",
        "8998",
    ) == (
        plan.router.argv[plan.router.argv.index("--prefill") + 1],
        plan.router.argv[plan.router.argv.index("--prefill") + 2],
    )
    assert _argv_value(plan.router.argv, "--decode") == "http://127.0.0.1:31002"
    assert plan.policy_contracts.executors.split_slp_ready
    assert plan.policy_contracts.executors.split_decode_ready
    assert plan.policy_contracts.executors.split_kv_transfer_ready
    assert plan.policy_contracts.shared_host_weights is not None
    assert not plan.policy_contracts.shared_host_weights.sharing_verified


def test_shared_weight_contract_rejects_writable_checkpoint_file(
    tmp_path: Path,
) -> None:
    contract = _shared_weight_contract(tmp_path)
    only_weight = next(contract.checkpoint_root.glob("*.safetensors"))
    only_weight.chmod(0o644)

    with pytest.raises(GLM52PDRuntimeContractError, match="is writable"):
        verify_shared_weight_contract(
            contract,
            host_capacity_bytes=128 * _GIB,
        )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda observations: (
                replace(observations[0], numa_node=1),
                observations[1],
            ),
            "moved to NUMA",
        ),
        (
            lambda observations: (
                replace(observations[0], free_bytes=1),
                observations[1],
            ),
            "free bytes",
        ),
        (
            lambda observations: (
                replace(observations[0], compute_process_ids=(1234,)),
                observations[1],
            ),
            "CUDA compute owners",
        ),
    ],
)
def test_gpu_identity_capacity_and_ownership_fail_closed(
    tmp_path: Path,
    mutate,
    expected: str,
) -> None:
    config = _config(tmp_path)
    observations = mutate(_observations())

    with pytest.raises(GLM52PDRuntimeContractError, match=expected):
        validate_gpu_contracts(config, observations)


def _write_process_stat(
    proc_root: Path,
    process_id: int,
    *,
    parent_process_id: int,
    start_time_ticks: int,
) -> None:
    process_root = proc_root / str(process_id)
    process_root.mkdir(parents=True)
    fields = ["S", str(parent_process_id), *(["0"] * 17), str(start_time_ticks)]
    (process_root / "stat").write_text(
        f"{process_id} (test process) {' '.join(fields)}\n"
    )


def _write_lease(
    path: Path,
    *,
    process_id: int,
    start_time_ticks: int,
    generation_id: str,
    content_id: str,
    boot_id: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "boot_id": boot_id,
                "content_id": content_id,
                "generation_id": generation_id,
                "kind": "kt_shared_host_weights_lease",
                "pid": process_id,
                "schema_version": 1,
                "start_time_ticks": start_time_ticks,
            }
        )
    )


def test_live_leases_promote_static_plan_to_one_shared_generation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    plan = build_glm52_pd_launch_plan(
        config,
        observations=_observations(),
        host_capacity_bytes=128 * _GIB,
    )
    proc_root = tmp_path / "proc"
    boot_id = "11111111-2222-3333-4444-555555555555"
    boot_path = proc_root / "sys/kernel/random"
    boot_path.mkdir(parents=True)
    (boot_path / "boot_id").write_text(boot_id)
    _write_process_stat(proc_root, 100, parent_process_id=1, start_time_ticks=1000)
    _write_process_stat(proc_root, 101, parent_process_id=100, start_time_ticks=1010)
    _write_process_stat(proc_root, 200, parent_process_id=1, start_time_ticks=2000)
    _write_process_stat(proc_root, 201, parent_process_id=200, start_time_ticks=2010)

    leases = config.shared_weights.state_directory / "leases"
    leases.mkdir(parents=True)
    _write_lease(
        leases / "prefill.json",
        process_id=101,
        start_time_ticks=1010,
        generation_id="generation-1",
        content_id=config.shared_weights.content_id,
        boot_id=boot_id,
    )
    _write_lease(
        leases / "decode.json",
        process_id=201,
        start_time_ticks=2010,
        generation_id="generation-1",
        content_id=config.shared_weights.content_id,
        boot_id=boot_id,
    )

    proof = verify_shared_generation_leases(
        config.shared_weights.state_directory,
        content_id=config.shared_weights.content_id,
        prefill_root_process_id=100,
        decode_root_process_id=200,
        proc_root=proc_root,
    )
    assert proof.generation_id == "generation-1"
    assert proof.prefill_lease_process_ids == (101,)
    assert proof.decode_lease_process_ids == (201,)

    live_contracts = live_policy_contracts(plan, proof)
    assert live_contracts.shared_host_weights is not None
    assert live_contracts.shared_host_weights.sharing_verified
    assert (
        live_contracts.shared_host_weights.allocation_identity
        == "kt-generation:generation-1"
    )


def test_lease_proof_rejects_two_generations(tmp_path: Path) -> None:
    contract = _shared_weight_contract(tmp_path)
    proc_root = tmp_path / "proc"
    boot_id = "11111111-2222-3333-4444-555555555555"
    boot_path = proc_root / "sys/kernel/random"
    boot_path.mkdir(parents=True)
    (boot_path / "boot_id").write_text(boot_id)
    for process_id, parent, start in (
        (100, 1, 1000),
        (101, 100, 1010),
        (200, 1, 2000),
        (201, 200, 2010),
    ):
        _write_process_stat(
            proc_root,
            process_id,
            parent_process_id=parent,
            start_time_ticks=start,
        )
    leases = contract.state_directory / "leases"
    leases.mkdir(parents=True)
    _write_lease(
        leases / "prefill.json",
        process_id=101,
        start_time_ticks=1010,
        generation_id="generation-prefill",
        content_id=contract.content_id,
        boot_id=boot_id,
    )
    _write_lease(
        leases / "decode.json",
        process_id=201,
        start_time_ticks=2010,
        generation_id="generation-decode",
        content_id=contract.content_id,
        boot_id=boot_id,
    )

    with pytest.raises(GLM52PDRuntimeContractError, match="one shared generation"):
        verify_shared_generation_leases(
            contract.state_directory,
            content_id=contract.content_id,
            prefill_root_process_id=100,
            decode_root_process_id=200,
            proc_root=proc_root,
        )
