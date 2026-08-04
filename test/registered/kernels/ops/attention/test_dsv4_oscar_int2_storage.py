from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    ARTIFACT_ALGORITHM,
    ARTIFACT_CLIP_SEMANTICS,
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    C4_ROTATION_COMPOSITION,
    C4_ROTATION_OBJECTIVE,
    CALIBRATION_DOMAIN,
    CLIP_OFFLINE_GROUP_THRESHOLDS,
    CLIP_PER_ROW_QUANTILE,
    CONSUMER_SCOPE,
    FORMAT_NAME,
    GROUP_SIZE,
    HEAD_DIM,
    LOGICAL_BYTES_PER_TOKEN,
    NOPE_DIM,
    NUM_GROUPS,
    OSCAR_CLIP_SOURCE_SHA256,
    OSCAR_ROTATION_SOURCE_SHA256,
    OSCAR_SOURCE_COMMIT,
    PACKED_NOPE_BYTES,
    PADDING_BYTES_PER_TOKEN,
    ROPE_DIM,
    ROPE_OFFSET_BYTES,
    SCALE_ZERO_OFFSET_BYTES,
    SHARED_LATENT_SOURCE,
    SHARED_ROTATION_COMPOSITION,
    SHARED_ROTATION_OBJECTIVE,
    STORAGE_BYTES_PER_TOKEN,
    OscarInt2Calibration,
    dequantize_dsv4_oscar_int2_cache_paged,
    dequantize_dsv4_oscar_int2_reference,
    load_dsv4_oscar_int2_calibrations,
    oscar_int2_page_bytes,
    quantize_dsv4_oscar_int2_cache_paged,
    quantize_dsv4_oscar_int2_reference,
    restore_dsv4_oscar_attention_output_shared_latent,
    restore_dsv4_oscar_full_head_attention_output,
    rotate_dsv4_oscar_full_head_shared_latent,
    rotate_dsv4_oscar_query_shared_latent,
    scatter_dsv4_oscar_int2_reference_paged,
    tree_sha256,
    validate_oscar_int2_calibration,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_CLIP_RATIO_VALUES = (0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97)


def _clip_tensors(*, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ratios = torch.tensor(_CLIP_RATIO_VALUES, dtype=torch.float32, device=device)
    indices = torch.minimum(
        torch.floor(ratios * GROUP_SIZE).to(torch.int16),
        torch.full((NUM_GROUPS,), GROUP_SIZE - 1, dtype=torch.int16, device=device),
    )
    return ratios, indices


def _signed_permutation_rotation(*, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260803)
    permutation = torch.randperm(NOPE_DIM, generator=generator)
    signs = torch.where(
        torch.arange(NOPE_DIM) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    rotation = torch.zeros((NOPE_DIM, NOPE_DIM), dtype=torch.bfloat16)
    rotation[torch.arange(NOPE_DIM), permutation] = signs.to(torch.bfloat16)
    return rotation.to(device)


def _dense_c4_rotation() -> tuple[torch.Tensor, float]:
    generator = torch.Generator().manual_seed(20260804)
    matrix = torch.randn((128, 128), generator=generator, dtype=torch.float64)
    rotation = torch.linalg.qr(matrix).Q.to(torch.float32)
    rotation_fp64 = rotation.to(torch.float64)
    orthogonality = float(
        (rotation_fp64.T @ rotation_fp64 - torch.eye(128, dtype=torch.float64))
        .abs()
        .amax()
    )
    return rotation, orthogonality


def _calibration(
    *,
    device: str,
    clip_mode: str = CLIP_OFFLINE_GROUP_THRESHOLDS,
) -> OscarInt2Calibration:
    thresholds = torch.linspace(1.25, 2.75, NUM_GROUPS, dtype=torch.float32).to(device)
    clip_ratios, clip_indices = _clip_tensors(device=device)
    return validate_oscar_int2_calibration(
        _signed_permutation_rotation(device=device),
        thresholds,
        clip_ratios,
        clip_indices,
        layer_id=17,
        clip_mode=clip_mode,
        clip_provenance="unit-test-signed-permutation-calibration",
    )


def _values(num_tokens: int, *, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(
        (num_tokens, HEAD_DIM),
        dtype=torch.float32,
        generator=generator,
    ).clamp(-4.0, 4.0)
    return values.to(torch.bfloat16).to(device)


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("the prototype targets NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("the prototype is fail-closed to SM86")


def test_oscar_int2_layout_is_distinct_from_symmetric_int4() -> None:
    assert FORMAT_NAME == "oscar-int2-asym-g64-v1"
    assert CALIBRATION_DOMAIN == "attention_shared_latent"
    assert NOPE_DIM == 448
    assert ROPE_DIM == 64
    assert GROUP_SIZE == 64
    assert NUM_GROUPS == 7
    assert PACKED_NOPE_BYTES == 112
    assert ROPE_OFFSET_BYTES == 112
    assert SCALE_ZERO_OFFSET_BYTES == 240
    assert LOGICAL_BYTES_PER_TOKEN == 268
    assert STORAGE_BYTES_PER_TOKEN == 272
    assert PADDING_BYTES_PER_TOKEN == 4
    assert STORAGE_BYTES_PER_TOKEN % 16 == 0
    assert oscar_int2_page_bytes(2) == 544
    assert oscar_int2_page_bytes(64) == 17_408
    assert oscar_int2_page_bytes(128) == 34_816


def test_calibration_is_explicit_orthogonal_and_provenanced() -> None:
    calibration = _calibration(device="cpu")
    assert calibration.domain == CALIBRATION_DOMAIN
    assert calibration.layer_id == 17
    expected_ratios, expected_indices = _clip_tensors(device="cpu")
    assert torch.equal(calibration.clip_ratios, expected_ratios)
    assert torch.equal(calibration.clip_indices, expected_indices)
    assert calibration.clip_mode == CLIP_OFFLINE_GROUP_THRESHOLDS
    assert calibration.clip_provenance.startswith("unit-test")
    assert calibration.max_orthogonality_error == 0.0


def _artifact() -> dict[str, object]:
    rotation = _signed_permutation_rotation(device="cpu").float()
    thresholds = torch.linspace(1.25, 2.75, NUM_GROUPS, dtype=torch.float32)
    clip_ratios, clip_indices = _clip_tensors(device="cpu")
    metrics = {
        "query_relative_error": 0.1,
        "value_relative_error": 0.1,
        "joint_relative_error": 0.1,
        "unrotated_query_relative_error": 0.2,
        "unrotated_value_relative_error": 0.2,
        "unrotated_joint_relative_error": 0.2,
        "improvement_vs_unrotated": 0.1,
        "row_count": 64,
    }
    layers: dict[int, dict[str, object]] = {
        layer_id: {
            CALIBRATION_DOMAIN: {
                "domain": CALIBRATION_DOMAIN,
                "rotation_objective": SHARED_ROTATION_OBJECTIVE,
                "rotation_composition": SHARED_ROTATION_COMPOSITION,
                "rotation": rotation,
                "eigenvalues": torch.ones(NOPE_DIM, dtype=torch.float32),
                "clip_thresholds": thresholds,
                "clip_mode": CLIP_PER_ROW_QUANTILE,
                "clip_ratios": clip_ratios,
                "clip_indices": clip_indices,
                "clip_semantics": ARTIFACT_CLIP_SEMANTICS,
                "clip_calibration": {
                    "candidate_ratios": [0.91, 0.97],
                    "train_group_relative_error": [[0.1, 0.2]],
                },
                "provenance_sha256": f"{layer_id:064x}",
                "orthogonality_max_abs": 0.0,
                "heldout_metrics": metrics,
            }
        }
        for layer_id in range(2, 43)
    }
    expected_ratios = (
        0,
        0,
        *(value for _ in range(20) for value in (4, 128)),
        4,
    )
    compression_ratios = torch.tensor(expected_ratios, dtype=torch.int16)
    for layer_id, layer in layers.items():
        layer["layer_id"] = layer_id
        layer["compress_ratio"] = expected_ratios[layer_id]
    c4_ratios = torch.tensor([0.94], dtype=torch.float32)
    c4_indices = torch.floor(c4_ratios * 128).to(torch.int16)
    c4_rotation, c4_orthogonality = _dense_c4_rotation()
    c4_payload = {
        "domain": "c4_scorer",
        "rotation_objective": C4_ROTATION_OBJECTIVE,
        "rotation_composition": C4_ROTATION_COMPOSITION,
        "rotation": c4_rotation,
        "eigenvalues": torch.ones(128, dtype=torch.float32),
        "clip_thresholds": torch.tensor([1.5], dtype=torch.float32),
        "clip_mode": CLIP_PER_ROW_QUANTILE,
        "clip_ratios": c4_ratios,
        "clip_indices": c4_indices,
        "clip_semantics": ARTIFACT_CLIP_SEMANTICS,
        "clip_calibration": {
            "candidate_ratios": [0.94],
            "train_group_relative_error": [[0.1, 0.2]],
        },
        "provenance_sha256": "3" * 64,
        "orthogonality_max_abs": c4_orthogonality,
        "heldout_metrics": metrics,
    }
    for layer_id, compress_ratio in enumerate(expected_ratios):
        if compress_ratio == 4:
            layers[layer_id]["c4_scorer"] = c4_payload
    artifact: dict[str, object] = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "algorithm": ARTIFACT_ALGORITHM,
        "model": {
            "model_id": "deepseek-v4-flash-test",
            "model_type": "deepseek_v4",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": HEAD_DIM,
            "latent_dim": NOPE_DIM,
            "rope_dim": ROPE_DIM,
            "index_head_dim": 128,
            "index_n_heads": 64,
            "compression_ratios": compression_ratios,
        },
        "quantization": {
            "bits": 2,
            "group_size": GROUP_SIZE,
            "num_groups": NUM_GROUPS,
            "storage_layout": FORMAT_NAME,
            "codes_bytes_per_token": PACKED_NOPE_BYTES,
            "scale_zero_bytes_per_token": 28,
            "rope_bytes_per_token": 128,
            "logical_bytes_per_token": LOGICAL_BYTES_PER_TOKEN,
            "padded_bytes_per_token": STORAGE_BYTES_PER_TOKEN,
        },
        "provenance": {
            "prompt_manifest_sha256": "c" * 64,
            "capture_manifest_sha256": "d" * 64,
            "checkpoint_fingerprint_sha256": "e" * 64,
            "capture_input_set_sha256": "f" * 64,
            "calibration_prompt_tokens": 1024,
            "train_latent_rows": 2048,
            "heldout_latent_rows": 1024,
            "shared_latent_source": SHARED_LATENT_SOURCE,
            "shared_rotation_objective": SHARED_ROTATION_OBJECTIVE,
            "c4_rotation_objective": C4_ROTATION_OBJECTIVE,
            "oscar_source_commit": OSCAR_SOURCE_COMMIT,
            "oscar_rotation_source_sha256": OSCAR_ROTATION_SOURCE_SHA256,
            "oscar_clip_source_sha256": OSCAR_CLIP_SOURCE_SHA256,
            "statistics_sha256": "1" * 64,
            "train_sample_rows": 1024,
            "heldout_sample_rows": 512,
            "consumer_scope": CONSUMER_SCOPE,
            "shared_rotation_composition": SHARED_ROTATION_COMPOSITION,
            "c4_rotation_composition": C4_ROTATION_COMPOSITION,
            "calibrator_source_sha256": "2" * 64,
        },
        "layers": layers,
        "artifact_provenance_sha256": "",
    }
    artifact["artifact_provenance_sha256"] = tree_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_provenance_sha256"
        }
    )
    return artifact


def test_artifact_loader_is_absolute_hash_bound_and_covers_compressed_layers(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "dsv4-oscar-int2.pt"
    torch.save(_artifact(), artifact_path)

    calibrations = load_dsv4_oscar_int2_calibrations(
        artifact_path,
        device="cpu",
        expected_metadata={
            "config_sha256": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "prompt_manifest_sha256": "c" * 64,
        },
    )
    assert set(calibrations) == set(range(2, 43))
    assert calibrations[42].domain == CALIBRATION_DOMAIN
    assert calibrations[42].layer_id == 42

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_dsv4_oscar_int2_calibrations(
            artifact_path,
            device="cpu",
            expected_metadata={"checkpoint_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="must be absolute"):
        load_dsv4_oscar_int2_calibrations(
            "relative.pt",
            device="cpu",
        )


def test_artifact_loader_rejects_a_self_hashed_failed_heldout_gate(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    shared = artifact["layers"][2][CALIBRATION_DOMAIN]  # type: ignore[index]
    shared["heldout_metrics"]["improvement_vs_unrotated"] = -0.01  # type: ignore[index]
    artifact["artifact_provenance_sha256"] = tree_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_provenance_sha256"
        }
    )
    artifact_path = tmp_path / "failed-gate.pt"
    torch.save(artifact, artifact_path)
    with pytest.raises(ValueError, match="improvement_vs_unrotated.*gate"):
        load_dsv4_oscar_int2_calibrations(artifact_path, device="cpu")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("non_orthogonal", "not orthogonal"),
        ("zero_threshold", "positive"),
        ("wrong_clip_indices", "clip_indices"),
        ("missing_provenance", "clip_provenance"),
    ],
)
def test_calibration_rejects_invalid_or_implicit_inputs(
    mutation: str, match: str
) -> None:
    rotation = _signed_permutation_rotation(device="cpu")
    thresholds = torch.ones(NUM_GROUPS, dtype=torch.float32)
    clip_ratios, clip_indices = _clip_tensors(device="cpu")
    provenance = "calibration-receipt-sha256"
    if mutation == "non_orthogonal":
        rotation[0].zero_()
    elif mutation == "zero_threshold":
        thresholds[2] = 0.0
    elif mutation == "wrong_clip_indices":
        clip_indices[0] -= 1
    elif mutation == "missing_provenance":
        provenance = ""
    with pytest.raises(ValueError, match=match):
        validate_oscar_int2_calibration(
            rotation,
            thresholds,
            clip_ratios,
            clip_indices,
            layer_id=0,
            clip_mode=CLIP_OFFLINE_GROUP_THRESHOLDS,
            clip_provenance=provenance,
        )


def test_reference_is_asymmetric_int2_with_exact_rope_and_padding() -> None:
    calibration = _calibration(device="cpu")
    values = _values(5, seed=11, device="cpu")
    storage = quantize_dsv4_oscar_int2_reference(values, calibration)
    decoded = dequantize_dsv4_oscar_int2_reference(storage)

    codes = storage[:, :PACKED_NOPE_BYTES]
    assert torch.all((codes & 0x03) <= 3)
    assert torch.all(((codes >> 2) & 0x03) <= 3)
    metadata = storage[:, SCALE_ZERO_OFFSET_BYTES:LOGICAL_BYTES_PER_TOKEN].view(
        torch.bfloat16
    )
    scales = metadata[:, 0::2]
    zeros = metadata[:, 1::2]
    assert torch.all(scales > 0)
    # Asymmetric zero points are data-dependent and not the symmetric-INT4 zero.
    assert torch.count_nonzero(zeros) > 0
    assert torch.equal(decoded[:, NOPE_DIM:], values[:, NOPE_DIM:])
    assert torch.count_nonzero(storage[:, LOGICAL_BYTES_PER_TOKEN:]) == 0


@pytest.mark.parametrize(
    "clip_mode",
    [CLIP_PER_ROW_QUANTILE, CLIP_OFFLINE_GROUP_THRESHOLDS],
)
def test_reference_clip_modes_are_explicit_and_not_silently_substituted(
    clip_mode: str,
) -> None:
    calibration = _calibration(device="cpu", clip_mode=clip_mode)
    values = _values(4, seed=17, device="cpu")
    storage = quantize_dsv4_oscar_int2_reference(values, calibration)
    assert storage.shape == (4, STORAGE_BYTES_PER_TOKEN)

    other_semantics = (
        CLIP_OFFLINE_GROUP_THRESHOLDS
        if clip_mode == CLIP_PER_ROW_QUANTILE
        else CLIP_PER_ROW_QUANTILE
    )
    other = _calibration(device="cpu", clip_mode=other_semantics)
    other_storage = quantize_dsv4_oscar_int2_reference(values, other)
    assert not torch.equal(storage, other_storage)


def test_shared_latent_rotation_preserves_scores_and_inverse_contract() -> None:
    calibration = _calibration(device="cpu")
    generator = torch.Generator().manual_seed(19)
    query = torch.randn((3, NOPE_DIM), generator=generator)
    latent = torch.randn((5, NOPE_DIM), generator=generator)
    rotation = calibration.rotation.float()

    expected_scores = query @ latent.T
    rotated_query = query @ rotation
    rotated_latent = latent @ rotation
    torch.testing.assert_close(
        rotated_query @ rotated_latent.T,
        expected_scores,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    restored = rotated_latent @ rotation.T
    torch.testing.assert_close(restored, latent, rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [2, 64, 128])
def test_reference_pages_negative_locations_and_canaries(page_size: int) -> None:
    calibration = _calibration(device="cpu")
    values = _values(5, seed=23 + page_size, device="cpu")
    locations = torch.tensor(
        [0, page_size - 1, page_size, page_size * 2 - 1, -1],
        dtype=torch.int32,
    )
    page_bytes = oscar_int2_page_bytes(page_size)
    backing = torch.full((2, page_bytes + 32), 0xA5, dtype=torch.uint8)
    storage = backing[:, 16 : 16 + page_bytes]

    scatter_dsv4_oscar_int2_reference_paged(
        values,
        calibration,
        storage,
        locations,
        page_size=page_size,
    )
    expected = quantize_dsv4_oscar_int2_reference(values, calibration)

    assert torch.all(backing[:, :16] == 0xA5)
    assert torch.all(backing[:, 16 + page_bytes :] == 0xA5)
    for input_row, location in enumerate(locations[:-1].tolist()):
        page, in_page = divmod(location, page_size)
        start = in_page * STORAGE_BYTES_PER_TOKEN
        assert torch.equal(
            storage[page, start : start + LOGICAL_BYTES_PER_TOKEN],
            expected[input_row, :LOGICAL_BYTES_PER_TOKEN],
        )
        assert torch.all(
            storage[
                page,
                start + LOGICAL_BYTES_PER_TOKEN : start + STORAGE_BYTES_PER_TOKEN,
            ]
            == 0xA5
        )


@pytest.mark.parametrize(
    ("page_size", "clip_mode"),
    [
        (2, CLIP_OFFLINE_GROUP_THRESHOLDS),
        (64, CLIP_OFFLINE_GROUP_THRESHOLDS),
        (128, CLIP_OFFLINE_GROUP_THRESHOLDS),
        (64, CLIP_PER_ROW_QUANTILE),
    ],
)
def test_sm86_fused_writer_decoder_matches_reference(
    page_size: int, clip_mode: str
) -> None:
    _require_sm86()
    calibration = _calibration(device="cuda", clip_mode=clip_mode)
    values = _values(6, seed=31 + page_size, device="cuda")
    locations = torch.tensor(
        [0, page_size - 1, page_size, page_size * 2 - 1, -1, 1],
        dtype=torch.int32,
        device="cuda",
    )
    write_mask = torch.tensor(
        [True, True, True, True, True, False], dtype=torch.bool, device="cuda"
    )
    page_bytes = oscar_int2_page_bytes(page_size)
    backing = torch.full((2, page_bytes + 32), 0xA5, dtype=torch.uint8, device="cuda")
    storage = backing[:, 16 : 16 + page_bytes]
    output = torch.full(
        (locations.numel(), HEAD_DIM),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )

    quantize_dsv4_oscar_int2_cache_paged(
        values,
        calibration,
        storage,
        locations,
        page_size=page_size,
        write_mask=write_mask,
    )
    dequantize_dsv4_oscar_int2_cache_paged(
        storage, locations, output, page_size=page_size
    )
    torch.cuda.synchronize()
    expected = quantize_dsv4_oscar_int2_reference(values, calibration)

    assert torch.all(backing[:, :16] == 0xA5)
    assert torch.all(backing[:, 16 + page_bytes :] == 0xA5)
    for input_row, location in enumerate(locations[:4].cpu().tolist()):
        page, in_page = divmod(location, page_size)
        start = in_page * STORAGE_BYTES_PER_TOKEN
        assert torch.equal(
            storage[page, start : start + LOGICAL_BYTES_PER_TOKEN],
            expected[input_row, :LOGICAL_BYTES_PER_TOKEN],
        )
        assert torch.all(
            storage[
                page,
                start + LOGICAL_BYTES_PER_TOKEN : start + STORAGE_BYTES_PER_TOKEN,
            ]
            == 0xA5
        )
    assert torch.isnan(output[4]).all()


def test_sm86_query_inverse_and_cache_replay_in_cuda_graph() -> None:
    _require_sm86()
    calibration = _calibration(device="cuda")
    page_size = 2
    values = _values(3, seed=47, device="cuda")
    locations = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    write_mask = torch.ones(3, dtype=torch.bool, device="cuda")
    storage = torch.full(
        (2, oscar_int2_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    decoded = torch.empty((3, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    query = _values(3, seed=53, device="cuda")[:, :NOPE_DIM].contiguous()
    rotated_query = torch.empty_like(query)
    restored_query = torch.empty_like(query)
    full_rotated = torch.empty_like(values)
    full_restored = torch.empty_like(values)

    def run() -> None:
        quantize_dsv4_oscar_int2_cache_paged(
            values,
            calibration,
            storage,
            locations,
            page_size=page_size,
            write_mask=write_mask,
        )
        dequantize_dsv4_oscar_int2_cache_paged(
            storage, locations, decoded, page_size=page_size
        )
        rotate_dsv4_oscar_query_shared_latent(query, calibration, rotated_query)
        restore_dsv4_oscar_attention_output_shared_latent(
            rotated_query, calibration, restored_query
        )
        rotate_dsv4_oscar_full_head_shared_latent(values, calibration, full_rotated)
        restore_dsv4_oscar_full_head_attention_output(
            full_rotated, calibration, full_restored
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    next_values = _values(3, seed=59, device="cuda")
    next_query = _values(3, seed=61, device="cuda")[:, :NOPE_DIM]
    values.copy_(next_values)
    query.copy_(next_query)
    graph.replay()
    torch.cuda.synchronize()

    expected_storage = quantize_dsv4_oscar_int2_reference(next_values, calibration)
    expected_decoded = dequantize_dsv4_oscar_int2_reference(expected_storage)
    torch.testing.assert_close(decoded, expected_decoded, rtol=0, atol=0)
    torch.testing.assert_close(restored_query, next_query, rtol=0, atol=0)
    torch.testing.assert_close(full_restored, next_values, rtol=0, atol=0)


def test_sm86_masked_writer_replay_reads_live_nonboundary_mask() -> None:
    """A captured decode writer must stay idle until its live boundary bit flips."""

    _require_sm86()
    calibration = _calibration(device="cuda")
    page_size = 2
    values = _values(4, seed=67, device="cuda")
    locations = torch.arange(4, dtype=torch.int32, device="cuda")
    write_mask = torch.zeros(4, dtype=torch.bool, device="cuda")
    storage = torch.full(
        (2, oscar_int2_page_bytes(page_size)),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    def run() -> None:
        quantize_dsv4_oscar_int2_cache_paged(
            values,
            calibration,
            storage,
            locations,
            page_size=page_size,
            write_mask=write_mask,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    storage.fill_(0xA5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(storage == 0xA5)

    values.copy_(_values(4, seed=71, device="cuda"))
    write_mask[1] = True
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.full_like(storage, 0xA5)
    scatter_dsv4_oscar_int2_reference_paged(
        values[1:2],
        calibration,
        expected,
        locations[1:2],
        page_size=page_size,
    )
    assert torch.equal(storage, expected)

    persisted = storage.clone()
    values.copy_(_values(4, seed=73, device="cuda"))
    write_mask.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(storage, persisted)
