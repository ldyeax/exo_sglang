from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import oscar_int2_c4_indexer as c4
from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    CALIBRATION_DOMAIN,
    CLIP_MODE,
    CLIP_SEMANTICS,
    CODES_BYTES_PER_PAGE,
    CODES_BYTES_PER_TOKEN,
    FORMAT_NAME,
    GROUP_SIZE,
    HEAD_DIM,
    METADATA_BYTES_PER_TOKEN,
    METADATA_OFFSET_BYTES,
    METADATA_VALUES_PER_TOKEN,
    NUM_GROUPS,
    NUM_HEADS,
    PAGE_BYTES,
    PAGE_SIZE,
    STORAGE_BYTES_PER_TOKEN,
    OscarInt2C4Calibration,
    dequantize_oscar_int2_c4_reference,
    load_dsv4_oscar_int2_c4_calibrations,
    oscar_int2_c4_page_bytes,
    oscar_int2_c4_paged_mqa_logits_reference,
    oscar_int2_c4_paged_mqa_logits_triton,
    pack_oscar_int2_c4_pages_reference,
    scatter_oscar_int2_c4_reference_paged,
    store_oscar_int2_c4_indexer_cache,
    unpack_oscar_int2_c4_pages_reference,
    validate_oscar_int2_c4_calibration,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_LAYER_ID = 2
_CLIP_RATIO = 0.95
_CLIP_INDEX = min(int(_CLIP_RATIO * GROUP_SIZE), GROUP_SIZE - 1)


def _dense_rotation(*, device: str) -> torch.Tensor:
    return _dense_rotation_fp32().to(torch.bfloat16).contiguous().to(device)


def _dense_rotation_fp32() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260803)
    matrix = torch.randn((HEAD_DIM, HEAD_DIM), generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    return rotation.contiguous()


def _signed_permutation_rotation(*, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260804)
    permutation = torch.randperm(HEAD_DIM, generator=generator)
    signs = torch.where(
        torch.arange(HEAD_DIM) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).to(torch.bfloat16)
    rotation = torch.zeros((HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16)
    rotation[torch.arange(HEAD_DIM), permutation] = signs
    return rotation.to(device)


def _calibration(
    *,
    device: str,
    dense: bool = True,
    layer_id: int = _LAYER_ID,
) -> OscarInt2C4Calibration:
    rotation = (
        _dense_rotation(device=device)
        if dense
        else _signed_permutation_rotation(device=device)
    )
    return validate_oscar_int2_c4_calibration(
        rotation,
        torch.tensor([2.75], dtype=torch.float32, device=device),
        torch.tensor([_CLIP_RATIO], dtype=torch.float32, device=device),
        torch.tensor([_CLIP_INDEX], dtype=torch.int16, device=device),
        layer_id=layer_id,
        clip_mode=CLIP_MODE,
        clip_provenance="unit-test-c4-calibration",
    )


def _keys(num_tokens: int, *, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((num_tokens, HEAD_DIM), generator=generator).clamp(-4, 4)
    return values.to(torch.bfloat16).to(device)


def _query(batch_size: int, *, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((batch_size, 1, NUM_HEADS, HEAD_DIM), generator=generator)
    return values.to(torch.bfloat16).to(device)


def _rope_frequencies(max_positions: int, *, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260804)
    angles = torch.randn((max_positions, 32), generator=generator)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64).to(device)


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("the indexer targets NVIDIA CUDA")
    if torch.cuda.get_device_capability(0) != (8, 6):
        pytest.skip("the indexer is fail-closed to exact SM86")
    torch.cuda.set_device(0)


def test_layout_is_exactly_40_bytes_per_token_without_padding() -> None:
    assert FORMAT_NAME == "oscar-int2-c4-asym-c128-fp32-adjacent4-v1"
    assert CALIBRATION_DOMAIN == "c4_scorer"
    assert GROUP_SIZE == 128
    assert NUM_GROUPS == 1
    assert CODES_BYTES_PER_TOKEN == 32
    assert METADATA_VALUES_PER_TOKEN == 2
    assert METADATA_BYTES_PER_TOKEN == 8
    assert STORAGE_BYTES_PER_TOKEN == 40
    assert CODES_BYTES_PER_PAGE == 2_048
    assert METADATA_OFFSET_BYTES == 2_048
    assert PAGE_BYTES == 2_560
    assert oscar_int2_c4_page_bytes(64) == 2_560
    assert oscar_int2_c4_page_bytes(13) == 520
    assert PAGE_BYTES % 16 == 0


def test_calibration_requires_dense_nonidentity_per_layer_rotation() -> None:
    calibration = _calibration(device="cpu")
    assert calibration.layer_id == _LAYER_ID
    assert calibration.domain == CALIBRATION_DOMAIN
    assert calibration.storage_format == FORMAT_NAME
    assert calibration.max_orthogonality_error < 2.0e-2
    assert calibration.max_identity_deviation > 0.5

    with pytest.raises(ValueError, match="identity.*no fallback"):
        validate_oscar_int2_c4_calibration(
            torch.eye(HEAD_DIM, dtype=torch.bfloat16),
            torch.ones(NUM_GROUPS, dtype=torch.float32),
            torch.tensor([_CLIP_RATIO], dtype=torch.float32),
            torch.tensor([_CLIP_INDEX], dtype=torch.int16),
            layer_id=_LAYER_ID,
            clip_mode=CLIP_MODE,
            clip_provenance="placeholder-is-forbidden",
        )


def _artifact() -> dict[str, object]:
    compression_ratios = torch.tensor(
        [0, 0, *(value for _ in range(20) for value in (4, 128)), 4],
        dtype=torch.int16,
    )
    rotation = _dense_rotation_fp32()
    orthogonality = (
        (
            rotation.double().T @ rotation.double()
            - torch.eye(HEAD_DIM, dtype=torch.float64)
        )
        .abs()
        .amax()
        .item()
    )
    layers: dict[int, dict[str, object]] = {}
    for layer_id in range(2, 43):
        compress_ratio = int(compression_ratios[layer_id].item())
        layer: dict[str, object] = {
            "layer_id": layer_id,
            "compress_ratio": compress_ratio,
            "attention_shared_latent": {
                "domain": "attention_shared_latent",
                "rotation_objective": c4.SHARED_ROTATION_OBJECTIVE,
                "rotation_composition": c4.SHARED_ROTATION_COMPOSITION,
            },
        }
        if compress_ratio == 4:
            layer[CALIBRATION_DOMAIN] = {
                "domain": CALIBRATION_DOMAIN,
                "rotation_objective": c4.C4_ROTATION_OBJECTIVE,
                "rotation_composition": c4.C4_ROTATION_COMPOSITION,
                "rotation": rotation,
                "eigenvalues": torch.ones(HEAD_DIM, dtype=torch.float32),
                "clip_thresholds": torch.tensor([2.75], dtype=torch.float32),
                "clip_mode": CLIP_MODE,
                "clip_ratios": torch.tensor([_CLIP_RATIO], dtype=torch.float32),
                "clip_indices": torch.tensor([_CLIP_INDEX], dtype=torch.int16),
                "clip_semantics": CLIP_SEMANTICS,
                "clip_calibration": {},
                "provenance_sha256": f"{layer_id:064x}",
                "orthogonality_max_abs": orthogonality,
                "heldout_metrics": {
                    "query_relative_error": 0.1,
                    "value_relative_error": 0.1,
                    "joint_relative_error": 0.1,
                    "unrotated_query_relative_error": 0.2,
                    "unrotated_value_relative_error": 0.2,
                    "unrotated_joint_relative_error": 0.2,
                    "improvement_vs_unrotated": 0.1,
                    "row_count": 16,
                },
            }
        layers[layer_id] = layer
    artifact: dict[str, object] = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "algorithm": c4.ARTIFACT_ALGORITHM,
        "model": {
            "model_id": "deepseek-v4-flash-test",
            "model_type": "deepseek_v4",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "latent_dim": 448,
            "rope_dim": 64,
            "index_head_dim": HEAD_DIM,
            "index_n_heads": NUM_HEADS,
            "compression_ratios": compression_ratios,
        },
        "quantization": {},
        "provenance": {
            "prompt_manifest_sha256": "c" * 64,
            "capture_manifest_sha256": "d" * 64,
            "checkpoint_fingerprint_sha256": "e" * 64,
            "capture_input_set_sha256": "f" * 64,
            "calibration_prompt_tokens": 1024,
            "train_latent_rows": 2048,
            "heldout_latent_rows": 1024,
            "shared_latent_source": c4.SHARED_LATENT_SOURCE,
            "shared_rotation_objective": c4.SHARED_ROTATION_OBJECTIVE,
            "c4_rotation_objective": c4.C4_ROTATION_OBJECTIVE,
            "oscar_source_commit": c4.OSCAR_SOURCE_COMMIT,
            "oscar_rotation_source_sha256": c4.OSCAR_ROTATION_SOURCE_SHA256,
            "oscar_clip_source_sha256": c4.OSCAR_CLIP_SOURCE_SHA256,
            "statistics_sha256": "1" * 64,
            "train_sample_rows": 1024,
            "heldout_sample_rows": 512,
            "consumer_scope": c4.CONSUMER_SCOPE,
            "shared_rotation_composition": c4.SHARED_ROTATION_COMPOSITION,
            "c4_rotation_composition": c4.C4_ROTATION_COMPOSITION,
            "calibrator_source_sha256": "2" * 64,
        },
        "layers": layers,
        "artifact_provenance_sha256": "",
    }
    artifact["artifact_provenance_sha256"] = c4.tree_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_provenance_sha256"
        }
    )
    return artifact


def test_loader_consumes_v2_c4_scorer_payload(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact_path = tmp_path / "dsv4-oscar-int2.pt"
    torch.save(artifact, artifact_path)
    calibrations = load_dsv4_oscar_int2_c4_calibrations(
        artifact_path,
        device="cpu",
        expected_metadata={
            "checkpoint_sha256": "a" * 64,
            "prompt_manifest_sha256": "c" * 64,
        },
    )
    expected_c4_layers = {
        layer_id
        for layer_id, ratio in enumerate(
            artifact["model"]["compression_ratios"].tolist()  # type: ignore[index,union-attr]
        )
        if ratio == 4
    }
    assert set(calibrations) == expected_c4_layers
    assert calibrations[_LAYER_ID].rotation.shape == (HEAD_DIM, HEAD_DIM)

    layer = artifact["layers"][_LAYER_ID]  # type: ignore[index]
    del layer[CALIBRATION_DOMAIN]  # type: ignore[index]
    artifact["artifact_provenance_sha256"] = c4.tree_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_provenance_sha256"
        }
    )
    torch.save(artifact, artifact_path)
    with pytest.raises(ValueError, match="missing=.*c4_scorer"):
        load_dsv4_oscar_int2_c4_calibrations(artifact_path, device="cpu")


def test_reference_is_unsigned_affine_adjacent4_with_fp32_metadata() -> None:
    calibration = _calibration(device="cpu")
    keys = _keys(5, seed=11, device="cpu")
    storage = c4.quantize_oscar_int2_c4_reference(keys, calibration)
    assert storage.shape == (5, 40)

    packed = storage[:, :CODES_BYTES_PER_TOKEN]
    metadata = storage[:, CODES_BYTES_PER_TOKEN:].view(torch.float32)
    scales = metadata[:, 0]
    zeros = metadata[:, 1]
    assert torch.all(scales > 0)
    assert torch.count_nonzero(zeros) > 0

    rotated = keys.float() @ calibration.rotation.float()
    threshold = rotated.abs().sort(dim=-1).values[:, calibration.clip_index]
    clipped = torch.minimum(
        torch.maximum(rotated, -threshold[:, None]), threshold[:, None]
    )
    minimum = clipped.amin(dim=-1)
    maximum = clipped.amax(dim=-1)
    expected_scales = torch.where(
        maximum == minimum, torch.ones_like(maximum), (maximum - minimum) / 3.0
    )
    expected_zeros = -minimum / expected_scales
    expected_codes = (
        torch.floor(clipped / expected_scales[:, None] + expected_zeros[:, None] + 0.5)
        .clamp(0, 3)
        .to(torch.uint8)
    )
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(zeros, expected_zeros, rtol=0, atol=0)
    assert torch.equal(packed & 0x03, expected_codes[:, 0::4])
    assert torch.equal((packed >> 2) & 0x03, expected_codes[:, 1::4])
    assert torch.equal((packed >> 4) & 0x03, expected_codes[:, 2::4])
    assert torch.equal((packed >> 6) & 0x03, expected_codes[:, 3::4])

    decoded = dequantize_oscar_int2_c4_reference(storage).float()
    expected_decoded = (
        ((expected_codes.float() - expected_zeros[:, None]) * expected_scales[:, None])
        .to(torch.bfloat16)
        .float()
    )
    torch.testing.assert_close(decoded, expected_decoded, rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [2, 64])
def test_reference_page_planes_negative_locations_and_canaries(page_size: int) -> None:
    calibration = _calibration(device="cpu", dense=False)
    keys = _keys(5, seed=17 + page_size, device="cpu")
    locations = torch.tensor(
        [0, page_size - 1, page_size, page_size * 2 - 1, -1], dtype=torch.int32
    )
    page_bytes = oscar_int2_c4_page_bytes(page_size)
    backing = torch.full((2, page_bytes + 32), 0xA5, dtype=torch.uint8)
    storage = backing[:, 16 : 16 + page_bytes]
    scatter_oscar_int2_c4_reference_paged(
        keys, calibration, storage, locations, page_size=page_size
    )
    records = c4.quantize_oscar_int2_c4_reference(keys, calibration)

    assert torch.all(backing[:, :16] == 0xA5)
    assert torch.all(backing[:, 16 + page_bytes :] == 0xA5)
    code_page_bytes = page_size * CODES_BYTES_PER_TOKEN
    for input_row, location in enumerate(locations[:-1].tolist()):
        page, in_page = divmod(location, page_size)
        code_start = in_page * CODES_BYTES_PER_TOKEN
        metadata_start = code_page_bytes + in_page * METADATA_BYTES_PER_TOKEN
        assert torch.equal(
            storage[page, code_start : code_start + CODES_BYTES_PER_TOKEN],
            records[input_row, :CODES_BYTES_PER_TOKEN],
        )
        assert torch.equal(
            storage[page, metadata_start : metadata_start + METADATA_BYTES_PER_TOKEN],
            records[input_row, CODES_BYTES_PER_TOKEN:],
        )


def test_reference_scorer_applies_the_same_rotation_to_query() -> None:
    calibration = _calibration(device="cpu")
    page_size = 4
    raw_pages = _keys(3 * page_size, seed=29, device="cpu").reshape(
        3, page_size, HEAD_DIM
    )
    storage = pack_oscar_int2_c4_pages_reference(raw_pages, calibration)
    query = _query(2, seed=31, device="cpu")
    weight = torch.randn((2, NUM_HEADS), generator=torch.Generator().manual_seed(37))
    seq_lens = torch.tensor([7, 6], dtype=torch.int32)
    page_table = torch.tensor([[2, 0], [-1, 1]], dtype=torch.int32)
    actual = oscar_int2_c4_paged_mqa_logits_reference(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        calibration,
        8,
        page_size=page_size,
    )

    decoded = unpack_oscar_int2_c4_pages_reference(storage, page_size=page_size).float()
    rotated_query = (
        (query[:, 0].float() @ calibration.rotation.float()).to(torch.bfloat16).float()
    )
    expected = torch.zeros_like(actual)
    for batch_index in range(2):
        for position in range(int(seq_lens[batch_index])):
            page_slot, in_page = divmod(position, page_size)
            page_id = int(page_table[batch_index, page_slot])
            if page_id < 0:
                continue
            logits = decoded[page_id, in_page] @ rotated_query[batch_index].T
            expected[batch_index, position] = (
                torch.relu(logits) * weight[batch_index]
            ).sum()
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_capability_gate_is_exact_sm86(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    with pytest.raises(RuntimeError, match="exact NVIDIA SM86"):
        c4._require_exact_sm86(torch.device("cuda"))


def test_sm86_fused_writer_matches_reference_and_skips_padding() -> None:
    _require_sm86()
    calibration = _calibration(device="cuda:0", dense=False)
    keys = _keys(6, seed=41, device="cuda:0")
    locations = torch.tensor(
        [0, PAGE_SIZE - 1, PAGE_SIZE, PAGE_SIZE * 2 - 1, -1, PAGE_SIZE * 2],
        dtype=torch.int32,
        device="cuda:0",
    )
    write_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.uint8, device="cuda:0")
    actual = torch.full((2, PAGE_BYTES), 0xA5, dtype=torch.uint8, device="cuda:0")
    expected = actual.clone()
    store_oscar_int2_c4_indexer_cache(
        keys,
        actual,
        locations,
        calibration=calibration,
        write_mask=write_mask,
    )
    scatter_oscar_int2_c4_reference_paged(
        keys[:5], calibration, expected, locations[:5]
    )

    def gather_records(storage: torch.Tensor) -> torch.Tensor:
        records = []
        for location in locations[:4].tolist():
            page, in_page = divmod(location, PAGE_SIZE)
            code_start = in_page * CODES_BYTES_PER_TOKEN
            metadata_start = CODES_BYTES_PER_PAGE + in_page * METADATA_BYTES_PER_TOKEN
            records.append(
                torch.cat(
                    (
                        storage[
                            page,
                            code_start : code_start + CODES_BYTES_PER_TOKEN,
                        ],
                        storage[
                            page,
                            metadata_start : metadata_start + METADATA_BYTES_PER_TOKEN,
                        ],
                    )
                )
            )
        return torch.stack(records)

    actual_records = gather_records(actual)
    expected_records = gather_records(expected)
    actual_packed = actual_records[:, :CODES_BYTES_PER_TOKEN]
    expected_packed = expected_records[:, :CODES_BYTES_PER_TOKEN]

    def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                packed & 0x03,
                (packed >> 2) & 0x03,
                (packed >> 4) & 0x03,
                (packed >> 6) & 0x03,
            ),
            dim=-1,
        ).reshape(packed.shape[0], HEAD_DIM)

    actual_codes = unpack_codes(actual_packed)
    expected_codes = unpack_codes(expected_packed)
    code_delta = (actual_codes.to(torch.int16) - expected_codes.to(torch.int16)).abs()
    # BF16 HMMA and Torch's FP32 matmul can choose opposite sides of an exact
    # half-bin. The packing order itself remains exact, and only a few codes
    # may move by one quantum at those boundaries.
    assert code_delta.max().item() <= 1
    assert torch.count_nonzero(code_delta).item() <= 4
    repacked = (
        actual_codes[:, 0::4]
        | (actual_codes[:, 1::4] << 2)
        | (actual_codes[:, 2::4] << 4)
        | (actual_codes[:, 3::4] << 6)
    ).to(torch.uint8)
    assert torch.equal(repacked, actual_packed)

    actual_metadata = actual_records[:, CODES_BYTES_PER_TOKEN:].view(torch.float32)
    expected_metadata = expected_records[:, CODES_BYTES_PER_TOKEN:].view(torch.float32)
    torch.testing.assert_close(actual_metadata, expected_metadata, rtol=0, atol=2.0e-7)
    actual_decoded = dequantize_oscar_int2_c4_reference(actual_records).float()
    expected_decoded = dequantize_oscar_int2_c4_reference(expected_records).float()
    decoded_error = (actual_decoded - expected_decoded).abs()
    assert decoded_error.mean().item() < 2.0e-2
    assert decoded_error.max().item() <= expected_metadata[:, 0].max().item() + 2.0e-2

    # Every difference is confined to a requested record; padding, the
    # negative location, and the masked out-of-capacity row remain canaries.
    record_mismatches = torch.count_nonzero(actual_records != expected_records)
    assert torch.count_nonzero(actual != expected) == record_mismatches


def test_sm86_fused_scorer_matches_reference_without_cache_decode() -> None:
    _require_sm86()
    calibration = _calibration(device="cuda:0")
    raw_pages = _keys(3 * PAGE_SIZE, seed=43, device="cuda:0").reshape(
        3, PAGE_SIZE, HEAD_DIM
    )
    storage = pack_oscar_int2_c4_pages_reference(raw_pages, calibration)
    query = _query(2, seed=47, device="cuda:0")
    weight = torch.randn(
        (2, NUM_HEADS), generator=torch.Generator().manual_seed(53)
    ).to("cuda:0")
    seq_lens = torch.tensor([117, 91], dtype=torch.int32, device="cuda:0")
    page_table = torch.tensor([[2, 0], [1, -1]], dtype=torch.int32, device="cuda:0")
    out = torch.empty((2, 128), dtype=torch.float32, device="cuda:0")

    actual = oscar_int2_c4_paged_mqa_logits_triton(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        None,
        128,
        True,
        calibration=calibration,
        out=out,
        rotated_query_out=torch.empty(
            (query.shape[0], NUM_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=query.device,
        ),
    )
    expected = oscar_int2_c4_paged_mqa_logits_reference(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        calibration,
        128,
    )
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=0.75)


def test_sm86_fused_rope_rotation_scorer_matches_two_kernel_path() -> None:
    _require_sm86()
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

    calibration = _calibration(device="cuda:0", dense=False)
    raw_pages = _keys(4 * PAGE_SIZE, seed=401, device="cuda:0").reshape(
        4, PAGE_SIZE, HEAD_DIM
    )
    storage = pack_oscar_int2_c4_pages_reference(raw_pages, calibration)
    query = _query(2, seed=403, device="cuda:0")
    frequencies = _rope_frequencies(64, device="cuda:0")
    frequencies_real = torch.view_as_real(frequencies).flatten(-2)
    positions = torch.tensor([7, 41], dtype=torch.int64, device="cuda:0")
    weight = torch.randn(
        (2, NUM_HEADS), generator=torch.Generator().manual_seed(405)
    ).to("cuda:0")
    seq_lens = torch.tensor([239, 193], dtype=torch.int32, device="cuda:0")
    page_table = torch.tensor(
        [[3, 0, 1, 2], [2, 1, 0, -1]], dtype=torch.int32, device="cuda:0"
    )

    roped_query = query.clone()
    fused_rope_inplace(
        roped_query[..., -64:],
        None,
        frequencies,
        positions,
    )
    baseline_out = torch.empty((2, 256), dtype=torch.float32, device="cuda:0")
    baseline_rotated = torch.empty(
        (2, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda:0"
    )
    oscar_int2_c4_paged_mqa_logits_triton(
        roped_query,
        storage,
        weight,
        seq_lens,
        page_table,
        None,
        256,
        True,
        calibration=calibration,
        out=baseline_out,
        rotated_query_out=baseline_rotated,
    )

    fused_out = torch.empty_like(baseline_out)
    fused_rotated = torch.empty_like(baseline_rotated)
    oscar_int2_c4_paged_mqa_logits_triton(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        None,
        256,
        True,
        calibration=calibration,
        out=fused_out,
        rotated_query_out=fused_rotated,
        freqs_cis_real=frequencies_real,
        positions=positions,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused_rotated, baseline_rotated, rtol=0, atol=0)
    torch.testing.assert_close(fused_out, baseline_out, rtol=0, atol=0)


def test_sm86_fused_pipeline_elides_short_rows_and_replays_live_threshold() -> None:
    _require_sm86()
    calibration = _calibration(device="cuda:0", dense=False)
    max_seq_len = 640
    selection_topk = 512
    raw_pages = _keys(10 * PAGE_SIZE, seed=411, device="cuda:0").reshape(
        10, PAGE_SIZE, HEAD_DIM
    )
    storage = pack_oscar_int2_c4_pages_reference(raw_pages, calibration)
    query = _query(1, seed=413, device="cuda:0")
    frequencies = _rope_frequencies(32, device="cuda:0")
    frequencies_real = torch.view_as_real(frequencies).flatten(-2)
    positions = torch.tensor([9], dtype=torch.int32, device="cuda:0")
    weight = torch.randn(
        (1, NUM_HEADS), generator=torch.Generator().manual_seed(415)
    ).to("cuda:0")
    seq_lens = torch.tensor([511], dtype=torch.int32, device="cuda:0")
    page_table = torch.arange(10, dtype=torch.int32, device="cuda:0")[None, :]
    out = torch.full((1, max_seq_len), 12345.0, device="cuda:0")
    rotated = torch.full(
        (1, NUM_HEADS, HEAD_DIM),
        -321.0,
        dtype=torch.bfloat16,
        device="cuda:0",
    )

    def run() -> None:
        oscar_int2_c4_paged_mqa_logits_triton(
            query,
            storage,
            weight,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            False,
            calibration=calibration,
            out=out,
            rotated_query_out=rotated,
            freqs_cis_real=frequencies_real,
            positions=positions,
            selection_topk=selection_topk,
        )

    run()
    torch.cuda.synchronize()
    assert torch.all(out == 12345.0)
    assert torch.all(rotated == -321.0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    seq_lens.fill_(513)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(torch.isfinite(out[:, :513]))
    assert torch.any(out[:, :513] != 12345.0)
    assert torch.any(rotated != -321.0)

    out.fill_(777.0)
    rotated.fill_(222.0)
    seq_lens.fill_(512)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out == 777.0)
    assert torch.all(rotated == 222.0)

    # Prefill graph tiers can be statically narrower than the configured C4
    # selection width.  That is an all-live-token tier, not an invalid top-k.
    short_max_seq_len = 256
    short_out = torch.full((1, short_max_seq_len), 4567.0, device="cuda:0")
    short_rotated = torch.full_like(rotated, -765.0)
    oscar_int2_c4_paged_mqa_logits_triton(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        None,
        short_max_seq_len,
        False,
        calibration=calibration,
        out=short_out,
        rotated_query_out=short_rotated,
        freqs_cis_real=frequencies_real,
        positions=positions,
        selection_topk=2_048,
    )
    torch.cuda.synchronize()
    assert torch.all(short_out == 4567.0)
    assert torch.all(short_rotated == -765.0)


def test_sm86_writer_and_scorer_replay_in_one_cuda_graph() -> None:
    _require_sm86()
    calibration = _calibration(device="cuda:0", dense=False)
    keys = _keys(2, seed=59, device="cuda:0")
    query = _query(1, seed=61, device="cuda:0")
    locations = torch.tensor([0, 1], dtype=torch.int32, device="cuda:0")
    storage = torch.zeros((1, PAGE_BYTES), dtype=torch.uint8, device="cuda:0")
    weight = torch.randn(
        (1, NUM_HEADS), generator=torch.Generator().manual_seed(67)
    ).to("cuda:0")
    seq_lens = torch.tensor([2], dtype=torch.int32, device="cuda:0")
    page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda:0")
    out = torch.empty((1, PAGE_SIZE), dtype=torch.float32, device="cuda:0")
    rotated_query_out = torch.empty(
        (1, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda:0"
    )

    store_oscar_int2_c4_indexer_cache(keys, storage, locations, calibration=calibration)
    oscar_int2_c4_paged_mqa_logits_triton(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        None,
        PAGE_SIZE,
        True,
        calibration=calibration,
        out=out,
        rotated_query_out=rotated_query_out,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    rotated_query_pointer = rotated_query_out.data_ptr()
    with torch.cuda.graph(graph):
        store_oscar_int2_c4_indexer_cache(
            keys, storage, locations, calibration=calibration
        )
        oscar_int2_c4_paged_mqa_logits_triton(
            query,
            storage,
            weight,
            seq_lens,
            page_table,
            None,
            PAGE_SIZE,
            True,
            calibration=calibration,
            out=out,
            rotated_query_out=rotated_query_out,
        )

    keys.copy_(_keys(2, seed=71, device="cuda:0"))
    query.copy_(_query(1, seed=73, device="cuda:0"))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert rotated_query_out.data_ptr() == rotated_query_pointer
    expected = oscar_int2_c4_paged_mqa_logits_reference(
        query,
        storage,
        weight,
        seq_lens,
        page_table,
        calibration,
        PAGE_SIZE,
    )
    torch.testing.assert_close(out, expected, rtol=2.0e-2, atol=0.75)
    expected_rotated_query = (query[:, 0].float() @ calibration.rotation.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        rotated_query_out, expected_rotated_query, rtol=2.0e-2, atol=2.0e-2
    )


def test_sm86_masked_c4_writer_replay_reads_live_nonboundary_mask() -> None:
    _require_sm86()
    calibration = _calibration(device="cuda:0", dense=False)
    keys = _keys(4, seed=79, device="cuda:0")
    locations = torch.arange(4, dtype=torch.int32, device="cuda:0")
    write_mask = torch.zeros(4, dtype=torch.bool, device="cuda:0")
    storage = torch.full((1, PAGE_BYTES), 0xA5, dtype=torch.uint8, device="cuda:0")

    def run() -> None:
        store_oscar_int2_c4_indexer_cache(
            keys,
            storage,
            locations,
            calibration=calibration,
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

    keys.copy_(_keys(4, seed=83, device="cuda:0"))
    write_mask[2] = True
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.full_like(storage, 0xA5)
    scatter_oscar_int2_c4_reference_paged(
        keys[2:3], calibration, expected, locations[2:3]
    )
    assert torch.equal(storage, expected)

    persisted = storage.clone()
    keys.copy_(_keys(4, seed=89, device="cuda:0"))
    write_mask.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(storage, persisted)
