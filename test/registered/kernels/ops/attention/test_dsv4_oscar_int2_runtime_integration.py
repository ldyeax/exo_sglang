from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
    CLIP_MODE,
    GROUP_SIZE,
    HEAD_DIM,
    PAGE_SIZE,
    STORAGE_BYTES_PER_TOKEN,
    validate_oscar_int2_c4_calibration,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
    FORMAT_NAME as C4_SCORER_FORMAT,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
    MASKED_WRITER_EXECUTION as C4_MASKED_WRITER_EXECUTION,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
    QUERY_ROTATION_EXECUTION as C4_QUERY_ROTATION_EXECUTION,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_decode import (
    SPLIT_HISTORY_EXECUTION,
    SPLIT_HISTORY_MAX_PARTIAL_ROWS,
    SPLIT_HISTORY_SPLIT_MAP,
    SPLIT_HISTORY_WORKSPACE_BYTES,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    FORMAT_NAME as SHARED_LATENT_FORMAT,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    MASKED_WRITER_EXECUTION as SHARED_MASKED_WRITER_EXECUTION,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    _oscar_store_locations_and_mask,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    C4Indexer,
    _OscarC4LogitsWorkspace,
    _OscarC4RotatedQueryWorkspace,
)
from sglang.srt.managers.scheduler import (
    _DSV4_OSCAR_WORKER_INIT_INFO_KEYS,
    Scheduler,
    _get_dsv4_oscar_split_history_telemetry,
    _get_dsv4_oscar_wo_a_absorption_telemetry,
    _get_dsv4_oscar_worker_telemetry,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4LayerItem,
    DeepSeekV4TokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm86() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("OSCAR-INT2 runtime integration targets NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("OSCAR-INT2 runtime integration is fail-closed to SM86")


def _calibration(*, layer_id: int):
    permutation = torch.arange(HEAD_DIM - 1, -1, -1, device="cuda")
    signs = torch.where(
        torch.arange(HEAD_DIM, device="cuda") % 3 == 0,
        torch.tensor(-1.0, device="cuda"),
        torch.tensor(1.0, device="cuda"),
    ).to(torch.bfloat16)
    rotation = torch.zeros((HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    rotation[torch.arange(HEAD_DIM, device="cuda"), permutation] = signs
    clip_ratio = 0.95
    return validate_oscar_int2_c4_calibration(
        rotation,
        torch.tensor([2.5], dtype=torch.float32, device="cuda"),
        torch.tensor([clip_ratio], dtype=torch.float32, device="cuda"),
        torch.tensor(
            [min(int(clip_ratio * GROUP_SIZE), GROUP_SIZE - 1)],
            dtype=torch.int16,
            device="cuda",
        ),
        layer_id=layer_id,
        clip_mode=CLIP_MODE,
        clip_provenance="runtime-integration-test",
    )


def _identity_freqs(num_positions: int) -> torch.Tensor:
    interleaved = torch.zeros((num_positions, 64), dtype=torch.float32, device="cuda")
    interleaved[:, 0::2] = 1.0
    return torch.view_as_complex(interleaved.view(num_positions, 32, 2))


class _FixedProjection(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, _: torch.Tensor):
        return self.output.clone(), None


class _FakeCompressPlan:
    def __init__(self, raw: torch.Tensor, *, is_decode: bool):
        self.raw = raw
        self.is_decode = is_decode

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 1:
            raise IndexError(index)
        return self.raw


class _BytesPerTokenPool:
    def __init__(
        self,
        value: int,
        *,
        use_oscar_int2_cache: bool = False,
        layer_num: int = 0,
    ) -> None:
        self.value = value
        self.use_oscar_int2_cache = use_oscar_int2_cache
        self.layer_num = layer_num

    def get_bytes_per_token(self) -> int:
        return self.value


def _target_absorption_state() -> dict[str, object]:
    return {
        "enabled": True,
        "consumer_role": "target_compressed",
        "target_only": True,
        "applied": True,
        "apply_count": 1,
        "artifact_sha256": "a" * 64,
        "admission_sha256": "b" * 64,
        "expected_local_compressed_layer_ids": [2],
        "absorbed_local_layer_ids": [2],
        "runtime_restore_skipped_layer_ids": [2],
        "all_local_target_compressed_layers_absorbed": True,
        "all_local_target_compressed_layers_skip_runtime_restore": True,
        "weight_dtype": "bfloat16",
        "head_layout": "per-head-nope448-rope64",
        "fold_orientation": "wo_a_nope@rotation",
        "rope_columns_unchanged": True,
    }


def test_runtime_indexer_pool_accounts_only_the_40_byte_oscar_layout() -> None:
    pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
    pool.index_head_dim = HEAD_DIM
    pool.use_bf16_cache = False
    pool.use_int4_cache = False
    pool.use_oscar_int2_cache = True
    pool.use_fp4_indexer = False

    assert pool.get_bytes_per_token() == STORAGE_BYTES_PER_TOKEN == 40


def test_oscar_prefill_and_decode_masks_reject_padded_plan_rows() -> None:
    out_loc = torch.tensor([100, 200], dtype=torch.int32)
    prefill_raw = torch.tensor(
        [[4, 1, 0, 0], [-1, -1, -1, -1], [8, 0, 0, 0]],
        dtype=torch.int32,
    )
    locations, mask = _oscar_store_locations_and_mask(
        _FakeCompressPlan(prefill_raw, is_decode=False), out_loc, 4
    )
    assert torch.equal(locations, torch.tensor([200, 100, 100], dtype=torch.int32))
    assert torch.equal(mask, torch.tensor([True, False, True]))

    decode_raw = torch.tensor(
        [[4, 0, 0, 0], [5, 0, 0, 0], [-1, 0, 0, 0]], dtype=torch.int32
    )
    decode_locations, decode_mask = _oscar_store_locations_and_mask(
        _FakeCompressPlan(decode_raw, is_decode=True),
        torch.tensor([300, 301, 302], dtype=torch.int32),
        4,
    )
    assert torch.equal(
        decode_locations, torch.tensor([300, 301, 302], dtype=torch.int32)
    )
    assert torch.equal(decode_mask, torch.tensor([True, False, False]))


@pytest.mark.parametrize(
    ("oscar_storage_active", "c4_layer_num"),
    [(True, 0), (True, 1), (False, 1)],
)
def test_scheduler_reports_canonical_oscar_formats_only_when_active(
    oscar_storage_active: bool,
    c4_layer_num: int,
) -> None:
    absorption_state = _target_absorption_state()
    indexer_pool = _BytesPerTokenPool(
        STORAGE_BYTES_PER_TOKEN,
        use_oscar_int2_cache=True,
        layer_num=c4_layer_num,
    )
    kv_pool = SimpleNamespace(
        use_int4_storage=False,
        use_int4_indexer_storage=False,
        use_oscar_int2_storage=oscar_storage_active,
        use_selective_c128_bf16_storage=False,
        swa_kv_pool=_BytesPerTokenPool(1_024),
        c4_kv_pool=_BytesPerTokenPool(272),
        c128_kv_pool=_BytesPerTokenPool(272),
        c4_indexer_kv_pool=indexer_pool,
        kv_storage_mode="oscar_int2_asymmetric+protected_swa_bfloat16",
        oscar_artifact_sha256=absorption_state["artifact_sha256"],
        oscar_admission_sha256=absorption_state["admission_sha256"],
    )
    scheduler = object.__new__(Scheduler)
    scheduler.max_total_num_tokens = 1_024
    scheduler.max_req_input_len = 1_023
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            token_to_kv_pool=kv_pool,
            get_dsv4_oscar_wo_a_absorption_state=lambda: dict(absorption_state),
        )
    )

    info = scheduler.get_init_info()

    c4_scorer_active = oscar_storage_active and c4_layer_num > 0
    assert info["dsv4_oscar_int2_kv_storage"] is oscar_storage_active
    assert info["dsv4_oscar_algorithm"] == (
        SHARED_LATENT_FORMAT if oscar_storage_active else ""
    )
    assert info["dsv4_oscar_masked_writer_execution"] == (
        SHARED_MASKED_WRITER_EXECUTION if oscar_storage_active else ""
    )
    assert info["dsv4_oscar_wo_a_absorption_state"] == (
        absorption_state if oscar_storage_active else {}
    )
    assert info["dsv4_oscar_c4_scorer"] is c4_scorer_active
    assert info["dsv4_oscar_c4_scorer_algorithm"] == (
        C4_SCORER_FORMAT if c4_scorer_active else ""
    )
    assert info["dsv4_oscar_c4_masked_writer_execution"] == (
        C4_MASKED_WRITER_EXECUTION if c4_scorer_active else ""
    )
    assert info["dsv4_oscar_c4_query_rotation_execution"] == (
        C4_QUERY_ROTATION_EXECUTION if c4_scorer_active else ""
    )
    assert info["dsv4_oscar_int2_split_history"] is False
    assert info["dsv4_oscar_int2_split_history_execution"] == ""
    assert info["dsv4_oscar_int2_split_history_split_map"] == {}
    assert info["dsv4_oscar_int2_split_history_workspace_bytes"] == 0
    assert info["dsv4_oscar_int2_split_history_prefill_enabled"] is False


def test_scheduler_split_history_telemetry_is_bound_to_live_backend() -> None:
    state = {
        "enabled": True,
        "execution": SPLIT_HISTORY_EXECUTION,
        "split_map": {
            str(num_tokens): split_count
            for num_tokens, split_count in SPLIT_HISTORY_SPLIT_MAP.items()
        },
        "workspace_bytes": SPLIT_HISTORY_WORKSPACE_BYTES,
        "max_partial_rows": SPLIT_HISTORY_MAX_PARTIAL_ROWS,
        "sink_owner": "stage2-exactly-once",
        "prefill_enabled": False,
        "fixed_address": True,
        "workspace_address": 0x12340000,
    }
    runner = SimpleNamespace(
        attn_backend=SimpleNamespace(
            get_dsv4_oscar_int2_split_history_telemetry=lambda: dict(state)
        )
    )
    pool = SimpleNamespace(oscar_consumer_role="target_compressed")

    with envs.SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY.override(True):
        actual = _get_dsv4_oscar_split_history_telemetry(
            runner,
            pool,
            oscar_storage_active=True,
        )
        assert actual == state

        state["workspace_bytes"] -= 4
        with pytest.raises(RuntimeError, match="contract mismatch"):
            _get_dsv4_oscar_split_history_telemetry(
                runner,
                pool,
                oscar_storage_active=True,
            )


def test_scheduler_split_history_opt_in_requires_authoritative_backend() -> None:
    with (
        envs.SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY.override(True),
        pytest.raises(RuntimeError, match="authoritative backend"),
    ):
        _get_dsv4_oscar_split_history_telemetry(
            SimpleNamespace(),
            SimpleNamespace(oscar_consumer_role="target_compressed"),
            oscar_storage_active=True,
        )


def test_oscar_wo_a_telemetry_fails_closed_without_authoritative_getter() -> None:
    pool = SimpleNamespace(
        oscar_artifact_sha256="a" * 64,
        oscar_admission_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="authoritative wo_a"):
        _get_dsv4_oscar_wo_a_absorption_telemetry(
            SimpleNamespace(), pool, oscar_storage_active=True
        )


def test_oscar_wo_a_telemetry_rejects_incomplete_runtime_skip_coverage() -> None:
    state = _target_absorption_state()
    state["runtime_restore_skipped_layer_ids"] = []
    runner = SimpleNamespace(get_dsv4_oscar_wo_a_absorption_state=lambda: state)
    pool = SimpleNamespace(
        oscar_artifact_sha256="a" * 64,
        oscar_admission_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="layer coverage"):
        _get_dsv4_oscar_wo_a_absorption_telemetry(
            runner, pool, oscar_storage_active=True
        )


def test_oscar_worker_telemetry_binds_rank_and_complete_init_info() -> None:
    init_info = {key: f"value:{key}" for key in _DSV4_OSCAR_WORKER_INIT_INFO_KEYS}
    init_info["dsv4_oscar_int2_kv_storage"] = True
    parallel_state = cast(
        ParallelState,
        SimpleNamespace(
            gpu_id=1,
            tp_rank=0,
            pp_rank=1,
            dp_rank=0,
        ),
    )

    telemetry = _get_dsv4_oscar_worker_telemetry(parallel_state, init_info)

    assert telemetry["pid"] > 0
    assert telemetry["gpu_id"] == 1
    assert telemetry["tp_rank"] == 0
    assert telemetry["pp_rank"] == 1
    assert telemetry["dp_rank"] == 0
    assert all(telemetry[key] == init_info[key] for key in init_info)

    init_info.pop("dsv4_oscar_admission_sha256")
    with pytest.raises(RuntimeError, match="incomplete"):
        _get_dsv4_oscar_worker_telemetry(parallel_state, init_info)


def test_oscar_c4_logits_workspace_reuses_contiguous_storage() -> None:
    workspace = _OscarC4LogitsWorkspace()
    reference = torch.empty(1)

    first = workspace.acquire(reference, query_rows=8, max_sequence_length=16)
    smaller = workspace.acquire(reference, query_rows=3, max_sequence_length=16)
    repeated = workspace.acquire(reference, query_rows=8, max_sequence_length=16)

    assert first.shape == (8, 16)
    assert smaller.shape == (3, 16)
    assert first.dtype == smaller.dtype == torch.float32
    assert first.is_contiguous() and smaller.is_contiguous()
    assert first.data_ptr() == smaller.data_ptr() == repeated.data_ptr()


def test_oscar_c4_logits_workspace_never_grows_during_graph_capture() -> None:
    workspace = _OscarC4LogitsWorkspace()
    reference = torch.empty(1)
    warmed = workspace.acquire(reference, query_rows=4, max_sequence_length=16)

    with patch.object(workspace, "_is_capturing", return_value=True):
        replay = workspace.acquire(reference, query_rows=4, max_sequence_length=16)
        with pytest.raises(RuntimeError, match="must be warmed"):
            workspace.acquire(reference, query_rows=16, max_sequence_length=16)

    assert replay.data_ptr() == warmed.data_ptr()


def test_oscar_c4_rotated_query_workspace_reuses_contiguous_storage() -> None:
    workspace = _OscarC4RotatedQueryWorkspace()
    reference = torch.empty((8, 1, 64, 128), dtype=torch.bfloat16)

    first = workspace.acquire(reference, query_rows=8)
    smaller = workspace.acquire(reference, query_rows=3)
    repeated = workspace.acquire(reference, query_rows=8)

    assert first.shape == (8, 64, 128)
    assert smaller.shape == (3, 64, 128)
    assert first.dtype == smaller.dtype == torch.bfloat16
    assert first.is_contiguous() and smaller.is_contiguous()
    assert first.data_ptr() == smaller.data_ptr() == repeated.data_ptr()


def test_oscar_c4_rotated_query_workspace_never_grows_during_capture() -> None:
    workspace = _OscarC4RotatedQueryWorkspace()
    reference = torch.empty((4, 1, 64, 128), dtype=torch.bfloat16)
    warmed = workspace.acquire(reference, query_rows=4)

    with patch.object(_OscarC4LogitsWorkspace, "_is_capturing", return_value=True):
        replay = workspace.acquire(reference, query_rows=4)
        with pytest.raises(RuntimeError, match="must be warmed"):
            workspace.acquire(reference, query_rows=16)

    assert replay.data_ptr() == warmed.data_ptr()


def test_runtime_pool_routes_layer_bound_oscar_writer() -> None:
    _require_sm86()
    layer_id = 2
    calibration = _calibration(layer_id=layer_id)
    indexer_pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
    indexer_pool.page_size = PAGE_SIZE
    indexer_pool.start_layer = 0
    indexer_pool.use_bf16_cache = False
    indexer_pool.use_int4_cache = False
    indexer_pool.use_oscar_int2_cache = True
    indexer_pool.index_k_with_scale_buffer = [
        torch.zeros(
            (2, PAGE_SIZE * STORAGE_BYTES_PER_TOKEN),
            dtype=torch.uint8,
            device="cuda",
        )
    ]

    token_pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    token_pool.use_oscar_int2_storage = True
    token_pool.oscar_consumer_role = "target_compressed_history"
    token_pool.oscar_c4_calibrations = {layer_id: calibration}
    token_pool.c4_indexer_kv_pool = indexer_pool
    token_pool.layer_mapping = [None, None, DeepSeekV4LayerItem(4, 0, None)]
    keys = torch.randn((4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    locations = torch.arange(4, dtype=torch.int32, device="cuda")

    token_pool.set_index_k_fused(layer_id, locations, keys)

    assert indexer_pool.index_k_with_scale_buffer[0].count_nonzero().item() > 0
    with pytest.raises(ValueError, match="does not cover model layer"):
        token_pool.get_oscar_c4_calibration(3)


def test_oscar_query_path_preserves_bf16_and_replays_in_cuda_graph() -> None:
    _require_sm86()
    batch_size = 1
    projected = torch.randn(
        (batch_size, 64, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    head_weight = torch.linspace(
        0.25, 1.25, 64, dtype=torch.bfloat16, device="cuda"
    ).view(batch_size, 64)
    positions = torch.tensor([3], dtype=torch.int32, device="cuda")
    q_lora = torch.zeros((batch_size, 1), dtype=torch.bfloat16, device="cuda")
    indexer = C4Indexer.__new__(C4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.wq_b = _FixedProjection(projected)
    indexer.n_local_heads = 64
    indexer.head_dim = HEAD_DIM
    indexer.rope_head_dim = 64
    indexer.layer_id = 2
    indexer.freqs_cis = _identity_freqs(8)
    indexer.weight_scale = 0.125
    indexer.use_fp4_indexer = False
    forward_batch = SimpleNamespace()

    with envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.override(True):
        query, weights = indexer.compute_q(
            q_lora, positions, head_weight, forward_batch=forward_batch
        )
        assert query.dtype == torch.bfloat16
        assert torch.equal(query, projected)
        torch.testing.assert_close(
            weights,
            head_weight.float().mul(0.125).unsqueeze(-1),
            rtol=0,
            atol=0,
        )

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_query, graph_weights = indexer.compute_q(
                q_lora, positions, head_weight, forward_batch=forward_batch
            )
        projected.add_(0.5)
        head_weight.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()

    assert torch.equal(graph_query, projected)
    torch.testing.assert_close(
        graph_weights,
        head_weight.float().mul(0.125).unsqueeze(-1),
        rtol=0,
        atol=0,
    )
