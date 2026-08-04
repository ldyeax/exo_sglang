from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import kt_hotspot
from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    _Mxfp4HotspotStaging,
    create_kt_config_from_server_args,
)


@pytest.fixture(autouse=True)
def reset_hotspot_registry() -> None:
    kt_hotspot.reset_hotspot_state_for_tests()
    yield
    kt_hotspot.reset_hotspot_state_for_tests()


def test_assign_experts_to_slots_retains_hits_and_fills_misses() -> None:
    assigned = kt_hotspot.assign_experts_to_slots(
        current_slot_experts=(8, 3, 6, 1),
        requested_experts=(9, 6, 8, 4),
    )

    assert assigned == (8, 4, 6, 9)


def test_assign_experts_to_slots_rejects_width_and_duplicates() -> None:
    with pytest.raises(ValueError, match="fixed GPU slot count"):
        kt_hotspot.assign_experts_to_slots((1, 2), (1,))
    with pytest.raises(ValueError, match="duplicate"):
        kt_hotspot.assign_experts_to_slots((1, 2), (3, 3))


def test_load_hotspot_plan_rejects_cross_rank_gpu_overlap(tmp_path) -> None:
    plan_path = tmp_path / "overlap.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [
                    [[True, False, True, False]],
                    [[True, True, False, False]],
                ],
                dtype=torch.bool,
            )
        },
        plan_path,
    )

    with pytest.raises(ValueError, match="multiple EP ranks"):
        kt_hotspot.load_hotspot_plan(
            str(plan_path),
            ep_rank=0,
            ep_size=2,
            expected_num_layers=1,
            expected_num_experts=4,
        )


def test_parallel_identity_uses_runtime_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(moe_ep_rank=1, moe_ep_size=2),
    )

    assert kt_hotspot._parallel_identity() == (1, 2)


def test_hotspot_config_keeps_rank_owned_gpu_experts_in_cpu_shadow(
    monkeypatch, tmp_path
) -> None:
    plan_path = tmp_path / "hybrid.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [
                    [[True, False, False, False]],
                    [[False, True, False, False]],
                ],
                dtype=torch.bool,
            ),
            "cpu_expert_ids_by_rank": (
                torch.tensor([[2]], dtype=torch.int64),
                torch.tensor([[3]], dtype=torch.int64),
            ),
        },
        plan_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(plan_path))
    monkeypatch.setenv("SGLANG_KT_HOTSPOT_EXPERT_CACHE", "1")
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=1, n_routed_experts=4)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=1,
        kt_cpuinfer=56,
        kt_threadpool_count=2,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    config = create_kt_config_from_server_args(server_args, layer_idx=0)

    assert config is not None
    assert config.hotspot_cpu_shadow
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([1, 3], dtype=torch.int64),
    )


class _FakeHotspotMethod:
    def __init__(self, layer_idx: int, slots: tuple[int, ...]) -> None:
        self.kt_config = SimpleNamespace(layer_idx=layer_idx)
        self.global_num_experts = 4
        self.gpu_index_to_logical = torch.tensor(slots, dtype=torch.int32)
        self.commits: list[tuple[int, ...]] = []

    def validate_hotspot_slots(self, slots: tuple[int, ...]) -> None:
        assert len(slots) == 2

    def commit_hotspot_slots(
        self, slots: tuple[int, ...], *, force_reload: bool = False
    ) -> int:
        self.commits.append(slots)
        self.gpu_index_to_logical = torch.tensor(slots, dtype=torch.int32)
        return 128


def test_apply_hotspot_plan_dry_run_and_commit(monkeypatch, tmp_path) -> None:
    plan_path = tmp_path / "hotspot.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [[[True, False, True, False], [False, True, False, True]]],
                dtype=torch.bool,
            )
        },
        plan_path,
    )
    methods = (
        _FakeHotspotMethod(0, (0, 1)),
        _FakeHotspotMethod(1, (0, 1)),
    )
    for method in methods:
        kt_hotspot.register_hotspot_method(method)
    monkeypatch.setattr(kt_hotspot, "_parallel_identity", lambda: (0, 1))
    monkeypatch.setattr(kt_hotspot, "_synchronize_hotspot_ranks", lambda: None)
    monkeypatch.setattr(kt_hotspot, "_collect_rank_errors", lambda error: [error])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    dry_run = kt_hotspot.apply_hotspot_plan(
        plan_path=str(plan_path), generation=0, dry_run=True
    )
    assert dry_run["total_swaps"] == 2
    assert all(not method.commits for method in methods)

    committed = kt_hotspot.apply_hotspot_plan(
        plan_path=str(plan_path), generation=1, dry_run=False
    )

    assert methods[0].commits == [(0, 2)]
    assert methods[1].commits == [(3, 1)]
    assert committed["copied_bytes"] == 256
    assert committed["last_committed_generation"] == 1

    with pytest.raises(RuntimeError, match="increase monotonically"):
        kt_hotspot.apply_hotspot_plan(
            plan_path=str(plan_path), generation=1, dry_run=False
        )


def test_apply_hotspot_plan_force_restores_partially_written_layer(
    monkeypatch, tmp_path
) -> None:
    plan_path = tmp_path / "hotspot.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [[[False, False, True, True]]], dtype=torch.bool
            )
        },
        plan_path,
    )

    class PartiallyFailingMethod(_FakeHotspotMethod):
        def __init__(self) -> None:
            super().__init__(0, (0, 1))
            self.slot_bytes = [0, 1]
            self.forced_restores: list[tuple[int, ...]] = []

        def commit_hotspot_slots(
            self, slots: tuple[int, ...], *, force_reload: bool = False
        ) -> int:
            self.commits.append(slots)
            if force_reload:
                self.forced_restores.append(slots)
                self.slot_bytes[:] = slots
                self.gpu_index_to_logical = torch.tensor(slots, dtype=torch.int32)
                return 256

            # Model a copy failing after one slot changed but before its logical
            # routing table was committed.
            self.slot_bytes[0] = slots[0]
            raise RuntimeError("injected partial promotion")

    method = PartiallyFailingMethod()
    kt_hotspot.register_hotspot_method(method)
    monkeypatch.setattr(kt_hotspot, "_parallel_identity", lambda: (0, 1))
    monkeypatch.setattr(kt_hotspot, "_synchronize_hotspot_ranks", lambda: None)
    monkeypatch.setattr(kt_hotspot, "_collect_rank_errors", lambda error: [error])
    monkeypatch.setattr(kt_hotspot, "_collect_rank_values", lambda value: [value])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="injected partial promotion"):
        kt_hotspot.apply_hotspot_plan(
            plan_path=str(plan_path), generation=1, dry_run=False
        )

    assert method.slot_bytes == [0, 1]
    assert method.forced_restores == [(0, 1)]
    torch.testing.assert_close(
        method.gpu_index_to_logical,
        torch.tensor([0, 1], dtype=torch.int32),
    )


def _make_cpu_hotspot_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[KTEPWrapperMethod, tuple[torch.Tensor, ...], list[str]]:
    w13 = torch.zeros((2, 3, 2), dtype=torch.uint8)
    # _hotspot_mxfp4_storages normalizes UE8M0 physical storage to bytes.
    w13_scale = torch.zeros((2, 2, 1), dtype=torch.uint8)
    w2 = torch.zeros((2, 4, 2), dtype=torch.uint8)
    w2_scale = torch.zeros((2, 2, 2), dtype=torch.uint8)
    staging = _Mxfp4HotspotStaging(
        host_w13=torch.empty((2, 3), dtype=torch.uint8),
        host_w13_scale=torch.empty((1, 2), dtype=torch.bfloat16),
        host_w2=torch.empty((2, 4), dtype=torch.uint8),
        host_w2_scale=torch.empty((2, 2), dtype=torch.bfloat16),
        device_w13=torch.empty((2, 3), dtype=torch.uint8),
        device_w13_scale_bf16=torch.empty((1, 2), dtype=torch.bfloat16),
        device_w13_scale_e8m0=torch.empty((1, 2), dtype=torch.float8_e8m0fnu),
        device_w2=torch.empty((2, 4), dtype=torch.uint8),
        device_w2_scale_bf16=torch.empty((2, 2), dtype=torch.bfloat16),
        device_w2_scale_e8m0=torch.empty((2, 2), dtype=torch.float8_e8m0fnu),
    )
    events: list[str] = []

    class FakeWrapper:
        def submit_write_weight_scale_to_buffer(
            self, gpu_tp_count: int, expert_id: int, *pointers: list[int]
        ) -> None:
            assert gpu_tp_count == 1
            assert pointers == (
                [staging.host_w13.data_ptr()],
                [staging.host_w13_scale.data_ptr()],
                [staging.host_w2.data_ptr()],
                [staging.host_w2_scale.data_ptr()],
            )
            events.append(f"submit:{expert_id}")
            staging.host_w13.fill_(expert_id + 1)
            staging.host_w13_scale.fill_(2 ** expert_id)
            staging.host_w2.fill_(expert_id + 11)
            staging.host_w2_scale.fill_(2 ** (expert_id + 1))

        def sync_write_weight_scale_to_buffer(self) -> None:
            events.append("export_sync")

    class FakeStream:
        def synchronize(self) -> None:
            events.append("stream_sync")

    method = object.__new__(KTEPWrapperMethod)
    method.cpu_expert_ids = torch.tensor([10, 11], dtype=torch.int64)
    method.wrapper = FakeWrapper()
    method.kt_config = SimpleNamespace(layer_idx=7)
    method._hotspot_mxfp4_storages = lambda _layer: (
        w13,
        w13_scale,
        w2,
        w2_scale,
    )
    monkeypatch.setattr(
        kt_ep_wrapper,
        "_get_mxfp4_hotspot_staging",
        lambda **_kwargs: staging,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: FakeStream())
    return method, (w13, w13_scale, w2, w2_scale), events


@pytest.mark.parametrize(
    ("w13_name", "w13_precision_name", "w2_name", "w2_precision_name"),
    (
        (
            "_dsv4_tk_w13",
            "_dsv4_tk_w13_precision",
            "_dsv4_tk_w2",
            "_dsv4_tk_w2_precision",
        ),
        ("_v4_tk_w13", "_v4_tk_w13_pcg", "_v4_tk_w2", "_v4_tk_w2_pcg"),
    ),
)
@pytest.mark.parametrize(
    ("scale_semantic_dtype", "scale_storage_dtype"),
    (
        (torch.float8_e8m0fnu, torch.float8_e8m0fnu),
        (torch.float8_e8m0fnu, torch.uint8),
        # matmul_ogs canonicalizes the live SM86 scale Tensor to this form.
        (torch.uint8, torch.uint8),
    ),
)
def test_hotspot_resolves_both_portable_mxfp4_attribute_schemes(
    w13_name: str,
    w13_precision_name: str,
    w2_name: str,
    w2_precision_name: str,
    scale_semantic_dtype: torch.dtype,
    scale_storage_dtype: torch.dtype,
) -> None:
    strided_layout_type = type("StridedLayout", (), {})

    def wrapped(
        tensor: torch.Tensor, *, semantic_dtype: torch.dtype | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            data=tensor,
            dtype=semantic_dtype if semantic_dtype is not None else tensor.dtype,
            storage=SimpleNamespace(layout=strided_layout_type()),
        )

    expected = (
        torch.empty((2, 3, 2), dtype=torch.uint8),
        torch.empty((2, 2, 1), dtype=scale_storage_dtype),
        torch.empty((2, 4, 2), dtype=torch.uint8),
        torch.empty((2, 2, 2), dtype=scale_storage_dtype),
    )
    layer = SimpleNamespace()
    setattr(layer, w13_name, wrapped(expected[0]))
    setattr(
        layer,
        w13_precision_name,
        SimpleNamespace(
            weight_scale=wrapped(
                expected[1], semantic_dtype=scale_semantic_dtype
            )
        ),
    )
    setattr(layer, w2_name, wrapped(expected[2]))
    setattr(
        layer,
        w2_precision_name,
        SimpleNamespace(
            weight_scale=wrapped(
                expected[3], semantic_dtype=scale_semantic_dtype
            )
        ),
    )
    method = object.__new__(KTEPWrapperMethod)

    actual = method._hotspot_mxfp4_storages(layer)

    assert actual[0] is expected[0]
    assert actual[2] is expected[2]
    for byte_view, physical_storage in (
        (actual[1], expected[1]),
        (actual[3], expected[3]),
    ):
        assert byte_view.dtype == torch.uint8
        assert byte_view.data_ptr() == physical_storage.data_ptr()
        assert byte_view.shape == physical_storage.shape
        assert byte_view.stride() == physical_storage.stride()
        assert byte_view.storage_offset() == physical_storage.storage_offset()


@pytest.mark.parametrize(
    ("semantic_dtype", "storage_dtype"),
    (
        (torch.int8, torch.uint8),
        (torch.uint8, torch.float8_e8m0fnu),
        (torch.float8_e8m0fnu, torch.int8),
    ),
)
def test_hotspot_rejects_non_e8m0_scale_representation(
    semantic_dtype: torch.dtype,
    storage_dtype: torch.dtype,
) -> None:
    strided_layout_type = type("StridedLayout", (), {})

    def wrapped(
        tensor: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            data=tensor,
            dtype=dtype if dtype is not None else tensor.dtype,
            storage=SimpleNamespace(layout=strided_layout_type()),
        )

    scale = wrapped(
        torch.empty((2, 2, 1), dtype=storage_dtype), dtype=semantic_dtype
    )
    layer = SimpleNamespace(
        _dsv4_tk_w13=wrapped(torch.empty((2, 3, 2), dtype=torch.uint8)),
        _dsv4_tk_w13_precision=SimpleNamespace(weight_scale=scale),
        _dsv4_tk_w2=wrapped(torch.empty((2, 4, 2), dtype=torch.uint8)),
        _dsv4_tk_w2_precision=SimpleNamespace(
            weight_scale=wrapped(
                torch.empty((2, 2, 2), dtype=torch.uint8),
                dtype=torch.float8_e8m0fnu,
            )
        ),
    )
    method = object.__new__(KTEPWrapperMethod)

    with pytest.raises(RuntimeError, match="UE8M0 byte representation"):
        method._hotspot_mxfp4_storages(layer)


def test_hotspot_promotion_finishes_shared_staging_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, storages, events = _make_cpu_hotspot_promotion(monkeypatch)

    method._promote_hotspot_mxfp4_expert(
        layer=SimpleNamespace(), global_expert_id=10, gpu_slot=0
    )
    method._promote_hotspot_mxfp4_expert(
        layer=SimpleNamespace(), global_expert_id=11, gpu_slot=1
    )

    assert events == [
        "submit:0",
        "export_sync",
        "stream_sync",
        "submit:1",
        "export_sync",
        "stream_sync",
    ]
    w13, w13_scale, w2, w2_scale = storages
    assert torch.equal(
        w13[0].transpose(-2, -1), torch.full((2, 3), 1, dtype=torch.uint8)
    )
    assert torch.equal(
        w13[1].transpose(-2, -1), torch.full((2, 3), 2, dtype=torch.uint8)
    )
    assert torch.equal(
        w2[0].transpose(-2, -1), torch.full((2, 4), 11, dtype=torch.uint8)
    )
    assert torch.equal(
        w2[1].transpose(-2, -1), torch.full((2, 4), 12, dtype=torch.uint8)
    )
    assert w13_scale.view(torch.uint8).unique().tolist() == [127, 128]
    assert w2_scale.view(torch.uint8).unique().tolist() == [128, 129]


def test_hotspot_promotion_rejects_destination_byte_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, _storages, _events = _make_cpu_hotspot_promotion(monkeypatch)
    original_equal = torch.equal
    equal_calls = 0

    def inject_mismatch(left: torch.Tensor, right: torch.Tensor) -> bool:
        nonlocal equal_calls
        equal_calls += 1
        return False if equal_calls == 1 else original_equal(left, right)

    monkeypatch.setattr(torch, "equal", inject_mismatch)

    with pytest.raises(RuntimeError, match="byte verification failed"):
        method._promote_hotspot_mxfp4_expert(
            layer=SimpleNamespace(), global_expert_id=10, gpu_slot=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hotspot_promotion_is_byte_exact_and_visible_to_existing_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    w13 = torch.zeros((1, 3, 2), dtype=torch.uint8, device=device)
    w13_scale = torch.zeros((1, 2, 1), dtype=torch.uint8, device=device)
    w2 = torch.zeros((1, 4, 2), dtype=torch.uint8, device=device)
    w2_scale = torch.zeros((1, 2, 2), dtype=torch.uint8, device=device)

    def pinned(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    staging = _Mxfp4HotspotStaging(
        host_w13=pinned((2, 3), torch.uint8),
        host_w13_scale=pinned((1, 2), torch.bfloat16),
        host_w2=pinned((2, 4), torch.uint8),
        host_w2_scale=pinned((2, 2), torch.bfloat16),
        device_w13=torch.empty((2, 3), dtype=torch.uint8, device=device),
        device_w13_scale_bf16=torch.empty(
            (1, 2), dtype=torch.bfloat16, device=device
        ),
        device_w13_scale_e8m0=torch.empty(
            (1, 2), dtype=torch.float8_e8m0fnu, device=device
        ),
        device_w2=torch.empty((2, 4), dtype=torch.uint8, device=device),
        device_w2_scale_bf16=torch.empty(
            (2, 2), dtype=torch.bfloat16, device=device
        ),
        device_w2_scale_e8m0=torch.empty(
            (2, 2), dtype=torch.float8_e8m0fnu, device=device
        ),
    )

    class FakeWrapper:
        def submit_write_weight_scale_to_buffer(
            self, gpu_tp_count: int, expert_id: int, *pointers: list[int]
        ) -> None:
            assert gpu_tp_count == 1
            assert pointers == (
                [staging.host_w13.data_ptr()],
                [staging.host_w13_scale.data_ptr()],
                [staging.host_w2.data_ptr()],
                [staging.host_w2_scale.data_ptr()],
            )
            staging.host_w13.fill_(expert_id + 1)
            staging.host_w13_scale.fill_(2 ** expert_id)
            staging.host_w2.fill_(expert_id + 11)
            staging.host_w2_scale.fill_(2 ** (expert_id + 1))

        def sync_write_weight_scale_to_buffer(self) -> None:
            return None

    method = object.__new__(KTEPWrapperMethod)
    method.cpu_expert_ids = torch.tensor([10, 11], dtype=torch.int64)
    method.wrapper = FakeWrapper()
    method.kt_config = SimpleNamespace(layer_idx=7)
    method._hotspot_mxfp4_storages = lambda _layer: (
        w13,
        w13_scale,
        w2,
        w2_scale,
    )
    monkeypatch.setattr(
        kt_ep_wrapper,
        "_get_mxfp4_hotspot_staging",
        lambda **_kwargs: staging,
    )

    captured_pointers = tuple(
        tensor.data_ptr() for tensor in (w13, w13_scale, w2, w2_scale)
    )
    method._promote_hotspot_mxfp4_expert(
        layer=SimpleNamespace(), global_expert_id=10, gpu_slot=0
    )

    graph_output = torch.empty_like(w13[0])
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        graph_output.copy_(w13[0])
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output.copy_(w13[0])
    graph.replay()
    torch.cuda.synchronize(device)
    assert graph_output.unique().item() == 1

    method._promote_hotspot_mxfp4_expert(
        layer=SimpleNamespace(), global_expert_id=11, gpu_slot=0
    )
    graph.replay()
    torch.cuda.synchronize(device)

    assert graph_output.unique().item() == 2
    assert tuple(
        tensor.data_ptr() for tensor in (w13, w13_scale, w2, w2_scale)
    ) == captured_pointers
    assert w13_scale.view(torch.uint8).unique().tolist() == [128]
    assert w2_scale.view(torch.uint8).unique().tolist() == [129]
