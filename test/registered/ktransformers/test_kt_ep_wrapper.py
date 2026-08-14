import hashlib
import json
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.kt_ep_wrapper import (
    _KT_HYBRID_TIMING_FILE_DESCRIPTORS,
    _KT_ROUTE_STATS_ERROR_LAST_LOGGED,
    KTEPWrapperMethod,
    _collect_kt_cpu_route_stats,
    _emit_kt_hybrid_timing_receipt,
    _merge_hybrid_expert_outputs,
    _should_sample_kt_hybrid_timing,
    build_logical_to_gpu_index,
    combine_remote_expert_tiers,
    create_kt_config_from_server_args,
    load_gpu_expert_mask_plan,
    load_hybrid_expert_shard_plan,
    load_profile_guided_gpu_expert_masks,
    mask_and_remap_expert_ids,
    mask_cpu_expert_ids,
    partition_remote_local_gpu_experts,
    resolve_gpu_experts_mask,
    select_remote_token_rows,
    validate_kt_fused_shared_experts,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.mxfp4_deepseek import (
    map_mxfp4_expert_ids_for_ep,
)


def test_hybrid_timing_sampling_keeps_first_and_interval(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_KT_HYBRID_TIMING_SAMPLE_EVERY", "3")

    assert _should_sample_kt_hybrid_timing(1)
    assert not _should_sample_kt_hybrid_timing(2)
    assert _should_sample_kt_hybrid_timing(3)
    assert _should_sample_kt_hybrid_timing(6)


def test_kt_fused_shared_experts_fail_before_weight_allocation() -> None:
    config = SimpleNamespace(
        global_num_experts=256,
        gpu_experts_mask=torch.zeros(256, dtype=torch.bool),
    )

    with pytest.raises(
        ValueError, match=r"does not support fused shared experts"
    ) as error:
        validate_kt_fused_shared_experts(
            kt_config=config,
            model_num_experts=257,
            num_fused_shared_experts=1,
        )

    message = str(error.value)
    assert "257 experts" in message
    assert "256 routed experts" in message
    assert "--disable-shared-experts-fusion" in message


def test_kt_nonfused_routed_layout_remains_supported() -> None:
    config = SimpleNamespace(
        global_num_experts=256,
        gpu_experts_mask=torch.zeros(256, dtype=torch.bool),
    )

    validate_kt_fused_shared_experts(
        kt_config=config,
        model_num_experts=256,
        num_fused_shared_experts=0,
    )


def test_hybrid_timing_receipt_is_jsonl(monkeypatch, tmp_path) -> None:
    receipt_path = tmp_path / "kt-timing.jsonl"
    monkeypatch.setenv("SGLANG_KT_HYBRID_TIMING_RECEIPT", str(receipt_path))
    receipt = {
        "format": "sglang_kt_hybrid_timing_v1",
        "tp_rank": 1,
        "ep_rank": 1,
        "layer": 42,
        "step": 7,
    }

    _emit_kt_hybrid_timing_receipt(receipt)

    file_descriptor = _KT_HYBRID_TIMING_FILE_DESCRIPTORS.pop(
        str(receipt_path.resolve())
    )
    os.close(file_descriptor)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_hybrid_timing_route_stats_failure_is_nonfatal(caplog) -> None:
    class BrokenRouteStats:
        def get_last_forward_route_stats(self) -> dict[str, object]:
            raise RuntimeError("synthetic route evidence failure")

    with caplog.at_level("ERROR"):
        stats, error_type = _collect_kt_cpu_route_stats(
            BrokenRouteStats(),
            layer_index=17,
        )

    assert stats is None
    assert error_type == "RuntimeError"
    assert "failed for layer 17" in caplog.text
    assert "model serving will continue" in caplog.text


def test_hybrid_timing_route_stats_missing_method_is_ignored() -> None:
    assert _collect_kt_cpu_route_stats(object(), layer_index=2) == (None, None)


def test_hybrid_timing_route_stats_none_is_accepted() -> None:
    class PendingRouteStats:
        def get_last_forward_route_stats(self) -> None:
            return None

    assert _collect_kt_cpu_route_stats(PendingRouteStats(), layer_index=3) == (
        None,
        None,
    )


def test_hybrid_timing_route_stats_non_dictionary_is_nonfatal(caplog) -> None:
    class InvalidRouteStats:
        def get_last_forward_route_stats(self) -> list[object]:
            return []

    with caplog.at_level("ERROR"):
        stats, error_type = _collect_kt_cpu_route_stats(
            InvalidRouteStats(), layer_index=4
        )

    assert stats is None
    assert error_type == "TypeError"
    assert "failed for layer 4" in caplog.text


def test_hybrid_timing_route_stats_exception_logging_is_rate_limited(
    caplog,
) -> None:
    class BrokenRouteStats:
        def get_last_forward_route_stats(self) -> dict[str, object]:
            raise RuntimeError("repeatable synthetic failure")

    wrapper = BrokenRouteStats()
    _KT_ROUTE_STATS_ERROR_LAST_LOGGED.clear()
    with caplog.at_level("ERROR"):
        first = _collect_kt_cpu_route_stats(wrapper, layer_index=5)
        second = _collect_kt_cpu_route_stats(wrapper, layer_index=5)

    assert first == second == (None, "RuntimeError")
    assert caplog.text.count("failed for layer 5") == 1


def test_mask_cpu_expert_ids_preserves_submitted_routing_tensor() -> None:
    submitted_ids = torch.tensor([[0, 3, 7], [2, 1, 6]], dtype=torch.int64)
    original_ids = submitted_ids.clone()

    gpu_ids = mask_cpu_expert_ids(submitted_ids, num_gpu_experts=3)

    torch.testing.assert_close(submitted_ids, original_ids)
    torch.testing.assert_close(
        gpu_ids,
        torch.tensor([[0, -1, -1], [2, 1, -1]], dtype=torch.int64),
    )


def test_noncontiguous_gpu_experts_are_compactly_remapped() -> None:
    mask = torch.tensor([False, True, False, False, True, False], dtype=torch.bool)
    logical_to_gpu = build_logical_to_gpu_index(mask)
    submitted_ids = torch.tensor([[4, 1, 5], [0, 4, 2]], dtype=torch.int64)

    gpu_ids = mask_and_remap_expert_ids(submitted_ids, mask, logical_to_gpu)

    torch.testing.assert_close(submitted_ids, torch.tensor([[4, 1, 5], [0, 4, 2]]))
    torch.testing.assert_close(
        logical_to_gpu, torch.tensor([-1, 0, -1, -1, 1, -1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        gpu_ids, torch.tensor([[1, 0, -1], [-1, 1, -1]], dtype=torch.int32)
    )


def test_weight_loader_keeps_global_ids_for_arbitrary_kt_ep_shards(
    monkeypatch,
) -> None:
    method = object.__new__(KTEPWrapperMethod)
    method.num_gpu_experts = 1
    method.gpu_experts_mask = torch.zeros(256, dtype=torch.bool)
    method.gpu_experts_mask[130] = True
    method.logical_to_gpu_index = torch.full((256,), -1, dtype=torch.int32)
    method.logical_to_gpu_index[130] = 0
    method.rank_local_logical_expert_ids = True

    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.quant_method = method
    layer.num_local_experts = 2
    layer._expert_storage_rank = 1
    layer._num_local_routed = 128
    layer._num_global_routed = 256
    layer._has_fused_shared = False
    layer.quant_config = None
    loaded = []

    def capture_weight(**kwargs) -> None:
        loaded.append(kwargs["expert_id"])

    monkeypatch.setattr(layer, "_weight_loader_impl", capture_weight)
    parameter = torch.nn.Parameter(torch.empty(1))

    layer.weight_loader(
        param=parameter,
        loaded_weight=torch.empty(1),
        weight_name="expert.weight",
        shard_id="w1",
        expert_id=130,
    )
    layer.weight_loader(
        param=parameter,
        loaded_weight=torch.empty(1),
        weight_name="expert.weight",
        shard_id="w1",
        expert_id=129,
    )

    assert loaded == [0]


def test_mxfp4_keeps_kt_compact_ids_on_nonzero_ep_rank() -> None:
    topk_ids = torch.tensor([[0, 1, -1]], dtype=torch.int32)

    compact = map_mxfp4_expert_ids_for_ep(
        topk_ids,
        moe_ep_rank=1,
        num_local_experts=128,
        already_compact=True,
    )
    ordinary = map_mxfp4_expert_ids_for_ep(
        topk_ids,
        moe_ep_rank=1,
        num_local_experts=128,
        already_compact=False,
    )

    torch.testing.assert_close(compact, topk_ids)
    torch.testing.assert_close(
        ordinary, torch.tensor([[128, 129, -1]], dtype=torch.int32)
    )


def test_cpu_expert_owner_uses_moe_tp_rank(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.KTRANSFORMERS_AVAILABLE", True
    )
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_tp_rank=0),
    )
    gpu_method = SimpleNamespace(num_gpu_experts=None)
    config = SimpleNamespace(
        method="BF16",
        gpu_experts_mask=torch.zeros(4, dtype=torch.bool),
        expert_lora_path=None,
        cpu_expert_ids=torch.tensor([0, 2]),
        remote_expert_id_tiers=None,
        remote_expert_endpoints=None,
        global_num_experts=4,
        gpu_prefill_token_threshold=None,
        kt_enable_dynamic_expert_update=False,
        chunked_prefill_size=8,
        layer_idx=0,
    )

    method = KTEPWrapperMethod(gpu_method, config)

    assert method.tp_rank == 0


def test_hybrid_inplace_merge_matches_allocating_merge() -> None:
    gpu_partial = torch.tensor([[1.0, -2.0], [3.5, 4.0]])
    cpu_partial = torch.tensor([[0.5, 2.0], [-1.5, 8.0]])
    expected = gpu_partial + cpu_partial

    allocating_input = gpu_partial.clone()
    allocating = _merge_hybrid_expert_outputs(
        allocating_input,
        cpu_partial,
        inplace=False,
    )
    inplace_input = gpu_partial.clone()
    inplace = _merge_hybrid_expert_outputs(
        inplace_input,
        cpu_partial,
        inplace=True,
    )

    torch.testing.assert_close(allocating, expected)
    torch.testing.assert_close(inplace, expected)
    assert allocating.data_ptr() != allocating_input.data_ptr()
    assert inplace.data_ptr() == inplace_input.data_ptr()


def test_caller_owned_output_is_limited_to_compact_local_hybrid(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SGLANG_DSV4_KT_INPLACE_MOE_OUTPUT", "1")
    method = object.__new__(KTEPWrapperMethod)
    method.num_gpu_experts = 2
    method.cpu_expert_ids = torch.tensor([2, 3], dtype=torch.int64)
    method.rank_local_logical_expert_ids = True
    method.tp_rank = 0
    method._cpu_stream = object()
    method.remote_clients = ()
    method.remote_expert_id_tiers = None
    method.gpu_method = SimpleNamespace(
        _kt_compact_ids=True,
        _supports_caller_owned_output=True,
        apply_with_output=lambda *args, **kwargs: None,
    )
    layer = SimpleNamespace(
        _v4_tk_path=True,
        moe_runner_config=SimpleNamespace(inplace=True),
        dispatcher=SimpleNamespace(_pre_combine_hooks=None),
    )

    assert method._use_caller_owned_moe_output(layer, remote_pending=None)

    method.remote_clients = (object(),)
    with pytest.raises(RuntimeError, match="no remote expert sidecar"):
        method._use_caller_owned_moe_output(layer, remote_pending=None)

    method.remote_clients = ()
    layer.dispatcher._pre_combine_hooks = object()
    with pytest.raises(RuntimeError, match="no deferred pre-combine input consumer"):
        method._use_caller_owned_moe_output(layer, remote_pending=None)


def test_caller_owned_output_accepts_production_portable_layout(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DSV4_KT_INPLACE_MOE_OUTPUT", "1")
    method = object.__new__(KTEPWrapperMethod)
    method.num_gpu_experts = 2
    method.cpu_expert_ids = torch.tensor([2, 3], dtype=torch.int64)
    method.rank_local_logical_expert_ids = True
    method.tp_rank = 0
    method._cpu_stream = object()
    method.remote_clients = ()
    method.remote_expert_id_tiers = None
    method.gpu_method = SimpleNamespace(
        _kt_compact_ids=True,
        _supports_caller_owned_output=True,
        apply_with_output=lambda *args, **kwargs: None,
    )
    layer = SimpleNamespace(
        _dsv4_mxfp4_backend="triton_kernels",
        moe_runner_config=SimpleNamespace(inplace=True),
        dispatcher=SimpleNamespace(_pre_combine_hooks=None),
    )

    assert method._use_caller_owned_moe_output(layer, remote_pending=None)


def test_caller_owned_output_rejects_alt_stream_shared_expert_consumer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SGLANG_DSV4_KT_INPLACE_MOE_OUTPUT", "1")
    method = object.__new__(KTEPWrapperMethod)
    method.num_gpu_experts = 2
    method.cpu_expert_ids = torch.tensor([2, 3], dtype=torch.int64)
    method.rank_local_logical_expert_ids = True
    method.tp_rank = 0
    method._cpu_stream = object()
    method.remote_clients = ()
    method.remote_expert_id_tiers = None
    method.gpu_method = SimpleNamespace(
        _kt_compact_ids=True,
        _supports_caller_owned_output=True,
        apply_with_output=lambda *args, **kwargs: None,
    )
    layer = SimpleNamespace(
        _v4_tk_path=True,
        moe_runner_config=SimpleNamespace(inplace=True),
        dispatcher=SimpleNamespace(_pre_combine_hooks=None),
        _has_alt_stream_shared_expert_input_consumer=True,
    )

    with pytest.raises(
        RuntimeError,
        match="no alternate-stream shared-expert input consumer",
    ):
        method._use_caller_owned_moe_output(layer, remote_pending=None)


def test_caller_owned_output_defaults_off() -> None:
    method = object.__new__(KTEPWrapperMethod)
    method.num_gpu_experts = 1

    assert not method._use_caller_owned_moe_output(
        SimpleNamespace(), remote_pending=None
    )


def test_caller_owned_output_skips_unsupported_gpu_method(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DSV4_KT_INPLACE_MOE_OUTPUT", "1")
    method = object.__new__(KTEPWrapperMethod)
    method.num_gpu_experts = 2
    method.gpu_method = SimpleNamespace()

    assert not method._use_caller_owned_moe_output(
        SimpleNamespace(_v4_tk_path=False), remote_pending=None
    )


def test_kt_runner_preserves_global_routes_for_arbitrary_ep_shards() -> None:
    method = object.__new__(KTEPWrapperMethod)
    method.global_num_experts = 4
    method.override_num_local_experts = True
    method.num_gpu_experts = 0
    captured = {}

    class FakeGpuMethod:
        def create_moe_runner(self, layer, config) -> None:
            captured["config"] = config

    method.gpu_method = FakeGpuMethod()
    config = MoeRunnerConfig(
        num_experts=4,
        num_local_experts=2,
        routed_scaling_factor=2.5,
    )

    method.create_moe_runner(SimpleNamespace(), config)

    torch.testing.assert_close(
        config.kt_global_to_local_expert_mapping,
        torch.arange(4, dtype=torch.int32),
    )
    assert captured["config"].num_local_experts == 0
    assert captured["config"].routed_scaling_factor is None


def test_create_weights_keeps_global_expert_space_under_ep() -> None:
    method = object.__new__(KTEPWrapperMethod)
    method.global_num_experts = 4
    method.gpu_experts_mask = torch.zeros(4, dtype=torch.bool)
    method.cpu_expert_ids = torch.tensor([1, 3], dtype=torch.int64)
    method.num_gpu_experts = 0
    method.logical_to_gpu_index = torch.full((4,), -1, dtype=torch.int32)
    method.remote_expert_id_tiers = None
    method.remote_expert_endpoints = None
    method.remote_clients = ()
    method.tp_rank = 1
    method.kt_expert_lora_enabled = False
    method.kt_config = SimpleNamespace(
        max_deferred_experts_per_token=0,
        num_layers=2,
        layer_idx=0,
    )

    class FakeGpuMethod:
        def create_weights(self, **kwargs) -> None:
            assert kwargs["num_experts"] == 0

    method.gpu_method = FakeGpuMethod()
    layer = torch.nn.Linear(1, 1)
    layer.top_k = 2
    layer.intermediate_size_per_partition = 8
    layer.moe_tp_size = 1

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=8,
        params_dtype=torch.bfloat16,
    )

    assert method.global_num_experts == 4
    assert method.global_to_local_expert_mapping_cuda.shape == (4,)
    torch.testing.assert_close(
        method.global_to_local_expert_mapping_cuda,
        torch.tensor([-1, 0, -1, 1], dtype=torch.int32),
    )


def test_remote_local_gpu_partition_is_disjoint_and_global() -> None:
    requested_gpu_mask = torch.tensor(
        [False, True, False, True, False, True, False, False],
        dtype=torch.bool,
    )
    remote_ids = torch.tensor([1, 6], dtype=torch.int64)

    gpu_mask, local_cpu_ids = partition_remote_local_gpu_experts(
        requested_gpu_mask,
        remote_ids,
    )

    torch.testing.assert_close(
        gpu_mask,
        torch.tensor(
            [False, False, False, True, False, True, False, False],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        local_cpu_ids,
        torch.tensor([0, 2, 4, 7], dtype=torch.int64),
    )
    assert not gpu_mask[remote_ids].any()


def test_multiple_remote_tiers_are_disjoint() -> None:
    combined = combine_remote_expert_tiers(
        (
            torch.tensor([1, 6], dtype=torch.int64),
            torch.tensor([0, 5, 7], dtype=torch.int64),
        ),
        num_experts=8,
    )
    torch.testing.assert_close(
        combined,
        torch.tensor([1, 6, 0, 5, 7], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="tiers must be disjoint"):
        combine_remote_expert_tiers(
            (
                torch.tensor([1, 6], dtype=torch.int64),
                torch.tensor([0, 6], dtype=torch.int64),
            ),
            num_experts=8,
        )


def test_remote_tier_sends_only_selected_token_rows() -> None:
    class FakeRemoteClient:
        def __init__(
            self,
            multiplier: float,
            rendezvous: threading.Barrier,
        ):
            self.multiplier = multiplier
            self.rendezvous = rendezvous
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            self.rendezvous.wait(timeout=5)
            return kwargs["hidden_states"] * self.multiplier

    method = object.__new__(KTEPWrapperMethod)
    method.global_num_experts = 8
    method.kt_config = SimpleNamespace(layer_idx=17)
    rendezvous = threading.Barrier(2)
    first_client = FakeRemoteClient(multiplier=10, rendezvous=rendezvous)
    second_client = FakeRemoteClient(multiplier=100, rendezvous=rendezvous)
    method.remote_clients = (first_client, second_client)
    method.remote_expert_masks_cuda = (
        torch.tensor([False, True, False, False, False, False, False, False]),
        torch.tensor([False, False, False, False, False, False, True, False]),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor([[0, 2], [1, 3], [6, 4], [5, 7]], dtype=torch.int64)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

    torch.testing.assert_close(
        select_remote_token_rows(
            topk_ids,
            method.remote_expert_masks_cuda[0],
        ),
        torch.tensor([False, True, False, False]),
    )
    output = method._run_remote_if_selected(
        x=hidden_states,
        global_topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert output is not None
    torch.testing.assert_close(
        output,
        torch.stack(
            (
                torch.zeros(3),
                hidden_states[1] * 10,
                hidden_states[2] * 100,
                torch.zeros(3),
            )
        ),
    )
    assert len(first_client.calls) == 1
    assert len(second_client.calls) == 1
    torch.testing.assert_close(
        first_client.calls[0]["topk_ids"],
        topk_ids[1:2],
    )
    torch.testing.assert_close(
        second_client.calls[0]["topk_ids"],
        topk_ids[2:3],
    )


def test_remote_submission_defers_join_until_after_local_work() -> None:
    class BlockingRemoteClient:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def forward(self, **kwargs):
            self.started.set()
            assert self.release.wait(timeout=5)
            return kwargs["hidden_states"] * 7

    method = object.__new__(KTEPWrapperMethod)
    method.global_num_experts = 8
    method.kt_config = SimpleNamespace(layer_idx=9)
    client = BlockingRemoteClient()
    method.remote_clients = (client,)
    method.remote_expert_masks_cuda = (
        torch.tensor([False, True, False, False, False, False, False, False]),
    )
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    topk_ids = torch.tensor([[1, 3], [2, 4]], dtype=torch.int64)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

    pending = method._submit_remote_if_selected(
        x=hidden_states,
        global_topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert pending is not None
    assert client.started.wait(timeout=5)
    assert not pending.futures[0].done()
    client.release.set()
    output = method._finish_remote(pending, x=hidden_states)
    torch.testing.assert_close(
        output,
        torch.stack((hidden_states[0] * 7, torch.zeros(3))),
    )


def test_plural_remote_tiers_build_compact_local_complement(
    monkeypatch, tmp_path
) -> None:
    first_plan = tmp_path / "first.pt"
    second_plan = tmp_path / "second.pt"
    torch.save(
        {"remote_expert_ids": torch.tensor([[1, 6]], dtype=torch.int64)},
        first_plan,
    )
    torch.save(
        {"remote_expert_ids": torch.tensor([[2]], dtype=torch.int64)},
        second_plan,
    )
    monkeypatch.setenv(
        "SGLANG_KT_REMOTE_EXPERT_PLANS",
        f"{first_plan};{second_plan}",
    )
    monkeypatch.setenv(
        "SGLANG_KT_REMOTE_EXPERT_ENDPOINTS",
        "127.0.0.1:29562;10.44.0.2:29561",
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=8)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=1,
        kt_cpuinfer=4,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=None,
    )

    config = create_kt_config_from_server_args(
        server_args,
        layer_idx=0,
    )

    assert config is not None
    assert config.remote_expert_endpoints == (
        "127.0.0.1:29562",
        "10.44.0.2:29561",
    )
    assert config.remote_expert_id_tiers is not None
    torch.testing.assert_close(
        config.remote_expert_id_tiers[0],
        torch.tensor([1, 6], dtype=torch.int64),
    )
    torch.testing.assert_close(
        config.remote_expert_id_tiers[1],
        torch.tensor([2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([3, 4, 5, 7], dtype=torch.int64),
    )
    torch.testing.assert_close(
        config.gpu_experts_mask,
        torch.tensor([True, False, False, False, False, False, False, False]),
    )


def test_gpu_only_tier_compacts_native_cpu_storage() -> None:
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=8)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=2,
        kt_cpuinfer=4,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=None,
    )

    config = create_kt_config_from_server_args(
        server_args,
        layer_idx=0,
    )

    assert config is not None
    torch.testing.assert_close(
        config.gpu_experts_mask,
        torch.tensor([True, True, False, False, False, False, False, False]),
    )
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int64),
    )


def test_dsv4_split_tier_selects_amxint4_target_and_mxfp4_draft(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU", "1")
    monkeypatch.setenv("SGLANG_KT_DRAFT_METHOD", "MXFP4")
    monkeypatch.setenv("SGLANG_KT_DRAFT_WEIGHT_PATH", "/native-mxfp4")
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=8)
        ),
        kt_weight_path="/amxint4",
        kt_num_gpu_experts=2,
        kt_cpuinfer=56,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="AMXINT4",
        kt_max_deferred_experts_per_token=0,
    )

    target = create_kt_config_from_server_args(server_args, layer_idx=0)
    draft = create_kt_config_from_server_args(
        server_args,
        layer_idx=0,
        prefix="model.stages.0.mlp.experts",
    )

    assert target is not None
    assert target.method == "AMXINT4"
    assert target.weight_path == "/amxint4"
    assert draft is not None
    assert draft.method == "MXFP4"
    assert draft.weight_path == "/native-mxfp4"
    assert draft.weight_key_prefix == "mtp.0"


def test_dsv4_split_tier_rejects_unpinned_draft(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU", "1")
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=8)
        ),
        kt_weight_path="/amxint4",
        kt_num_gpu_experts=2,
        kt_cpuinfer=56,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="AMXINT4",
        kt_max_deferred_experts_per_token=0,
    )

    with pytest.raises(ValueError, match="SGLANG_KT_DRAFT_METHOD=MXFP4"):
        create_kt_config_from_server_args(
            server_args,
            layer_idx=0,
            prefix="model.stages.0.mlp.experts",
        )


def test_dsv4_split_tier_rejects_non_amxint4_target(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU", "1")
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=8)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=2,
        kt_cpuinfer=56,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    with pytest.raises(ValueError, match="target --kt-method AMXINT4"):
        create_kt_config_from_server_args(server_args, layer_idx=0)


def test_gpu_only_tier_stays_on_cpu_under_accelerator_default_device() -> None:
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=8)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=2,
        kt_cpuinfer=4,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=None,
    )

    with torch.device("meta"):
        config = create_kt_config_from_server_args(server_args, layer_idx=0)

    assert config is not None
    assert config.gpu_experts_mask.device.type == "cpu"
    assert config.cpu_expert_ids is not None
    assert config.cpu_expert_ids.device.type == "cpu"
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int64),
    )


def test_cpu_shard_selects_one_numa_pool_per_ep_rank(monkeypatch, tmp_path) -> None:
    shard_path = tmp_path / "cpu-shards.pt"
    torch.save(
        {
            "expert_ids_by_rank": (
                torch.tensor([[0, 2], [1, 3]], dtype=torch.int64),
                torch.tensor([[1, 3], [0, 2]], dtype=torch.int64),
            )
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_CPU_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=0,
        kt_cpuinfer=52,
        kt_threadpool_count=1,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    config = create_kt_config_from_server_args(server_args, layer_idx=0)

    assert config is not None
    assert config.threadpool_count == 1
    assert config.numa_nodes == [1]
    assert config.cpuinfer_threads == 52
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([1, 3], dtype=torch.int64),
    )


def test_cpu_shard_plan_stays_on_cpu_under_accelerator_default_device(
    monkeypatch, tmp_path
) -> None:
    shard_path = tmp_path / "cpu-shards.pt"
    torch.save(
        {
            "expert_ids_by_rank": (
                torch.tensor([[0, 2], [1, 3]], dtype=torch.int64),
                torch.tensor([[1, 3], [0, 2]], dtype=torch.int64),
            )
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_CPU_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=0),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=0,
        kt_cpuinfer=56,
        kt_threadpool_count=1,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    with torch.device("meta"):
        config = create_kt_config_from_server_args(server_args, layer_idx=0)

    assert config is not None
    assert config.cpu_expert_ids is not None
    assert config.cpu_expert_ids.device.type == "cpu"
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([0, 2], dtype=torch.int64),
    )


def test_hybrid_shard_selects_disjoint_gpu_cpu_tiers_and_local_numa(
    monkeypatch, tmp_path
) -> None:
    shard_path = tmp_path / "hybrid-shards.pt"
    gpu_masks = torch.tensor(
        [
            [[True, False, False, False], [False, True, False, False]],
            [[False, True, False, False], [True, False, False, False]],
        ],
        dtype=torch.bool,
    )
    cpu_shards = (
        torch.tensor([[2], [3]], dtype=torch.int64),
        torch.tensor([[3], [2]], dtype=torch.int64),
    )
    torch.save(
        {
            "gpu_experts_mask_by_rank": gpu_masks,
            "cpu_expert_ids_by_rank": cpu_shards,
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
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

    with torch.device("meta"):
        config = create_kt_config_from_server_args(server_args, layer_idx=0)

    assert config is not None
    assert config.numa_nodes == [1]
    assert config.threadpool_count == 1
    assert config.gpu_experts_mask.device.type == "cpu"
    assert config.rank_local_logical_expert_ids
    assert config.cpu_expert_ids is not None
    assert config.cpu_expert_ids.device.type == "cpu"
    torch.testing.assert_close(config.gpu_experts_mask, gpu_masks[1, 0])
    torch.testing.assert_close(config.cpu_expert_ids, cpu_shards[1][0])


def test_hybrid_shard_plan_hash_is_loader_enforced(monkeypatch, tmp_path) -> None:
    shard_path = tmp_path / "hash-bound-hybrid-shards.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [[[True, False]]], dtype=torch.bool
            ),
            "cpu_expert_ids_by_rank": (torch.tensor([[1]], dtype=torch.int64),),
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_PLAN_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="launcher-admitted digest"):
        load_hybrid_expert_shard_plan(
            str(shard_path),
            num_layers=1,
            num_experts=2,
            ep_size=1,
            ep_rank=0,
        )

    monkeypatch.setenv(
        "SGLANG_KT_HYBRID_EXPERT_PLAN_SHA256",
        hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    )
    gpu_mask, cpu_ids = load_hybrid_expert_shard_plan(
        str(shard_path),
        num_layers=1,
        num_experts=2,
        ep_size=1,
        ep_rank=0,
    )

    torch.testing.assert_close(gpu_mask, torch.tensor([[True, False]]))
    torch.testing.assert_close(cpu_ids, torch.tensor([[1]], dtype=torch.int64))


def test_variable_width_hybrid_plan_loads_padded_rank_shards(tmp_path) -> None:
    shard_path = tmp_path / "variable-hybrid-shards.pt"
    gpu_masks = torch.tensor(
        [
            [
                [True, True, False, False, False, False],
                [True, False, False, False, False, False],
            ],
            [
                [False, False, False, True, False, False],
                [False, False, False, True, True, False],
            ],
        ],
        dtype=torch.bool,
    )
    cpu_padded = torch.tensor(
        [
            [[2, -1], [1, 2]],
            [[4, 5], [5, -1]],
        ],
        dtype=torch.int64,
    )
    cpu_counts = torch.tensor([[1, 2], [2, 1]], dtype=torch.int64)
    torch.save(
        {
            "format": "sglang_kt_hybrid_expert_shard_v2_variable",
            "gpu_experts_mask_by_rank": gpu_masks,
            "cpu_expert_ids_padded_by_rank": cpu_padded,
            "cpu_rank_counts_by_layer": cpu_counts,
            "gpu_rank_counts_by_layer": gpu_masks.sum(dim=2),
        },
        shard_path,
    )

    rank0_masks, rank0_cpu = load_hybrid_expert_shard_plan(
        str(shard_path),
        num_layers=2,
        num_experts=6,
        ep_size=2,
        ep_rank=0,
    )
    rank1_masks, rank1_cpu = load_hybrid_expert_shard_plan(
        str(shard_path),
        num_layers=2,
        num_experts=6,
        ep_size=2,
        ep_rank=1,
    )

    torch.testing.assert_close(rank0_masks, gpu_masks[0])
    torch.testing.assert_close(rank1_masks, gpu_masks[1])
    torch.testing.assert_close(rank0_cpu, cpu_padded[0])
    torch.testing.assert_close(rank1_cpu, cpu_padded[1])


def test_variable_width_hybrid_config_filters_padding_and_uses_max_ceiling(
    monkeypatch, tmp_path
) -> None:
    shard_path = tmp_path / "variable-config.pt"
    gpu_masks = torch.tensor(
        [
            [[True, True, False, False]],
            [[False, False, True, False]],
        ],
        dtype=torch.bool,
    )
    torch.save(
        {
            "format": "sglang_kt_hybrid_expert_shard_v2_variable",
            "gpu_experts_mask_by_rank": gpu_masks,
            "cpu_expert_ids_padded_by_rank": torch.tensor(
                [[[-1]], [[3]]], dtype=torch.int64
            ),
            "cpu_rank_counts_by_layer": torch.tensor([[0], [1]], dtype=torch.int64),
            "gpu_rank_counts_by_layer": torch.tensor([[2], [1]], dtype=torch.int64),
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=1, n_routed_experts=4)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=2,
        kt_cpuinfer=56,
        kt_threadpool_count=2,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    with torch.device("meta"):
        config = create_kt_config_from_server_args(server_args, layer_idx=0)

    assert config is not None
    assert int(config.gpu_experts_mask.sum()) == 1
    torch.testing.assert_close(
        config.cpu_expert_ids, torch.tensor([3], dtype=torch.int64)
    )


def test_variable_width_hybrid_plan_rejects_nonnegative_padding(tmp_path) -> None:
    shard_path = tmp_path / "bad-variable-padding.pt"
    torch.save(
        {
            "format": "sglang_kt_hybrid_expert_shard_v2_variable",
            "gpu_experts_mask_by_rank": torch.tensor(
                [[[True, False, False]]], dtype=torch.bool
            ),
            "cpu_expert_ids_padded_by_rank": torch.tensor(
                [[[1, 2]]], dtype=torch.int64
            ),
            "cpu_rank_counts_by_layer": torch.tensor([[1]], dtype=torch.int64),
            "gpu_rank_counts_by_layer": torch.tensor([[1]], dtype=torch.int64),
        },
        shard_path,
    )

    with pytest.raises(ValueError, match="padding must be -1"):
        load_hybrid_expert_shard_plan(
            str(shard_path),
            num_layers=1,
            num_experts=3,
            ep_size=1,
            ep_rank=0,
        )


def test_hybrid_shard_selects_one_numa_pool_per_tp1_pipeline_rank(
    monkeypatch, tmp_path
) -> None:
    shard_path = tmp_path / "pp-hybrid-shards.pt"
    gpu_masks = torch.tensor(
        [
            [
                [True, True, False, False],
                [False, False, True, True],
            ]
        ],
        dtype=torch.bool,
    )
    cpu_shards = (torch.tensor([[2, 3], [0, 1]], dtype=torch.int64),)
    torch.save(
        {
            "gpu_experts_mask_by_rank": gpu_masks,
            "cpu_expert_ids_by_rank": cpu_shards,
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(
            moe_ep_size=1,
            moe_ep_rank=0,
            tp_size=1,
            pp_size=2,
            pp_rank=1,
        ),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=2,
        kt_cpuinfer=56,
        kt_threadpool_count=2,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    config = create_kt_config_from_server_args(server_args, layer_idx=1)

    assert config is not None
    assert config.numa_nodes == [1]
    assert config.threadpool_count == 1
    torch.testing.assert_close(config.gpu_experts_mask, gpu_masks[0, 1])
    torch.testing.assert_close(config.cpu_expert_ids, cpu_shards[0][1])


def test_hybrid_shard_rejects_missing_tp1_pipeline_numa_owner(
    monkeypatch, tmp_path
) -> None:
    shard_path = tmp_path / "pp-hybrid-shards.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [[[True, False], [False, True]]], dtype=torch.bool
            ),
            "cpu_expert_ids_by_rank": (torch.tensor([[1], [0]], dtype=torch.int64),),
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(
            moe_ep_size=1,
            moe_ep_rank=0,
            tp_size=1,
            pp_size=2,
            pp_rank=1,
        ),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=2)
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=1,
        kt_cpuinfer=56,
        kt_threadpool_count=1,
        kt_numa_nodes=[0],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    with pytest.raises(
        ValueError,
        match="requires one NUMA node per PP rank",
    ):
        create_kt_config_from_server_args(server_args, layer_idx=1)


def test_hybrid_shard_demotes_target_gpu_experts_for_dspark_draft(
    monkeypatch, tmp_path
) -> None:
    shard_path = tmp_path / "hybrid-draft-shards.pt"
    gpu_masks = torch.tensor(
        [
            [[True, False, False, False]],
            [[False, True, False, False]],
        ],
        dtype=torch.bool,
    )
    cpu_shards = (
        torch.tensor([[2]], dtype=torch.int64),
        torch.tensor([[3]], dtype=torch.int64),
    )
    torch.save(
        {
            "gpu_experts_mask_by_rank": gpu_masks,
            "cpu_expert_ids_by_rank": cpu_shards,
        },
        shard_path,
    )
    monkeypatch.setenv("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN", str(shard_path))
    monkeypatch.setenv("SGLANG_KT_DRAFT_GPU_EXPERTS", "0")
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
        kt_threadpool_count=1,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    config = create_kt_config_from_server_args(
        server_args, layer_idx=0, prefix="model.stages.0.mlp.experts"
    )

    assert config is not None
    assert not config.gpu_experts_mask.any()
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([1, 3], dtype=torch.int64),
    )
    assert config.weight_key_prefix == "mtp.0"
    assert config.rank_local_logical_expert_ids


def test_dedicated_draft_hybrid_shard_retains_draft_gpu_experts(
    monkeypatch, tmp_path
) -> None:
    draft_shard_path = tmp_path / "dedicated-draft-hybrid-shards.pt"
    draft_gpu_masks = torch.tensor(
        [
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
            ],
            [
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ],
        ],
        dtype=torch.bool,
    )
    draft_cpu_shards = (
        torch.tensor([[2], [3], [0]], dtype=torch.int64),
        torch.tensor([[3], [0], [1]], dtype=torch.int64),
    )
    torch.save(
        {
            "gpu_experts_mask_by_rank": draft_gpu_masks,
            "cpu_expert_ids_by_rank": draft_cpu_shards,
        },
        draft_shard_path,
    )
    monkeypatch.setenv(
        "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN", str(draft_shard_path)
    )
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=43,
                n_routed_experts=4,
                dspark_target_layer_ids=[40, 41, 42],
            )
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

    config = create_kt_config_from_server_args(
        server_args, layer_idx=1, prefix="model.stages.1.mlp.experts"
    )

    assert config is not None
    assert config.num_layers == 3
    assert config.numa_nodes == [1]
    assert config.threadpool_count == 1
    assert config.weight_key_prefix == "mtp.1"
    assert config.rank_local_logical_expert_ids
    torch.testing.assert_close(config.gpu_experts_mask, draft_gpu_masks[1, 1])
    torch.testing.assert_close(config.cpu_expert_ids, draft_cpu_shards[1][1])


def test_dedicated_draft_hybrid_shard_rejects_legacy_draft_count(
    monkeypatch, tmp_path
) -> None:
    draft_shard_path = tmp_path / "dedicated-draft-hybrid-shards.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": torch.tensor(
                [
                    [[True, False], [False, True], [True, False]],
                    [[False, True], [True, False], [False, True]],
                ],
                dtype=torch.bool,
            ),
            "cpu_expert_ids_by_rank": (
                torch.empty((3, 0), dtype=torch.int64),
                torch.empty((3, 0), dtype=torch.int64),
            ),
        },
        draft_shard_path,
    )
    monkeypatch.setenv(
        "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN", str(draft_shard_path)
    )
    monkeypatch.setenv("SGLANG_KT_DRAFT_GPU_EXPERTS", "1")
    monkeypatch.setattr(
        "sglang.srt.layers.moe.kt_ep_wrapper.get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=0),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=43,
                n_routed_experts=2,
                dspark_target_layer_ids=[40, 41, 42],
            )
        ),
        kt_weight_path="/model",
        kt_num_gpu_experts=1,
        kt_cpuinfer=56,
        kt_threadpool_count=1,
        kt_numa_nodes=[0, 1],
        chunked_prefill_size=32,
        kt_method="MXFP4",
        kt_max_deferred_experts_per_token=0,
    )

    with pytest.raises(
        ValueError,
        match="cannot be combined with SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
    ):
        create_kt_config_from_server_args(
            server_args, layer_idx=0, prefix="model.stages.0.mlp.experts"
        )


def test_hybrid_shard_rejects_overlapping_ownership(tmp_path) -> None:
    shard_path = tmp_path / "overlapping-hybrid-shards.pt"
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
                torch.tensor([[1]], dtype=torch.int64),
                torch.tensor([[3]], dtype=torch.int64),
            ),
        },
        shard_path,
    )

    with pytest.raises(ValueError, match="exact, disjoint cover"):
        load_hybrid_expert_shard_plan(
            str(shard_path),
            num_layers=1,
            num_experts=4,
            ep_size=2,
            ep_rank=0,
        )


def test_profile_selects_global_hottest_expert_slots(tmp_path) -> None:
    counts = torch.tensor(
        [[[1, 9, 2, 3], [8, 7, 6, 5]]],
        dtype=torch.int64,
    )
    profile_path = tmp_path / "profile.pt"
    torch.save({"logical_count": counts}, profile_path)

    masks = load_profile_guided_gpu_expert_masks(
        str(profile_path),
        num_layers=2,
        num_experts=4,
        num_gpu_experts_per_layer=1,
    )

    torch.testing.assert_close(
        masks,
        torch.tensor(
            [[False, True, False, False], [True, False, False, False]],
            dtype=torch.bool,
        ),
    )


def test_merged_two_dimensional_profile_selects_hottest_slots(tmp_path) -> None:
    counts = torch.tensor(
        [[1, 9, 2, 3], [8, 7, 6, 5]],
        dtype=torch.int64,
    )
    profile_path = tmp_path / "merged-profile.pt"
    torch.save({"logical_count": counts}, profile_path)

    masks = load_profile_guided_gpu_expert_masks(
        str(profile_path),
        num_layers=2,
        num_experts=4,
        num_gpu_experts_per_layer=1,
    )

    torch.testing.assert_close(
        masks,
        torch.tensor(
            [[False, True, False, False], [True, False, False, False]],
            dtype=torch.bool,
        ),
    )


def test_explicit_gpu_mask_plan_preserves_variable_layer_widths(tmp_path) -> None:
    expected = torch.tensor(
        [[False, True, False, True], [True, False, False, False]],
        dtype=torch.bool,
    )
    plan_path = tmp_path / "gpu-mask.pt"
    torch.save({"gpu_experts_mask": expected}, plan_path)

    masks = load_gpu_expert_mask_plan(
        str(plan_path),
        num_layers=2,
        num_experts=4,
    )

    torch.testing.assert_close(masks, expected)


def test_explicit_gpu_mask_plan_rejects_profile_ambiguity(
    monkeypatch, tmp_path
) -> None:
    plan_path = tmp_path / "gpu-mask.pt"
    torch.save(
        {"gpu_experts_mask": torch.zeros((2, 4), dtype=torch.bool)},
        plan_path,
    )
    monkeypatch.setenv("SGLANG_KT_GPU_EXPERT_MASK_PLAN", str(plan_path))
    monkeypatch.setenv("SGLANG_KT_EXPERT_PROFILE", str(plan_path))
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
        ),
        kt_num_gpu_experts=1,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_gpu_experts_mask(
            server_args,
            layer_idx=0,
            weight_key_prefix=None,
        )


def test_explicit_gpu_mask_plan_drives_target_placement(monkeypatch, tmp_path) -> None:
    expected = torch.tensor(
        [[False, True, False, True], [True, False, False, False]],
        dtype=torch.bool,
    )
    plan_path = tmp_path / "gpu-mask.pt"
    torch.save({"gpu_experts_mask": expected}, plan_path)
    monkeypatch.setenv("SGLANG_KT_GPU_EXPERT_MASK_PLAN", str(plan_path))
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
        ),
        kt_num_gpu_experts=1,
    )

    mask = resolve_gpu_experts_mask(
        server_args,
        layer_idx=0,
        weight_key_prefix=None,
    )

    torch.testing.assert_close(mask, expected[0])


def test_dspark_draft_stages_remain_all_cpu(monkeypatch, tmp_path) -> None:
    counts = torch.ones((1, 2, 4), dtype=torch.int64)
    profile_path = tmp_path / "profile.pt"
    torch.save({"logical_count": counts}, profile_path)
    monkeypatch.setenv("SGLANG_KT_EXPERT_PROFILE", str(profile_path))
    server_args = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
        ),
        kt_num_gpu_experts=1,
    )

    mask = resolve_gpu_experts_mask(
        server_args,
        layer_idx=0,
        weight_key_prefix="mtp.0",
    )

    torch.testing.assert_close(mask, torch.zeros(4, dtype=torch.bool))


def test_mxfp4_marlin_quant_type_is_admitted_on_sm86() -> None:
    from sgl_kernel.scalar_type import scalar_types

    from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported

    assert check_marlin_supported(
        scalar_types.float4_e2m1f,
        group_size=32,
        has_zp=False,
        device_capability=86,
    )
