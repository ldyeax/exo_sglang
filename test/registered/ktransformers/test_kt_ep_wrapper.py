import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    build_logical_to_gpu_index,
    combine_remote_expert_tiers,
    create_kt_config_from_server_args,
    load_gpu_expert_mask_plan,
    load_profile_guided_gpu_expert_masks,
    mask_and_remap_expert_ids,
    mask_cpu_expert_ids,
    partition_remote_local_gpu_experts,
    resolve_gpu_experts_mask,
    select_remote_token_rows,
)


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
        torch.tensor(
            [False, True, False, False, False, False, False, False]
        ),
        torch.tensor(
            [False, False, False, False, False, False, True, False]
        ),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    topk_ids = torch.tensor(
        [[0, 2], [1, 3], [6, 4], [5, 7]], dtype=torch.int64
    )
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
        torch.tensor(
            [False, True, False, False, False, False, False, False]
        ),
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
        torch.tensor(
            [True, False, False, False, False, False, False, False]
        ),
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
        torch.tensor(
            [True, True, False, False, False, False, False, False]
        ),
    )
    torch.testing.assert_close(
        config.cpu_expert_ids,
        torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int64),
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


def test_explicit_gpu_mask_plan_drives_target_placement(
    monkeypatch, tmp_path
) -> None:
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
