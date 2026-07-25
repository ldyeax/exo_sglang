from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.moe.small_ep import (
    SmallEPContractError,
    SmallEPLayout,
    SmallEPLocalRoutes,
    TorchSmallEPCollective,
    compute_small_ep_reference_partial,
    execute_small_ep,
    localize_small_ep_routes,
    mark_small_ep_partial_produced,
    reduce_small_ep_reference_partials,
    select_small_ep_model_forward,
    small_ep_payload_bytes,
    small_ep_forward_scope,
)


def _run_two_rank_small_ep_collective(
    rank: int,
    init_file: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        local_hidden_states = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            if rank == 0
            else torch.tensor([[5.0, 6.0]])
        )
        layout = SmallEPLayout.contiguous(num_experts=4)

        def gate(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            first = values[:, 0].to(torch.int64).remainder(4)
            return (
                torch.stack((first, (first + 1).remainder(4)), dim=1),
                values.new_tensor([0.75, 0.25]).expand(values.shape[0], -1),
            )

        def local_experts(
            values: torch.Tensor,
            routes: SmallEPLocalRoutes,
        ) -> torch.Tensor:
            partial = torch.zeros_like(values)
            for choice in range(routes.local_topk_ids.shape[1]):
                local_ids = routes.local_topk_ids[:, choice]
                for local_id, global_id in enumerate(routes.owned_global_experts):
                    selected = torch.where(local_ids == local_id)[0]
                    if selected.numel() == 0:
                        continue
                    weighted = (
                        values[selected]
                        * float(global_id + 1)
                        * routes.local_topk_weights[selected, choice].unsqueeze(-1)
                    )
                    partial.index_add_(0, selected, weighted)
            return partial

        output = execute_small_ep(
            local_hidden_states,
            layout,
            gate,
            local_experts,
            TorchSmallEPCollective(),
        )

        all_hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        topk_ids, topk_weights = gate(all_hidden_states)
        expected = torch.zeros_like(all_hidden_states)
        for token in range(all_hidden_states.shape[0]):
            for choice in range(topk_ids.shape[1]):
                expected[token] += (
                    all_hidden_states[token]
                    * float(int(topk_ids[token, choice]) + 1)
                    * topk_weights[token, choice]
                )
        expected_stripe = expected[:2] if rank == 0 else expected[2:]
        assert torch.allclose(output, expected_stripe)
    finally:
        dist.destroy_process_group()


def test_torch_small_ep_collective_runs_two_real_gloo_ranks(tmp_path: Path) -> None:
    mp.spawn(
        _run_two_rank_small_ep_collective,
        args=(str(tmp_path / "small-ep-init"),),
        nprocs=2,
        join=True,
    )


def test_model_selection_and_forward_scope_fail_closed() -> None:
    assert select_small_ep_model_forward(
        enabled=True,
        architecture="GlmMoeDsaForCausalLM",
        kt_wrapper_present=True,
        context_parallel_extend=True,
        num_tokens=4096,
        token_threshold=4096,
    )
    with pytest.raises(
        SmallEPContractError,
        match="not an NSA context-parallel extend",
    ):
        select_small_ep_model_forward(
            enabled=True,
            architecture="GlmMoeDsaForCausalLM",
            kt_wrapper_present=True,
            context_parallel_extend=False,
            num_tokens=4096,
            token_threshold=4096,
        )

    with small_ep_forward_scope(enabled=True) as state:
        mark_small_ep_partial_produced()
        assert state.partial_produced
    with pytest.raises(SmallEPContractError, match="model-forward scope"):
        mark_small_ep_partial_produced()


def test_two_rank_local_routes_preserve_global_choice_positions() -> None:
    layout = SmallEPLayout.contiguous(num_experts=4)
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 3]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.60, 0.40], [0.55, 0.45]],
        dtype=torch.float32,
    )

    rank_zero = localize_small_ep_routes(topk_ids, topk_weights, layout, rank=0)
    assert rank_zero.owned_global_experts == (0, 1)
    assert torch.equal(
        rank_zero.local_topk_ids,
        torch.tensor([[0, -1], [1, -1], [-1, -1]]),
    )
    assert torch.equal(
        rank_zero.local_topk_weights,
        torch.tensor([[0.75, 0.00], [0.60, 0.00], [0.00, 0.00]]),
    )

    rank_one = localize_small_ep_routes(topk_ids, topk_weights, layout, rank=1)
    assert rank_one.owned_global_experts == (2, 3)
    assert torch.equal(
        rank_one.local_topk_ids,
        torch.tensor([[-1, 0], [-1, 1], [0, 1]]),
    )
    assert torch.equal(
        rank_one.local_topk_weights,
        torch.tensor([[0.00, 0.25], [0.00, 0.40], [0.55, 0.45]]),
    )


def test_local_partials_reduce_to_dense_reference_and_original_stripes() -> None:
    layout = SmallEPLayout.contiguous(num_experts=4)
    hidden_states = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 3]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.60, 0.40], [0.55, 0.45]],
        dtype=torch.float32,
    )

    def expert(expert_id: int, values: torch.Tensor) -> torch.Tensor:
        return values * float(expert_id + 1)

    partials = [
        compute_small_ep_reference_partial(
            hidden_states,
            topk_ids,
            topk_weights,
            layout,
            rank,
            expert,
        )
        for rank in range(2)
    ]
    dense_reference = torch.zeros_like(hidden_states)
    for token in range(hidden_states.shape[0]):
        for choice in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token, choice])
            dense_reference[token] += (
                expert(expert_id, hidden_states[token]) * topk_weights[token, choice]
            )

    rank_zero = reduce_small_ep_reference_partials(
        partials,
        source_lengths=(2, 1),
        rank=0,
    )
    rank_one = reduce_small_ep_reference_partials(
        partials,
        source_lengths=(2, 1),
        rank=1,
    )
    assert torch.allclose(rank_zero, dense_reference[:2])
    assert torch.allclose(rank_one, dense_reference[2:])
    assert torch.allclose(torch.cat((rank_zero, rank_one)), dense_reference)


def test_round_robin_layout_allows_explicit_load_balancing() -> None:
    layout = SmallEPLayout.round_robin(num_experts=8)
    assert layout.owned_experts(0) == (0, 2, 4, 6)
    assert layout.owned_experts(1) == (1, 3, 5, 7)


def test_small_ep_payload_is_hidden_size_not_topk_size() -> None:
    assert (
        small_ep_payload_bytes(
            source_tokens_per_rank=(4096, 4096),
            hidden_size=6144,
            element_size=2,
            rank=0,
        )
        == 144 * 1024**2
    )


def test_contract_rejects_invalid_routes_and_ownership() -> None:
    with pytest.raises(SmallEPContractError, match="world_size=2"):
        SmallEPLayout.contiguous(num_experts=6, world_size=3)

    layout = SmallEPLayout.contiguous(num_experts=4)
    with pytest.raises(SmallEPContractError, match="invalid expert"):
        localize_small_ep_routes(
            torch.tensor([[0, 4]]),
            torch.tensor([[0.5, 0.5]]),
            layout,
            rank=0,
        )
