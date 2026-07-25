# SPDX-License-Identifier: Apache-2.0
"""Communication-lean two-rank expert-parallel contract.

SmallEP is intentionally different from a standard token-dispatch all-to-all:

1. ranks all-gather *unsorted* hidden states;
2. every rank repeats gating/sorting deterministically;
3. each rank computes only experts it owns and reduces its top-k choices into
   one hidden-size partial per gathered token;
4. one hidden-size all-reduce combines those partials, after which each rank
   selects its original context stripe.

The redundant gate/sort work trades inexpensive local compute for substantially
less payload on a two-GPU PCIe/InfiniBand-class topology.  This module supplies
the exact indexing/reduction contract and a working torch.distributed
collective.  Wiring it into GLM's context-parallel attention and a local-expert
GPU runner remains a separate model-level step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist


class SmallEPContractError(ValueError):
    """Raised when ranks, routes, or ownership violate the SmallEP contract."""


@dataclass(frozen=True)
class SmallEPLayout:
    """Explicit global-expert ownership for a small EP group."""

    expert_to_rank: tuple[int, ...]
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size != 2:
            raise SmallEPContractError(
                f"the first SmallEP implementation requires world_size=2, got {self.world_size}"
            )
        if not self.expert_to_rank:
            raise SmallEPContractError("SmallEP requires at least one expert")
        invalid = [
            owner
            for owner in self.expert_to_rank
            if owner < 0 or owner >= self.world_size
        ]
        if invalid:
            raise SmallEPContractError(
                f"SmallEP expert owners are outside [0,{self.world_size}): {invalid}"
            )
        for rank in range(self.world_size):
            if rank not in self.expert_to_rank:
                raise SmallEPContractError(f"SmallEP rank {rank} owns no experts")

    @classmethod
    def contiguous(cls, num_experts: int, world_size: int = 2) -> "SmallEPLayout":
        if num_experts <= 0:
            raise SmallEPContractError("num_experts must be positive")
        if num_experts % world_size != 0:
            raise SmallEPContractError(
                f"{num_experts} experts cannot be divided evenly over {world_size} ranks"
            )
        experts_per_rank = num_experts // world_size
        return cls(
            expert_to_rank=tuple(
                expert_id // experts_per_rank for expert_id in range(num_experts)
            ),
            world_size=world_size,
        )

    @classmethod
    def round_robin(cls, num_experts: int, world_size: int = 2) -> "SmallEPLayout":
        if num_experts <= 0:
            raise SmallEPContractError("num_experts must be positive")
        return cls(
            expert_to_rank=tuple(
                expert_id % world_size for expert_id in range(num_experts)
            ),
            world_size=world_size,
        )

    def owned_experts(self, rank: int) -> tuple[int, ...]:
        self._validate_rank(rank)
        return tuple(
            expert_id
            for expert_id, owner in enumerate(self.expert_to_rank)
            if owner == rank
        )

    def _validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.world_size:
            raise SmallEPContractError(
                f"SmallEP rank must be in [0,{self.world_size}), got {rank}"
            )


@dataclass(frozen=True)
class SmallEPGatheredBatch:
    """Rank-major unsorted hidden states and their source stripe sizes."""

    hidden_states: torch.Tensor
    source_lengths: tuple[int, ...]

    def source_slice(self, rank: int) -> slice:
        if rank < 0 or rank >= len(self.source_lengths):
            raise SmallEPContractError(f"source rank {rank} is out of range")
        start = sum(self.source_lengths[:rank])
        return slice(start, start + self.source_lengths[rank])


@dataclass(frozen=True)
class SmallEPLocalRoutes:
    """Global top-k routes masked/remapped to one rank's local expert table."""

    local_topk_ids: torch.Tensor
    local_topk_weights: torch.Tensor
    owned_global_experts: tuple[int, ...]


def localize_small_ep_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: SmallEPLayout,
    rank: int,
) -> SmallEPLocalRoutes:
    """Mask non-owned choices and remap owned global IDs to local indices."""

    layout._validate_rank(rank)
    if topk_ids.shape != topk_weights.shape:
        raise SmallEPContractError(
            "SmallEP top-k ids and weights must have identical shapes"
        )
    if topk_ids.ndim != 2:
        raise SmallEPContractError(
            f"SmallEP top-k tensors must be rank 2, got {topk_ids.ndim}"
        )
    if topk_ids.numel() and (
        bool((topk_ids < 0).any())
        or bool((topk_ids >= len(layout.expert_to_rank)).any())
    ):
        raise SmallEPContractError("SmallEP top-k ids contain an invalid expert")

    owned = layout.owned_experts(rank)
    global_to_local = torch.full(
        (len(layout.expert_to_rank),),
        -1,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    owned_tensor = torch.tensor(
        owned,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    global_to_local[owned_tensor] = torch.arange(
        len(owned),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    local_ids = global_to_local[topk_ids]
    local_weights = torch.where(
        local_ids >= 0,
        topk_weights,
        torch.zeros((), dtype=topk_weights.dtype, device=topk_weights.device),
    )
    return SmallEPLocalRoutes(
        local_topk_ids=local_ids,
        local_topk_weights=local_weights,
        owned_global_experts=owned,
    )


def compute_small_ep_reference_partial(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: SmallEPLayout,
    rank: int,
    expert: Callable[[int, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Reference local weighted reduction used by CPU/mock correctness tests.

    Production fused-MoE code should consume :func:`localize_small_ep_routes`
    directly.  This deliberately simple implementation preserves repeated
    token/expert choices and uses ``index_add_`` to define the expected result.
    """

    if hidden_states.ndim != 2:
        raise SmallEPContractError(
            f"SmallEP hidden states must be rank 2, got {hidden_states.ndim}"
        )
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise SmallEPContractError(
            "SmallEP routing token count does not match gathered hidden states"
        )
    localize_small_ep_routes(topk_ids, topk_weights, layout, rank)

    output = torch.zeros_like(hidden_states)
    for global_expert_id in layout.owned_experts(rank):
        token_indices, choice_indices = torch.where(
            topk_ids == global_expert_id
        )
        if token_indices.numel() == 0:
            continue
        transformed = expert(
            global_expert_id,
            hidden_states[token_indices],
        )
        if transformed.shape != hidden_states[token_indices].shape:
            raise SmallEPContractError(
                f"expert {global_expert_id} returned shape "
                f"{tuple(transformed.shape)}, expected "
                f"{tuple(hidden_states[token_indices].shape)}"
            )
        weighted = transformed * topk_weights[
            token_indices, choice_indices
        ].unsqueeze(-1)
        output.index_add_(0, token_indices, weighted)
    return output


def reduce_small_ep_reference_partials(
    partials: Sequence[torch.Tensor],
    source_lengths: Sequence[int],
    rank: int,
) -> torch.Tensor:
    """Sum already hidden-reduced rank partials and select one source stripe."""

    if len(partials) != len(source_lengths):
        raise SmallEPContractError(
            "SmallEP needs one local partial and source length per rank"
        )
    if not partials:
        raise SmallEPContractError("SmallEP partial list must not be empty")
    if any(partial.shape != partials[0].shape for partial in partials):
        raise SmallEPContractError("SmallEP partial tensors must have equal shapes")
    if sum(source_lengths) != partials[0].shape[0]:
        raise SmallEPContractError(
            "SmallEP source lengths do not cover the gathered token dimension"
        )
    gathered = SmallEPGatheredBatch(
        hidden_states=partials[0],
        source_lengths=tuple(int(length) for length in source_lengths),
    )
    reduced = torch.stack(tuple(partials), dim=0).sum(dim=0)
    return reduced[gathered.source_slice(rank)]


class TorchSmallEPCollective:
    """Executable torch.distributed implementation of the SmallEP exchanges."""

    def __init__(self, group: Optional[dist.ProcessGroup] = None):
        if not dist.is_initialized():
            raise SmallEPContractError(
                "torch.distributed must be initialized before SmallEP"
            )
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world_size != 2:
            raise SmallEPContractError(
                f"the first SmallEP collective requires two ranks, got {self.world_size}"
            )

    def gather_unsorted(
        self,
        local_hidden_states: torch.Tensor,
    ) -> SmallEPGatheredBatch:
        """All-gather rank-major hidden states without gate/sort metadata."""

        if local_hidden_states.ndim != 2:
            raise SmallEPContractError(
                "SmallEP local hidden states must have [tokens, hidden] shape"
            )
        local_length = torch.tensor(
            [local_hidden_states.shape[0]],
            dtype=torch.int64,
            device=local_hidden_states.device,
        )
        gathered_lengths = [torch.empty_like(local_length) for _ in range(2)]
        dist.all_gather(gathered_lengths, local_length, group=self.group)
        source_lengths = tuple(int(length.item()) for length in gathered_lengths)
        maximum_length = max(source_lengths)

        padded = torch.zeros(
            (maximum_length, local_hidden_states.shape[1]),
            dtype=local_hidden_states.dtype,
            device=local_hidden_states.device,
        )
        padded[: local_hidden_states.shape[0]].copy_(local_hidden_states)
        gathered = [torch.empty_like(padded) for _ in range(2)]
        dist.all_gather(gathered, padded, group=self.group)
        hidden_states = torch.cat(
            tuple(tensor[:length] for tensor, length in zip(gathered, source_lengths)),
            dim=0,
        )
        return SmallEPGatheredBatch(
            hidden_states=hidden_states,
            source_lengths=source_lengths,
        )

    def reduce_hidden_partials(
        self,
        local_partial: torch.Tensor,
        gathered: SmallEPGatheredBatch,
    ) -> torch.Tensor:
        """All-reduce hidden-size partials, then retain this rank's stripe."""

        if local_partial.shape != gathered.hidden_states.shape:
            raise SmallEPContractError(
                "SmallEP local partial must cover every gathered token"
            )
        dist.all_reduce(local_partial, op=dist.ReduceOp.SUM, group=self.group)
        return local_partial[gathered.source_slice(self.rank)]


def execute_small_ep(
    local_hidden_states: torch.Tensor,
    layout: SmallEPLayout,
    gate: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    local_experts: Callable[
        [torch.Tensor, SmallEPLocalRoutes], torch.Tensor
    ],
    collective: TorchSmallEPCollective,
) -> torch.Tensor:
    """Run the complete collective contract with injectable gate/expert kernels."""

    if layout.world_size != collective.world_size:
        raise SmallEPContractError(
            "SmallEP ownership and collective world sizes do not match"
        )
    gathered = collective.gather_unsorted(local_hidden_states)
    topk_ids, topk_weights = gate(gathered.hidden_states)
    routes = localize_small_ep_routes(
        topk_ids,
        topk_weights,
        layout,
        collective.rank,
    )
    local_partial = local_experts(gathered.hidden_states, routes)
    return collective.reduce_hidden_partials(local_partial, gathered)


def small_ep_payload_bytes(
    source_tokens_per_rank: Sequence[int],
    hidden_size: int,
    element_size: int,
    rank: int,
) -> int:
    """Per-rank logical payload for one gather plus one hidden reduction."""

    if len(source_tokens_per_rank) != 2:
        raise SmallEPContractError("SmallEP payload sizing requires two ranks")
    if any(tokens < 0 for tokens in source_tokens_per_rank):
        raise SmallEPContractError("SmallEP token counts must not be negative")
    if rank not in (0, 1):
        raise SmallEPContractError(f"SmallEP payload rank must be 0 or 1, got {rank}")
    if hidden_size <= 0 or element_size <= 0:
        raise SmallEPContractError("hidden size and element size must be positive")
    total_tokens = sum(source_tokens_per_rank)
    # At two ranks, each rank sends/receives one peer stripe for the all-gather
    # and one global hidden-size partial for the all-reduce ring step.
    peer_tokens = total_tokens - source_tokens_per_rank[rank]
    gather_bytes = peer_tokens * hidden_size * element_size
    reduction_bytes = total_tokens * hidden_size * element_size
    return gather_bytes + reduction_bytes
