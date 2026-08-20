"""CPU coverage for tensor-parallel FR-Spec LM-head construction."""

import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.speculative.spec_utils import (
    _hot_lm_head_shard_contribution,
    build_tp_hot_lm_head,
    load_token_map,
    prepare_fp8_marlin_hot_lm_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeAllReduceGroup:
    def __init__(
        self,
        *,
        rank: int,
        hot_token_id: torch.Tensor,
        peer_weight: torch.Tensor,
        peer_shard_indices: VocabParallelEmbeddingShardIndices,
    ) -> None:
        self.world_size = 2
        self.rank_in_group = rank
        self._hot_token_id = hot_token_id
        self._peer_weight = peer_weight
        self._peer_shard_indices = peer_shard_indices
        self._call_index = 0

    def all_reduce(self, contribution: torch.Tensor) -> torch.Tensor:
        tokens_per_partition = self._hot_token_id.numel() // self.world_size
        start = self._call_index * tokens_per_partition
        destination_token_ids = self._hot_token_id.narrow(
            0, start, tokens_per_partition
        )
        self._call_index += 1
        peer_contribution = _hot_lm_head_shard_contribution(
            self._peer_weight,
            destination_token_ids,
            shard_indices=self._peer_shard_indices,
        )
        return contribution + peer_contribution


class TestTPHotLMHead(CustomTestCase):
    def test_tp2_reconstructs_hot_order_from_vocab_shards(self) -> None:
        full_weight = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        hot_token_id = torch.tensor([7, 1, 9, 0, 6, 4], dtype=torch.int64)
        shard_indices = [
            VocabParallelEmbedding._get_indices(12, 12, 10, 10, rank, 2)
            for rank in range(2)
        ]
        local_weights = [
            full_weight[:6].clone(),
            torch.cat((full_weight[6:].clone(), torch.zeros((2, 3))), dim=0),
        ]

        local_hot_weights: list[torch.Tensor] = []
        for rank in range(2):
            fake_group = _FakeAllReduceGroup(
                rank=rank,
                hot_token_id=hot_token_id,
                peer_weight=local_weights[1 - rank],
                peer_shard_indices=shard_indices[1 - rank],
            )
            local_hot_weights.append(
                build_tp_hot_lm_head(
                    local_weights[rank],
                    hot_token_id,
                    num_embeddings=10,
                    shard_indices=shard_indices[rank],
                    tp_group=cast(GroupCoordinator, fake_group),
                )
            )

        gathered_hot_weight = torch.cat(local_hot_weights, dim=0)
        torch.testing.assert_close(
            gathered_hot_weight, full_weight.index_select(0, hot_token_id)
        )

        hidden = torch.tensor([[0.25, -0.5, 1.0]])
        reduced_logits = hidden @ gathered_hot_weight.T
        reduced_index = torch.argmax(reduced_logits, dim=-1)
        global_token_id = hot_token_id[reduced_index]
        expected_token_id = hot_token_id[
            torch.argmax(hidden @ full_weight.index_select(0, hot_token_id).T, dim=-1)
        ]
        torch.testing.assert_close(global_token_id, expected_token_id)

    def test_map_length_must_be_divisible_by_tp_size(self) -> None:
        shard_indices = VocabParallelEmbedding._get_indices(4, 4, 4, 4, 0, 2)
        fake_group = _FakeAllReduceGroup(
            rank=0,
            hot_token_id=torch.tensor([0, 1, 2]),
            peer_weight=torch.zeros((2, 2)),
            peer_shard_indices=VocabParallelEmbedding._get_indices(4, 4, 4, 4, 1, 2),
        )

        with self.assertRaisesRegex(ValueError, "divisible by tensor parallel"):
            build_tp_hot_lm_head(
                torch.zeros((2, 2)),
                torch.tensor([0, 1, 2]),
                num_embeddings=4,
                shard_indices=shard_indices,
                tp_group=cast(GroupCoordinator, fake_group),
            )

    def test_token_map_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_map_path = Path(temp_dir) / "duplicate.pt"
            torch.save(torch.tensor([4, 2, 4]), token_map_path)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_token_map(str(token_map_path))

    def test_fp8_marlin_packs_only_private_draft_head(self) -> None:
        target = torch.nn.Module()
        target.weight = torch.nn.Parameter(
            torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
            requires_grad=False,
        )
        draft = torch.nn.Module()
        draft.weight = torch.nn.Parameter(
            target.weight.detach().index_select(0, torch.tensor([5, 1, 3])).clone(),
            requires_grad=False,
        )
        target_before = target.weight.detach().clone()

        class FakeFp8MarlinMethod:
            use_marlin = True
            quant_config = object()

            def process_weights_after_loading(self, layer) -> None:
                self.seen_metadata = (
                    layer.input_size_per_partition,
                    layer.output_size_per_partition,
                    layer.logical_widths,
                    layer.orig_dtype,
                )
                layer.weight = torch.nn.Parameter(
                    torch.zeros((4, 3), dtype=torch.int32), requires_grad=False
                )

        quant_method = FakeFp8MarlinMethod()
        prepare_fp8_marlin_hot_lm_head(draft, target, quant_method=quant_method)

        self.assertEqual(quant_method.seen_metadata, (4, 3, [3], torch.bfloat16))
        self.assertIs(draft.quant_method, quant_method)
        self.assertIs(draft.quant_config, quant_method.quant_config)
        self.assertEqual(draft.weight.dtype, torch.int32)
        torch.testing.assert_close(target.weight, target_before)

    def test_fp8_marlin_rejects_target_storage_alias(self) -> None:
        target = torch.nn.Module()
        target.weight = torch.nn.Parameter(
            torch.zeros((4, 4), dtype=torch.bfloat16), requires_grad=False
        )
        draft = torch.nn.Module()
        draft.weight = target.weight

        with self.assertRaisesRegex(ValueError, "storage shared with target"):
            prepare_fp8_marlin_hot_lm_head(draft, target, quant_method=object())


if __name__ == "__main__":
    unittest.main()
