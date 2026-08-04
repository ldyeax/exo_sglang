from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_dspark
from sglang.srt.models.deepseek_v4_dspark import (
    DSparkV4MarkovHead,
    MarkovW2ShardGeometry,
)


class _TwoRankGather:
    world_size = 2

    def __init__(self, rank_one: torch.Tensor) -> None:
        self.rank_one = rank_one

    def all_gather(self, rank_zero: torch.Tensor, *, dim: int) -> torch.Tensor:
        return torch.cat((rank_zero, self.rank_one), dim=dim)


def test_fp32_tp_sharded_markov_logits_match_reference(monkeypatch):
    """Adding each vocab shard before all-gather must preserve reference logits."""
    with (
        envs.SGLANG_DSPARK_OPT_MARKOV_W2_BF16.override(False),
        envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.override(True),
    ):
        head = DSparkV4MarkovHead(vocab_size=8, markov_rank=4)

    embedding_weight = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, -0.6, 0.7, -0.8],
            [-0.2, 0.4, -0.6, 0.8],
            [0.9, 0.3, -0.5, -0.7],
            [-0.1, -0.2, 0.3, 0.4],
            [0.6, 0.5, -0.4, -0.3],
            [0.2, -0.8, 0.4, -0.6],
            [-0.9, 0.7, -0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    projection_weight = torch.arange(32, dtype=torch.float32).view(8, 4) / 17
    head.markov_w2.weight.data.copy_(projection_weight)
    head._tp_shard = MarkovW2ShardGeometry(
        tp_size=2,
        org_vocab_start=0,
        org_vocab_end=4,
        num_embeddings_per_partition=4,
        num_embeddings_padded=8,
    )

    token_ids = torch.tensor([1, 6])
    base_full = torch.tensor(
        [[0.2, -0.1, 0.4, 0.8, -0.5, 0.7, 0.3, -0.9]] * 2,
        dtype=torch.float32,
    )
    embeddings = functional.embedding(token_ids, embedding_weight)
    bias_full = functional.linear(embeddings, projection_weight)
    rank_one = base_full[:, 4:] + bias_full[:, 4:]
    gather = _TwoRankGather(rank_one)
    monkeypatch.setattr(
        head,
        "get_prev_embeddings",
        lambda ids: functional.embedding(ids, embedding_weight),
    )
    monkeypatch.setattr(
        deepseek_v4_dspark,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_group=gather),
    )

    actual = head.apply_step_logits(
        base_full[:, :4], token_ids=token_ids, hidden_states=None
    )

    torch.testing.assert_close(actual, base_full + bias_full, rtol=0, atol=0)
