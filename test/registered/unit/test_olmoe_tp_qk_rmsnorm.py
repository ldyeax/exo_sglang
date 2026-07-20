import unittest
from unittest.mock import patch

import torch

# Importing the model pulls in fused-MoE registration that probes CUDA on some
# installations. These tests exercise only the CPU tensor-parallel norm math.
with (
    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    patch.object(torch.cuda, "current_device", return_value=0),
):
    from sglang.srt.models import olmoe


class TestOlmoeQKRMSNormTP(unittest.TestCase):
    tp_world = 2
    epsilon = 1e-5

    def _make_norm(
        self, rank: int, loaded_weight: torch.Tensor
    ) -> olmoe.OlmoeQKRMSNormTP:
        with (
            patch.object(
                olmoe,
                "get_parallel",
                return_value=type(
                    "ParallelContext",
                    (),
                    {"tp_size": self.tp_world, "tp_rank": rank},
                )(),
            ),
        ):
            norm = olmoe.OlmoeQKRMSNormTP(loaded_weight.numel(), eps=self.epsilon)
            norm.weight.weight_loader(norm.weight, loaded_weight)
        return norm

    @staticmethod
    def _full_rms_norm(
        value: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        variance = value.to(torch.float32).square().mean(dim=-1, keepdim=True)
        return (value.to(torch.float32) * torch.rsqrt(variance + epsilon) * weight).to(
            value.dtype
        )

    def test_weight_loader_selects_each_rank_local_slice(self) -> None:
        loaded_weight = torch.tensor(
            [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            dtype=torch.float32,
        )

        rank_zero = self._make_norm(0, loaded_weight)
        rank_one = self._make_norm(1, loaded_weight)

        torch.testing.assert_close(rank_zero.weight, loaded_weight[:4])
        torch.testing.assert_close(rank_one.weight, loaded_weight[4:])

    def test_tp2_outputs_and_statistics_match_full_vector_reference(self) -> None:
        q = torch.tensor(
            [
                [0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0],
                [1.25, 0.75, -0.5, -1.5, 2.25, -2.75, 3.25, -3.75],
                [-0.25, 0.625, -1.125, 1.875, -2.375, 2.875, -3.375, 4.0],
            ],
            dtype=torch.bfloat16,
        )
        k = torch.tensor(
            [
                [-1.5, 1.0, -0.5, 0.25, 2.0, -2.5, 3.0, -3.5],
                [0.375, -0.875, 1.375, -1.875, 2.375, -2.875, 3.375, -3.875],
                [4.0, -3.25, 2.5, -1.75, 1.0, -0.625, 0.375, -0.125],
            ],
            dtype=torch.bfloat16,
        )
        q_weight = torch.linspace(0.5, 1.5, q.shape[-1], dtype=torch.float32)
        k_weight = torch.linspace(1.75, 0.75, k.shape[-1], dtype=torch.float32)
        q_shards = q.chunk(self.tp_world, dim=-1)
        k_shards = k.chunk(self.tp_world, dim=-1)
        local_sums = tuple(
            olmoe._olmoe_qk_rms_norm_local_sums(q_shard, k_shard)
            for q_shard, k_shard in zip(q_shards, k_shards, strict=True)
        )
        reduced_sums = local_sums[0] + local_sums[1]
        full_sums = torch.cat(
            (
                q.to(torch.float32).square().sum(dim=-1, keepdim=True),
                k.to(torch.float32).square().sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        collective_payloads = []

        def all_reduce(local_sum: torch.Tensor) -> torch.Tensor:
            collective_payloads.append(local_sum.clone())
            return reduced_sums

        q_outputs = []
        k_outputs = []
        with patch.object(
            olmoe, "tensor_model_parallel_all_reduce", side_effect=all_reduce
        ):
            for rank, (q_shard, k_shard) in enumerate(
                zip(q_shards, k_shards, strict=True)
            ):
                q_norm = self._make_norm(rank, q_weight)
                k_norm = self._make_norm(rank, k_weight)
                q_output, k_output = olmoe.OlmoeQKRMSNormTP.forward_qk(
                    q_norm,
                    k_norm,
                    q_shard.contiguous(),
                    k_shard.contiguous(),
                )
                q_outputs.append(q_output)
                k_outputs.append(k_output)

        self.assertEqual(len(collective_payloads), self.tp_world)
        self.assertEqual(
            [tuple(payload.shape) for payload in collective_payloads],
            [(q.shape[0], 2), (q.shape[0], 2)],
        )
        torch.testing.assert_close(collective_payloads[0], local_sums[0])
        torch.testing.assert_close(collective_payloads[1], local_sums[1])
        torch.testing.assert_close(reduced_sums, full_sums)
        torch.testing.assert_close(
            torch.cat(q_outputs, dim=-1),
            self._full_rms_norm(q, q_weight, self.epsilon),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.cat(k_outputs, dim=-1),
            self._full_rms_norm(k, k_weight, self.epsilon),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
