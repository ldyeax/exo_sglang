import unittest

import torch
import torch.nn.functional as functional
from sglang.srt.layers.quantization.v4_triton_kernels_moe import (
    apply_v4_triton_kernels_moe,
    convert_v4_weights_to_triton_kernels,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-large-1-gpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestV4Mxfp4PortableMoe(CustomTestCase):
    @staticmethod
    def _dequantize_mxfp4(
        packed: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        packed_u8 = packed.view(torch.uint8)
        values = torch.empty(
            (*packed_u8.shape[:-1], packed_u8.shape[-1] * 2),
            dtype=torch.uint8,
            device=packed.device,
        )
        values[..., 0::2] = packed_u8 & 0x0F
        values[..., 1::2] = packed_u8 >> 4
        e2m1 = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=packed.device,
        )
        sign = 1.0 - 2.0 * ((values >> 3) & 1).to(torch.float32)
        magnitude = e2m1[(values & 7).to(torch.long)]
        block_scale = scale.to(torch.float32).repeat_interleave(32, dim=-1)
        return (sign * magnitude * block_scale).to(torch.bfloat16)

    @classmethod
    def _reference_moe(
        cls,
        hidden_states: torch.Tensor,
        w13_packed: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_packed: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        w13 = cls._dequantize_mxfp4(w13_packed, w13_scale)
        w2 = cls._dequantize_mxfp4(w2_packed, w2_scale)
        intermediate_size = w13.shape[1] // 2
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for token_index in range(hidden_states.shape[0]):
            token = hidden_states[token_index : token_index + 1]
            for slot_index in range(topk_ids.shape[1]):
                expert_index = int(topk_ids[token_index, slot_index])
                gate_up = token @ w13[expert_index].transpose(0, 1)
                intermediate = (
                    functional.silu(
                        gate_up[:, :intermediate_size].to(torch.float32)
                    )
                    * gate_up[:, intermediate_size:].to(torch.float32)
                ).to(torch.bfloat16)
                expert_output = intermediate @ w2[expert_index].transpose(0, 1)
                output[token_index] += (
                    expert_output[0].to(torch.float32)
                    * topk_weights[token_index, slot_index].to(torch.float32)
                )
        return output.to(torch.bfloat16)

    def test_portable_moe_matches_direct_native_mxfp4(self):
        torch.manual_seed(20260726)
        device = torch.device("cuda")
        num_experts = 4
        num_tokens = 8
        hidden_size = 256
        intermediate_size = 128

        w13_packed = torch.randint(
            0,
            256,
            (num_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2_packed = torch.randint(
            0,
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w13_scale = torch.randint(
            122,
            128,
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e8m0fnu)
        w2_scale = torch.randint(
            122,
            128,
            (num_experts, hidden_size, intermediate_size // 32),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e8m0fnu)
        hidden_states = (
            torch.randn(num_tokens, hidden_size, device=device) / 4
        ).to(torch.bfloat16)
        topk_ids = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 3],
                [0, 2],
                [3, 0],
                [2, 1],
                [0, 3],
                [1, 2],
            ],
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.25, 0.75],
                [0.9, 0.1],
                [0.65, 0.35],
                [0.55, 0.45],
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        expected = self._reference_moe(
            hidden_states,
            w13_packed,
            w13_scale,
            w2_packed,
            w2_scale,
            topk_ids,
            topk_weights,
        )
        w13, w13_precision, w2, w2_precision = (
            convert_v4_weights_to_triton_kernels(
                w13_packed,
                w13_scale,
                w2_packed,
                w2_scale,
            )
        )
        actual = apply_v4_triton_kernels_moe(
            hidden_states=hidden_states,
            w13_swiz=w13,
            w13_pcg=w13_precision,
            w2_swiz=w2,
            w2_pcg=w2_precision,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        )
        torch.cuda.synchronize()

        delta = (actual.float() - expected.float()).abs()
        relative_l2 = delta.norm() / expected.float().norm()
        self.assertLessEqual(delta.max().item(), 8.0)
        self.assertLessEqual(relative_l2.item(), 0.005)


if __name__ == "__main__":
    unittest.main()
