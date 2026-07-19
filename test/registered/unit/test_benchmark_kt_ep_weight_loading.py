import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.srt.layers.moe.benchmark_kt_ep import BenchmarkKTWrapper


class TestBenchmarkKtEpWeightLoading(unittest.TestCase):
    hidden_size = 3
    intermediate_size = 2
    num_gpu_experts = 2
    layer_idx = 1

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_wrapper(self):
        wrapper = BenchmarkKTWrapper.__new__(BenchmarkKTWrapper)
        wrapper.num_gpu_experts = self.num_gpu_experts
        wrapper.hidden_size = self.hidden_size
        wrapper.intermediate_size = self.intermediate_size
        wrapper.params_dtype = torch.bfloat16
        wrapper.mock_layer = SimpleNamespace(
            w13_weight=torch.full(
                (
                    self.num_gpu_experts,
                    2 * self.intermediate_size,
                    self.hidden_size,
                ),
                -1,
                dtype=torch.bfloat16,
            ),
            w2_weight=torch.full(
                (
                    self.num_gpu_experts,
                    self.hidden_size,
                    self.intermediate_size,
                ),
                -1,
                dtype=torch.bfloat16,
            ),
        )
        return wrapper

    def _checkpoint_weights(self):
        weights = {}
        for expert_idx in range(self.num_gpu_experts):
            prefix = f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}"
            base = 100 * expert_idx
            weights[f"{prefix}.gate_proj.weight"] = (
                torch.arange(
                    self.intermediate_size * self.hidden_size,
                    dtype=torch.float32,
                ).reshape(self.intermediate_size, self.hidden_size)
                + base
                + 10
            ).to(torch.bfloat16)
            weights[f"{prefix}.up_proj.weight"] = (
                torch.arange(
                    self.intermediate_size * self.hidden_size,
                    dtype=torch.float32,
                ).reshape(self.intermediate_size, self.hidden_size)
                + base
                + 20
            ).to(torch.bfloat16)
            weights[f"{prefix}.down_proj.weight"] = (
                torch.arange(
                    self.hidden_size * self.intermediate_size,
                    dtype=torch.float32,
                ).reshape(self.hidden_size, self.intermediate_size)
                + base
                + 30
            ).to(torch.bfloat16)
        return weights

    def _save(self, weights, name="model.safetensors"):
        save_file(weights, self.model_path / name)

    def _assert_destinations_unchanged(self, wrapper):
        torch.testing.assert_close(
            wrapper.mock_layer.w13_weight,
            torch.full_like(wrapper.mock_layer.w13_weight, -1),
        )
        torch.testing.assert_close(
            wrapper.mock_layer.w2_weight,
            torch.full_like(wrapper.mock_layer.w2_weight, -1),
        )

    def test_loads_non_triton_gate_up_down_layout(self):
        weights = self._checkpoint_weights()
        self._save(weights)
        wrapper = self._make_wrapper()

        loaded_count = wrapper.load_gpu_weights(
            str(self.model_path), layer_idx=self.layer_idx
        )

        self.assertEqual(loaded_count, 3 * self.num_gpu_experts)
        for expert_idx in range(self.num_gpu_experts):
            prefix = f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}"
            torch.testing.assert_close(
                wrapper.mock_layer.w13_weight[expert_idx, : self.intermediate_size, :],
                weights[f"{prefix}.gate_proj.weight"],
            )
            torch.testing.assert_close(
                wrapper.mock_layer.w13_weight[expert_idx, self.intermediate_size :, :],
                weights[f"{prefix}.up_proj.weight"],
            )
            torch.testing.assert_close(
                wrapper.mock_layer.w2_weight[expert_idx],
                weights[f"{prefix}.down_proj.weight"],
            )

    def test_rejects_missing_projection_before_copy(self):
        weights = self._checkpoint_weights()
        del weights[f"model.layers.{self.layer_idx}.mlp.experts.1.up_proj.weight"]
        self._save(weights)
        wrapper = self._make_wrapper()

        with self.assertRaisesRegex(
            RuntimeError, "Missing GPU expert projections.*expert 1 up"
        ):
            wrapper.load_gpu_weights(str(self.model_path), self.layer_idx)

        self._assert_destinations_unchanged(wrapper)

    def test_rejects_duplicate_projection_before_copy(self):
        weights = self._checkpoint_weights()
        weights[f"model.layers.{self.layer_idx}.mlp.experts.0.w1.weight"] = weights[
            f"model.layers.{self.layer_idx}.mlp.experts.0.gate_proj.weight"
        ].clone()
        self._save(weights)
        wrapper = self._make_wrapper()

        with self.assertRaisesRegex(
            RuntimeError, "Duplicate GPU expert projection for expert 0 gate"
        ):
            wrapper.load_gpu_weights(str(self.model_path), self.layer_idx)

        self._assert_destinations_unchanged(wrapper)

    def test_rejects_checkpoint_shape_mismatch_before_copy(self):
        weights = self._checkpoint_weights()
        malformed_key = f"model.layers.{self.layer_idx}.mlp.experts.0.gate_proj.weight"
        weights[malformed_key] = torch.zeros(
            self.hidden_size,
            self.intermediate_size,
            dtype=torch.bfloat16,
        )
        self._save(weights)
        wrapper = self._make_wrapper()

        with self.assertRaisesRegex(
            RuntimeError,
            r"checkpoint shape mismatch.*expected \(2, 3\), got \(3, 2\)",
        ):
            wrapper.load_gpu_weights(str(self.model_path), self.layer_idx)

        self._assert_destinations_unchanged(wrapper)


if __name__ == "__main__":
    unittest.main()
