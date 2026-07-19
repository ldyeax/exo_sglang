import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

# GLM coverage validation is pure metadata inspection. Keep module import from
# probing a physical CUDA device when this test runs on a CPU-only worker.
with (
    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    patch.object(torch.cuda, "current_device", return_value=0),
):
    from sglang.srt.models import glm4_moe_lite as glm


class _FakeFusedMoE:
    pass


class _FakeSparseMoeBlock:
    pass


class TestGlm47FlashKtEpCoverage(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            num_hidden_layers=47,
            first_k_dense_replace=1,
            n_routed_experts=64,
            moe_layer_freq=1,
        )

    def _server_args(self, **overrides):
        values = {
            "kt_weight_path": "/models/glm47-kt",
            "kt_num_gpu_experts": 3,
            "kt_method": "BF16",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _coverage_fixture(self):
        config = self._config()
        masks = torch.zeros((47, 64), dtype=torch.bool)
        masks[0, :] = True
        masks[1:, :3] = True
        layers = [SimpleNamespace(layer_id=0, mlp=object())]
        for layer_id in range(1, 47):
            kt_config = SimpleNamespace(
                layer_idx=layer_id,
                num_layers=47,
                gpu_experts_mask=masks[layer_id],
            )
            quant_method = SimpleNamespace(
                _quant_wrapper_id="kt_ep",
                kt_config=kt_config,
            )
            experts = _FakeFusedMoE()
            experts.layer_id = layer_id
            experts._registry_prefix = f"model.layers.{layer_id}.mlp.experts"
            experts.quant_method = quant_method
            experts.moe_runner_config = SimpleNamespace(layer_id=layer_id)
            sparse_block = _FakeSparseMoeBlock()
            sparse_block.layer_id = layer_id
            sparse_block.config = config
            sparse_block.experts = experts
            layers.append(SimpleNamespace(layer_id=layer_id, mlp=sparse_block))
        return config, masks, layers

    def _build_receipt(self, config, masks, layers, server_args=None):
        fake_kt_module = SimpleNamespace(
            get_kt_ep_gpu_experts_masks=lambda: masks,
        )
        with (
            patch.object(glm, "FusedMoE", _FakeFusedMoE),
            patch.object(glm, "Glm4MoeLiteSparseMoeBlock", _FakeSparseMoeBlock),
            patch.object(
                glm,
                "get_global_server_args",
                return_value=server_args or self._server_args(),
            ),
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.kt_ep_wrapper": fake_kt_module},
            ),
        ):
            return glm._build_glm47_flash_kt_ep_coverage_receipt(
                layers=layers,
                config=config,
                prefix="model",
            )

    def test_preconstruction_requirement_accepts_exact_profile(self):
        require_registration = Mock()
        fake_kt_module = SimpleNamespace(
            require_kt_ep_registration=require_registration,
        )
        with (
            patch.object(
                glm,
                "get_global_server_args",
                return_value=self._server_args(),
            ),
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.kt_ep_wrapper": fake_kt_module},
            ),
        ):
            enabled = glm._require_glm47_flash_kt_ep(
                config=self._config(), pp_size=1, prefix="model"
            )

        self.assertTrue(enabled)
        require_registration.assert_called_once_with()

    def test_preconstruction_requirement_rejects_pipeline_parallelism(self):
        fake_kt_module = SimpleNamespace(require_kt_ep_registration=lambda: None)
        with (
            patch.object(
                glm,
                "get_global_server_args",
                return_value=self._server_args(),
            ),
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.kt_ep_wrapper": fake_kt_module},
            ),
            self.assertRaisesRegex(RuntimeError, "pipeline parallel size 1"),
        ):
            glm._require_glm47_flash_kt_ep(
                config=self._config(), pp_size=2, prefix="model"
            )

    def test_coverage_receipt_binds_all_46_routed_layers(self):
        config, masks, layers = self._coverage_fixture()

        receipt = self._build_receipt(config, masks, layers)

        self.assertEqual(receipt["gpu_expert_mask_shape"], [47, 64])
        self.assertEqual(receipt["configured_gpu_resident_experts_per_layer"], 3)
        self.assertEqual(receipt["routed_layer_ids"], list(range(1, 47)))
        self.assertEqual(len(receipt["layers"]), 46)
        self.assertEqual(
            receipt["layers"][0]["module_path"],
            "model.layers.1.mlp.experts",
        )
        self.assertEqual(
            receipt["layers"][-1]["module_path"],
            "model.layers.46.mlp.experts",
        )
        self.assertEqual(len(receipt["gpu_expert_mask_sha256"]), 64)

    def test_coverage_rejects_noncanonical_fused_moe_path(self):
        config, masks, layers = self._coverage_fixture()
        layers[12].mlp.experts._registry_prefix = "model.layers.11.mlp.experts"

        with self.assertRaisesRegex(RuntimeError, "registry path mismatch"):
            self._build_receipt(config, masks, layers)

    def test_coverage_rejects_unwrapped_layer(self):
        config, masks, layers = self._coverage_fixture()
        layers[23].mlp.experts.quant_method._quant_wrapper_id = "other"

        with self.assertRaisesRegex(RuntimeError, "is not wrapped by 'kt_ep'"):
            self._build_receipt(config, masks, layers)

    def test_coverage_rejects_stale_layer_mask(self):
        config, masks, layers = self._coverage_fixture()
        stale_mask = masks[34].clone()
        stale_mask[3] = True
        layers[34].mlp.experts.quant_method.kt_config.gpu_experts_mask = stale_mask

        with self.assertRaisesRegex(RuntimeError, "not linked to placement row 34"):
            self._build_receipt(config, masks, layers)

    def test_coverage_rejects_layer_linkage_mismatch(self):
        config, masks, layers = self._coverage_fixture()
        layers[45].mlp.experts.quant_method.kt_config.layer_idx = 44

        with self.assertRaisesRegex(RuntimeError, "layer linkage mismatch"):
            self._build_receipt(config, masks, layers)


if __name__ == "__main__":
    unittest.main()
