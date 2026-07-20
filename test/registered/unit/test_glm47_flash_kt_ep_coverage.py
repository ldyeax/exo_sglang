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


class _FakeKTEPWrapperMethod:
    def __init__(self, kt_config):
        self.kt_config = kt_config


class _FakeGlm4MoeLiteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kt_ep_coverage_receipt = None
        self.layers = torch.nn.ModuleList()


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
            quant_method = _FakeKTEPWrapperMethod(kt_config)
            experts = _FakeFusedMoE()
            experts.layer_id = layer_id
            experts.layer_name = f"model.layers.{layer_id}.mlp.experts"
            experts.quant_method = quant_method
            experts.moe_runner_config = SimpleNamespace(layer_id=layer_id)
            sparse_block = _FakeSparseMoeBlock()
            sparse_block.layer_id = layer_id
            sparse_block.config = config
            sparse_block.experts = experts
            sparse_block.is_hash = False
            layers.append(SimpleNamespace(layer_id=layer_id, mlp=sparse_block))
        return config, masks, layers

    def _construct_causal_model(self, *, is_last_rank):
        config = SimpleNamespace(vocab_size=154880, hidden_size=2048)
        pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        model_body = _FakeGlm4MoeLiteModel()
        expected_lm_head = torch.nn.Identity()
        lm_head_constructor = Mock(return_value=expected_lm_head)
        server_args = SimpleNamespace(enable_dp_lm_head=False)

        with (
            patch.object(glm, "get_pp_group", return_value=pp_group),
            patch.object(
                glm,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
            patch.object(
                glm.Glm4MoeLiteForCausalLM,
                "determine_num_fused_shared_experts",
            ),
            patch.object(glm, "Glm4MoeLiteModel", return_value=model_body),
            patch.object(glm, "ParallelLMHead", lm_head_constructor),
            patch.object(glm, "LogitsProcessor", return_value=torch.nn.Identity()),
            patch.object(glm, "get_server_args", return_value=server_args),
        ):
            model = glm.Glm4MoeLiteForCausalLM(config)

        return model, expected_lm_head, lm_head_constructor

    def _build_receipt(
        self,
        config,
        masks,
        layers,
        server_args=None,
        *,
        pp_rank=0,
        pp_size=1,
        start_layer=0,
        end_layer=47,
    ):
        fake_kt_module = SimpleNamespace(
            get_kt_ep_gpu_experts_masks=lambda: masks,
        )
        with (
            patch.object(glm, "FusedMoE", _FakeFusedMoE),
            patch.object(glm, "KTEPWrapperMethod", _FakeKTEPWrapperMethod),
            patch.object(glm, "Glm4MoeLiteSparseMoeBlock", _FakeSparseMoeBlock),
            patch.object(
                glm,
                "get_server_args",
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
                pp_rank=pp_rank,
                pp_size=pp_size,
                start_layer=start_layer,
                end_layer=end_layer,
            )

    def test_preconstruction_requirement_accepts_exact_profile(self):
        require_registration = Mock()
        fake_kt_module = SimpleNamespace(
            require_kt_ep_registration=require_registration,
        )
        with (
            patch.object(
                glm,
                "get_server_args",
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

    def test_non_last_pipeline_rank_does_not_allocate_lm_head(self):
        model, _, lm_head_constructor = self._construct_causal_model(is_last_rank=False)

        lm_head_constructor.assert_not_called()
        self.assertIsInstance(model.lm_head, glm.PPMissingLayer)
        self.assertNotIn("lm_head.weight", dict(model.named_parameters()))

    def test_last_pipeline_rank_retains_lm_head(self):
        model, expected_lm_head, lm_head_constructor = self._construct_causal_model(
            is_last_rank=True
        )

        self.assertIs(model.lm_head, expected_lm_head)
        lm_head_constructor.assert_called_once_with(
            154880,
            2048,
            quant_config=None,
            prefix="lm_head",
            use_attn_tp_group=False,
        )

    def test_preconstruction_requirement_accepts_pipeline_parallelism(self):
        fake_kt_module = SimpleNamespace(require_kt_ep_registration=lambda: None)
        with (
            patch.object(
                glm,
                "get_server_args",
                return_value=self._server_args(),
            ),
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.kt_ep_wrapper": fake_kt_module},
            ),
        ):
            enabled = glm._require_glm47_flash_kt_ep(
                config=self._config(), pp_size=3, prefix="model"
            )

        self.assertTrue(enabled)

    def test_preconstruction_requirement_rejects_invalid_pipeline_size(self):
        fake_kt_module = SimpleNamespace(require_kt_ep_registration=lambda: None)
        with (
            patch.object(
                glm,
                "get_server_args",
                return_value=self._server_args(),
            ),
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.kt_ep_wrapper": fake_kt_module},
            ),
            self.assertRaisesRegex(RuntimeError, "between 1 and 47"),
        ):
            glm._require_glm47_flash_kt_ep(
                config=self._config(), pp_size=0, prefix="model"
            )

    def test_preconstruction_requirement_rejects_hash_layers(self):
        config = self._config()
        config.n_hash_layers = 1
        fake_kt_module = SimpleNamespace(require_kt_ep_registration=lambda: None)
        with (
            patch.object(
                glm,
                "get_server_args",
                return_value=self._server_args(),
            ),
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.kt_ep_wrapper": fake_kt_module},
            ),
            self.assertRaisesRegex(RuntimeError, "does not support hash-routed MoE"),
        ):
            glm._require_glm47_flash_kt_ep(config=config, pp_size=1, prefix="model")

    def test_coverage_receipt_binds_all_46_routed_layers(self):
        config, masks, layers = self._coverage_fixture()

        receipt = self._build_receipt(config, masks, layers)

        self.assertEqual(receipt["gpu_expert_mask_shape"], [47, 64])
        self.assertEqual(receipt["configured_gpu_resident_experts_per_layer"], 3)
        self.assertEqual(receipt["pipeline_parallel_rank"], 0)
        self.assertEqual(receipt["pipeline_parallel_size"], 1)
        self.assertEqual(receipt["pipeline_layer_start"], 0)
        self.assertEqual(receipt["pipeline_layer_end"], 47)
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

    def test_coverage_receipt_binds_each_pp3_stage_local_range(self):
        stage_ranges = ((0, 16), (16, 32), (32, 47))
        for pp_rank, (start_layer, end_layer) in enumerate(stage_ranges):
            with self.subTest(pp_rank=pp_rank):
                config, masks, all_layers = self._coverage_fixture()
                layers = [
                    layer
                    if start_layer <= layer_id < end_layer
                    else SimpleNamespace(layer_id=layer_id, mlp=object())
                    for layer_id, layer in enumerate(all_layers)
                ]

                receipt = self._build_receipt(
                    config,
                    masks,
                    layers,
                    pp_rank=pp_rank,
                    pp_size=3,
                    start_layer=start_layer,
                    end_layer=end_layer,
                )

                expected_layer_ids = list(range(max(start_layer, 1), end_layer))
                self.assertEqual(receipt["pipeline_parallel_rank"], pp_rank)
                self.assertEqual(receipt["pipeline_parallel_size"], 3)
                self.assertEqual(receipt["pipeline_layer_start"], start_layer)
                self.assertEqual(receipt["pipeline_layer_end"], end_layer)
                self.assertEqual(receipt["routed_layer_ids"], expected_layer_ids)
                self.assertEqual(
                    [layer["layer_id"] for layer in receipt["layers"]],
                    expected_layer_ids,
                )

    def test_coverage_rejects_routed_layer_outside_local_range(self):
        config, masks, all_layers = self._coverage_fixture()
        layers = [
            layer
            if 16 <= layer_id < 32 or layer_id == 15
            else SimpleNamespace(layer_id=layer_id, mlp=object())
            for layer_id, layer in enumerate(all_layers)
        ]

        with self.assertRaisesRegex(RuntimeError, "routed-layer coverage mismatch"):
            self._build_receipt(
                config,
                masks,
                layers,
                pp_rank=1,
                pp_size=3,
                start_layer=16,
                end_layer=32,
            )

    def test_coverage_rejects_invalid_local_pipeline_range(self):
        config, masks, layers = self._coverage_fixture()

        with self.assertRaisesRegex(RuntimeError, "invalid local pipeline range"):
            self._build_receipt(
                config,
                masks,
                layers,
                pp_rank=3,
                pp_size=3,
                start_layer=32,
                end_layer=47,
            )

    def test_coverage_rejects_noncanonical_fused_moe_path(self):
        config, masks, layers = self._coverage_fixture()
        layers[12].mlp.experts.layer_name = "model.layers.11.mlp.experts"

        with self.assertRaisesRegex(RuntimeError, "layer path mismatch"):
            self._build_receipt(config, masks, layers)

    def test_coverage_rejects_missing_hash_routing_state(self):
        config, masks, layers = self._coverage_fixture()
        del layers[11].mlp.is_hash

        with self.assertRaisesRegex(RuntimeError, "must disable DeepSeek hash routing"):
            self._build_receipt(config, masks, layers)

    def test_coverage_rejects_enabled_hash_routing(self):
        config, masks, layers = self._coverage_fixture()
        layers[11].mlp.is_hash = True

        with self.assertRaisesRegex(RuntimeError, "must disable DeepSeek hash routing"):
            self._build_receipt(config, masks, layers)

    def test_coverage_rejects_unwrapped_layer(self):
        config, masks, layers = self._coverage_fixture()
        layers[23].mlp.experts.quant_method = object()

        with self.assertRaisesRegex(RuntimeError, "is not wrapped by KTEPWrapperMethod"):
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
