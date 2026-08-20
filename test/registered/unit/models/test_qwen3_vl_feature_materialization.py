"""Regression tests for Qwen3-VL multimodal feature materialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingVisual:
    device = torch.device("meta")
    dtype = torch.bfloat16

    def __init__(self):
        self.pixel_values = None
        self.grid_thw = None

    def __call__(self, pixel_values, *, grid_thw):
        self.pixel_values = pixel_values
        self.grid_thw = grid_thw
        return pixel_values


class _FakeLanguageModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class _FakeVisualModel(nn.Module):
    def __init__(self, deepstack_visual_indexes):
        super().__init__()
        self.deepstack_visual_indexes = deepstack_visual_indexes


class TestQwen3VLFeatureMaterialization(CustomTestCase):
    @staticmethod
    def _model_config(*, language_only: bool):
        return SimpleNamespace(
            encoder_only=False,
            language_only=language_only,
            tie_word_embeddings=False,
            vision_config=SimpleNamespace(deepstack_visual_indexes=[2, 6, 10]),
            text_config=SimpleNamespace(
                encoder_only=False,
                language_only=language_only,
                tie_word_embeddings=False,
                vocab_size=16,
                hidden_size=8,
                rope_scaling={},
            ),
        )

    @staticmethod
    def _construct_model(*, language_only: bool):
        visual = _FakeVisualModel([2, 6, 10])
        with (
            patch(
                "sglang.srt.models.qwen3_vl.get_pp_group",
                return_value=SimpleNamespace(
                    is_first_rank=True, is_last_rank=True, world_size=1
                ),
            ),
            patch(
                "sglang.srt.models.qwen3_vl.get_mm",
                return_value=SimpleNamespace(mm_enable_dp_encoder=False),
            ),
            patch(
                "sglang.srt.models.qwen3_vl.get_server_args",
                return_value=SimpleNamespace(enable_dp_lm_head=False),
            ),
            patch(
                "sglang.srt.models.qwen3_vl.Qwen3VLMoeVisionModel",
                return_value=visual,
            ) as visual_constructor,
            patch(
                "sglang.srt.models.qwen3_vl.ParallelLMHead",
                return_value=nn.Linear(8, 16, bias=False),
            ),
            patch(
                "sglang.srt.models.qwen3_vl.LogitsProcessor", return_value=nn.Identity()
            ),
            patch("sglang.srt.models.qwen3_vl.Pooler", return_value=nn.Identity()),
        ):
            model = Qwen3VLForConditionalGeneration(
                TestQwen3VLFeatureMaterialization._model_config(
                    language_only=language_only
                ),
                language_model_cls=_FakeLanguageModel,
            )
        return model, visual, visual_constructor

    def test_language_only_does_not_construct_visual_model(self):
        model, _, visual_constructor = self._construct_model(language_only=True)

        visual_constructor.assert_not_called()
        self.assertIsNone(model.visual)
        self.assertEqual(model.deepstack_visual_indexes, [2, 6, 10])

    def test_regular_worker_still_constructs_visual_model(self):
        model, visual, visual_constructor = self._construct_model(language_only=False)

        visual_constructor.assert_called_once()
        self.assertIs(model.visual, visual)

    def test_language_only_rejects_accidental_local_image_execution(self):
        model = SimpleNamespace(visual=None, use_data_parallel=False)
        with self.assertRaisesRegex(RuntimeError, "--language-only"):
            Qwen3VLForConditionalGeneration.get_image_feature(model, [])

    def test_image_features_are_packed_on_the_visual_device(self):
        visual = _RecordingVisual()
        model = SimpleNamespace(visual=visual, use_data_parallel=False)
        items = [
            SimpleNamespace(
                feature=torch.ones(2, 3),
                image_grid_thw=torch.tensor([[1, 1, 2]]),
            ),
            SimpleNamespace(
                feature=torch.ones(1, 3),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
            ),
        ]
        output = Qwen3VLForConditionalGeneration.get_image_feature(model, items)

        self.assertIs(visual.pixel_values, output)
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(output.device, visual.device)
        self.assertEqual(output.dtype, visual.dtype)

    def test_video_features_are_packed_on_the_visual_device(self):
        visual = _RecordingVisual()
        model = SimpleNamespace(visual=visual, use_data_parallel=False)
        items = [
            SimpleNamespace(
                feature=torch.ones(3, 4),
                video_grid_thw=torch.tensor([[1, 1, 3]]),
            ),
            SimpleNamespace(
                feature=torch.ones(2, 4),
                video_grid_thw=torch.tensor([[1, 1, 2]]),
            ),
        ]
        output = Qwen3VLForConditionalGeneration.get_video_feature(model, items)

        self.assertIs(visual.pixel_values, output)
        self.assertEqual(output.shape, (5, 4))
        self.assertEqual(output.device, visual.device)
        self.assertEqual(output.dtype, visual.dtype)
        self.assertTrue(
            torch.equal(visual.grid_thw, torch.tensor([[1, 1, 3], [1, 1, 2]]))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
