import unittest
from unittest.mock import patch

import torch

# This unit test exercises registration only. Avoid probing a real CUDA device
# while importing SGLang's quantization modules in CPU-only test environments.
with (
    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    patch.object(torch.cuda, "current_device", return_value=0),
):
    from sglang.srt.layers.moe import kt_ep_wrapper
    from sglang.srt.layers.moe import quant_method_registry as registry


class TestMoeQuantMethodRegistry(unittest.TestCase):
    def setUp(self):
        self.original_wrappers = list(registry._QUANT_WRAPPERS)

    def tearDown(self):
        registry._QUANT_WRAPPERS[:] = self.original_wrappers

    def test_registration_query_and_requirement(self):
        registry._QUANT_WRAPPERS.clear()
        registry.register_moe_quant_wrapper(
            "test_wrapper",
            lambda _layer, _server_args: None,
            lambda _layer, gpu_method, _context: gpu_method,
        )

        self.assertTrue(registry.is_moe_quant_wrapper_registered("test_wrapper"))
        registry.require_moe_quant_wrapper_registered("test_wrapper")

    def test_missing_required_wrapper_fails_closed(self):
        registry._QUANT_WRAPPERS.clear()

        with self.assertRaisesRegex(
            RuntimeError, "Required MoE quant-method wrapper 'kt_ep' is not registered"
        ):
            registry.require_moe_quant_wrapper_registered("kt_ep")

    def test_kt_requirement_rejects_missing_runtime(self):
        with patch.object(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "kt_kernel is not installed"):
                kt_ep_wrapper.require_kt_ep_registration()

    def test_kt_requirement_propagates_missing_registration(self):
        with (
            patch.object(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True),
            patch.object(
                kt_ep_wrapper,
                "require_moe_quant_wrapper_registered",
                side_effect=RuntimeError("registration missing"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "registration missing"):
                kt_ep_wrapper.require_kt_ep_registration()

    def test_mask_query_fails_before_initialization(self):
        with patch.object(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None):
            with self.assertRaisesRegex(RuntimeError, "masks are not initialized"):
                kt_ep_wrapper.get_kt_ep_gpu_experts_masks()


if __name__ == "__main__":
    unittest.main()
