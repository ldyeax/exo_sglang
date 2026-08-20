"""Regressions for native-MTP endpoint sharing and worker compatibility."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagleEarlyWeightSharing(CustomTestCase):
    def test_non_overlap_scheduler_pp_proxy_keyword_is_accepted_for_tp(self):
        signature = inspect.signature(EAGLEWorkerV2.forward_batch_generation)

        bound = signature.bind(
            object(),
            object(),
            pp_proxy_tensors=None,
        )

        self.assertIsNone(bound.arguments["pp_proxy_tensors"])

    def test_real_pp_proxy_tensors_fail_closed(self):
        worker = object.__new__(EAGLEWorkerV2)

        with self.assertRaisesRegex(NotImplementedError, "pipeline-parallel"):
            worker.forward_batch_generation(
                object(),
                pp_proxy_tensors=object(),
            )

    def test_draft_endpoint_initialization_is_idempotent(self):
        worker = object.__new__(EagleDraftWorker)
        worker._shared_embedding_and_lm_head_initialized = False
        worker.device = "cuda"
        worker.gpu_id = 0
        worker.init_token_map = MagicMock()
        worker.init_lm_head = MagicMock()

        device_module = MagicMock()
        with (
            patch("sglang.srt.speculative.eagle_worker_v2.gc.collect") as collect,
            patch(
                "sglang.srt.speculative.eagle_worker_v2.torch.get_device_module",
                return_value=device_module,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.empty_device_cache"
            ) as empty_device_cache,
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_available_gpu_memory",
                return_value=12.5,
            ) as get_available_gpu_memory,
        ):
            worker.initialize_shared_embedding_and_lm_head()
            worker.initialize_shared_embedding_and_lm_head()

        worker.init_token_map.assert_called_once_with()
        worker.init_lm_head.assert_called_once_with()
        collect.assert_called_once_with()
        empty_device_cache.assert_called_once_with(device_module)
        device_module.synchronize.assert_called_once_with()
        get_available_gpu_memory.assert_called_once_with("cuda", 0)
        self.assertTrue(worker._shared_embedding_and_lm_head_initialized)

    def test_eagle_worker_delegates_pre_profile_preparation(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = MagicMock()

        worker.prepare_for_target_memory_pool()

        (
            worker._draft_worker.initialize_shared_embedding_and_lm_head.assert_called_once_with()
        )

    def test_scheduler_releases_draft_weights_before_target_pool_profile(self):
        order = []
        scheduler = object.__new__(Scheduler)
        scheduler.draft_worker = MagicMock()
        scheduler.draft_worker.prepare_for_target_memory_pool.side_effect = (
            lambda: order.append("prepare_draft")
        )
        scheduler.draft_worker.alloc_memory_pool.side_effect = lambda **_: order.append(
            "allocate_draft_pool"
        )
        scheduler.init_target_memory_pool = MagicMock(
            side_effect=lambda: order.append("profile_target_pool")
        )
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(memory_pool_config=object()),
            get_memory_pool=MagicMock(return_value=(object(), object())),
        )

        Scheduler.init_memory_pools(scheduler)

        self.assertEqual(
            order,
            ["prepare_draft", "profile_target_pool", "allocate_draft_pool"],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
