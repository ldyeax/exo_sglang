"""CPU-only tests for immutable MXFP4 AVX scale-fold server proof."""

import asyncio
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.entrypoints import http_server
from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.managers import scheduler
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _native_telemetry() -> dict:
    return {
        "schema_version": 1,
        "requested_mode": "lut-v1",
        "configuration_valid": True,
        "architecture_supported": True,
        "execution_mode": "not-executed",
        "n_block": 128,
        "fold_safe_minimum": 2,
        "fold_safe_maximum": 252,
        "lut_identity": "mxfp4-e2m1-bf16-ue8m0-lut-v1",
        "lut_hash_algorithm": "fnv1a64-le",
        "lut_hash": "06d1a83dbf20f545",
        "lut_bytes": 16_384,
        "buffers_constructed": 258,
        "buffers_finalized": 258,
        "buffers_admitted": 258,
        "buffers_rejected": 0,
        "whole_buffer_domain_finalized": True,
        "whole_buffer_domain_admitted": True,
        "scale_bytes_audited": 1_000_000,
        "unsafe_scale_bytes": 0,
        "nan_scale_bytes": 0,
        "invalid_mode_requests": 0,
        "observed_scale_minimum": 118,
        "observed_scale_maximum": 126,
        "decode_dispatch_count": 0,
        "prefill_dispatch_count": 0,
        "real_dispatch_count": 0,
        "scale_fold_dispatch_count": 0,
        "lut_decode_dispatch_count": 0,
        "lut_prefill_dispatch_count": 0,
        "exponent_decode_dispatch_count": 0,
        "exponent_prefill_dispatch_count": 0,
        "fallback_dispatch_count": 0,
        "fallback_decode_dispatch_count": 0,
        "fallback_prefill_dispatch_count": 0,
        "zero_invalid_or_fallback_counts": True,
    }


def _worker_record(
    tp_rank: int,
    *,
    pp_rank: int = 0,
    moe_ep_rank: int | None = None,
    gpu_id: int | None = None,
) -> dict:
    effective_gpu_id = tp_rank if gpu_id is None else gpu_id
    return {
        "pid": 3000 + effective_gpu_id,
        "gpu_id": effective_gpu_id,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": None,
        "moe_ep_rank": tp_rank if moe_ep_rank is None else moe_ep_rank,
        "moe_dp_rank": 0,
        "telemetry": _native_telemetry(),
        "validation_error": None,
    }


def _call_server_info(server_args: ServerArgs, internal_states: list[dict]) -> dict:
    async def fake_internal_state() -> list[dict]:
        return internal_states

    stub_state = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            server_args=server_args,
            get_internal_state=fake_internal_state,
        ),
        scheduler_info={"max_req_input_len": 1024},
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(stub_state)
    try:
        return asyncio.run(http_server.server_info())
    finally:
        http_server._global_state = prior_state


class TestKTMXFP4AVXScaleFoldNativeReadback(CustomTestCase):
    def test_static_admission_does_not_require_startup_dispatch(self):
        extension = SimpleNamespace(mxfp4_avx_scale_fold_telemetry=_native_telemetry)
        with (
            patch.dict(
                os.environ,
                {"KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1"},
                clear=True,
            ),
            patch.object(kt_ep_wrapper, "kt_kernel_ext", extension),
        ):
            telemetry = kt_ep_wrapper.get_kt_mxfp4_avx_scale_fold_telemetry()

        self.assertEqual(telemetry["real_dispatch_count"], 0)
        self.assertTrue(telemetry["whole_buffer_domain_admitted"])
        self.assertEqual(telemetry["n_block"], 128)

    def test_native_schema_domain_and_counter_failures_are_rejected(self):
        mutations = (
            lambda value: value.update(n_block=64),
            lambda value: value.update(buffers_finalized=257),
            lambda value: value.update(unsafe_scale_bytes=1),
            lambda value: value.update(observed_scale_maximum=127),
            lambda value: value.update(fallback_dispatch_count=1),
            lambda value: value.pop("lut_hash"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                telemetry = _native_telemetry()
                mutate(telemetry)
                extension = SimpleNamespace(
                    mxfp4_avx_scale_fold_telemetry=lambda value=telemetry: value
                )
                with (
                    patch.dict(
                        os.environ,
                        {"KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1"},
                        clear=True,
                    ),
                    patch.object(kt_ep_wrapper, "kt_kernel_ext", extension),
                    self.assertRaises(RuntimeError),
                ):
                    kt_ep_wrapper.get_kt_mxfp4_avx_scale_fold_telemetry()


class TestKTMXFP4AVXScaleFoldSchedulerCollection(CustomTestCase):
    def test_failure_is_rank_tagged_and_world_gather_is_preserved(self):
        ps = ParallelState.trivial(
            tp_rank=1,
            tp_size=2,
            moe_ep_rank=1,
            moe_ep_size=2,
            gpu_id=1,
        )
        with patch.object(
            kt_ep_wrapper,
            "get_kt_mxfp4_avx_scale_fold_telemetry",
            side_effect=RuntimeError("native getter unavailable"),
        ):
            failure = scheduler._get_kt_mxfp4_avx_scale_fold_worker_record(ps)
        self.assertEqual(failure["tp_rank"], 1)
        self.assertIsNone(failure["telemetry"])
        self.assertIn("native getter unavailable", failure["validation_error"])

        records = [_worker_record(0), _worker_record(1)]
        group = SimpleNamespace(world_size=2, all_gather_object=lambda _local: records)
        gathered = scheduler._gather_kt_mxfp4_avx_scale_fold_workers(group, records[0])
        self.assertEqual(gathered, records)


class TestKTMXFP4AVXScaleFoldHTTPAggregation(CustomTestCase):
    def setUp(self) -> None:
        self.args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            ep_size=2,
            kt_numa_nodes=[0, 1],
        )
        self.records = [_worker_record(1), _worker_record(0)]

    def test_exact_ep2_static_workers_are_active_and_sorted(self):
        summary = http_server._summarize_kt_mxfp4_avx_scale_fold(
            [{"kt_mxfp4_avx_scale_fold_workers": self.records}],
            "lut-v1",
            self.args,
        )

        self.assertEqual(summary["kt_mxfp4_avx_scale_fold_expected_worker_count"], 2)
        self.assertEqual(summary["kt_mxfp4_avx_scale_fold_active_worker_count"], 2)
        self.assertEqual(summary["kt_mxfp4_avx_scale_fold_invalid_worker_count"], 0)
        self.assertTrue(summary["kt_mxfp4_avx_scale_fold_rank_coverage_valid"])
        self.assertTrue(summary["kt_mxfp4_avx_scale_fold_ep2_topology_valid"])
        self.assertTrue(summary["kt_mxfp4_avx_scale_fold_all_workers_active"])
        self.assertEqual(
            [
                record["tp_rank"]
                for record in summary["kt_mxfp4_avx_scale_fold_worker_telemetry"]
            ],
            [0, 1],
        )

    def test_domain_binary_rank_and_duplicate_fail_closed(self):
        mutations = (
            lambda records: records[0]["telemetry"].update(n_block=64),
            lambda records: records[0]["telemetry"].update(buffers_admitted=257),
            lambda records: records[0]["telemetry"].update(observed_scale_minimum=117),
            lambda records: records[0].update(pid=records[1]["pid"]),
            lambda records: records[0].update(gpu_id=0),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                records = copy.deepcopy(self.records)
                mutate(records)
                summary = http_server._summarize_kt_mxfp4_avx_scale_fold(
                    [{"kt_mxfp4_avx_scale_fold_workers": records}],
                    "lut-v1",
                    self.args,
                )
                self.assertFalse(summary["kt_mxfp4_avx_scale_fold_all_workers_active"])

        duplicate = copy.deepcopy(self.records[0])
        summary = http_server._summarize_kt_mxfp4_avx_scale_fold(
            [
                {
                    "kt_mxfp4_avx_scale_fold_workers": [
                        self.records[0],
                        duplicate,
                    ]
                }
            ],
            "lut-v1",
            self.args,
        )
        self.assertEqual(summary["kt_mxfp4_avx_scale_fold_duplicate_worker_count"], 1)
        self.assertFalse(summary["kt_mxfp4_avx_scale_fold_rank_coverage_valid"])

    def test_exact_pp2_workers_use_pipeline_rank_and_gpu_identity(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=1,
            pp_size=2,
            ep_size=1,
            kt_numa_nodes=[0, 1],
        )
        records = [
            _worker_record(0, pp_rank=1, moe_ep_rank=0, gpu_id=1),
            _worker_record(0, pp_rank=0, moe_ep_rank=0, gpu_id=0),
        ]
        summary = http_server._summarize_kt_mxfp4_avx_scale_fold(
            [{"kt_mxfp4_avx_scale_fold_workers": records}],
            "lut-v1",
            args,
        )

        self.assertEqual(summary["kt_mxfp4_avx_scale_fold_topology"], "pp2-ep1")
        self.assertTrue(
            summary["kt_mxfp4_avx_scale_fold_supported_topology_valid"]
        )
        self.assertFalse(summary["kt_mxfp4_avx_scale_fold_ep2_topology_valid"])
        self.assertTrue(summary["kt_mxfp4_avx_scale_fold_rank_coverage_valid"])
        self.assertTrue(summary["kt_mxfp4_avx_scale_fold_all_workers_active"])

    def test_default_off_and_server_info_export(self):
        off = http_server._summarize_kt_mxfp4_avx_scale_fold(
            [{"unrelated": True}], "off", self.args
        )
        self.assertEqual(off["kt_mxfp4_avx_scale_fold_reporting_worker_count"], 0)
        self.assertFalse(off["kt_mxfp4_avx_scale_fold_all_workers_active"])

        with patch.dict(
            os.environ,
            {"KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1"},
            clear=True,
        ):
            info = _call_server_info(
                self.args,
                [{"kt_mxfp4_avx_scale_fold_workers": self.records}],
            )
        self.assertTrue(info["kt_mxfp4_avx_scale_fold_configured"])
        self.assertEqual(info["kt_mxfp4_avx_scale_fold_requested_mode"], "lut-v1")
        self.assertEqual(info["kt_mxfp4_avx_scale_fold_active_worker_count"], 2)
        self.assertTrue(info["kt_mxfp4_avx_scale_fold_all_workers_active"])


if __name__ == "__main__":
    unittest.main()
