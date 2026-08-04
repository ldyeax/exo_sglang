"""CPU-only tests for fail-closed KT TaskQueue affinity `/server_info` proof."""

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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeCPUInfer:
    def __init__(self, telemetry: dict) -> None:
        self.telemetry = telemetry

    def task_queue_affinity(self) -> dict:
        return copy.deepcopy(self.telemetry)


def _native_affinity(numa_id: int = 0, cpu_id: int = 12) -> dict:
    return {
        "environment_enabled": True,
        "eligible_single_numa_subpool": True,
        "requested": True,
        "active": True,
        "status": "active",
        "numa_id": numa_id,
        "cpu_id": cpu_id,
        "native_thread_id": 4321 + numa_id,
    }


def _worker_record(
    *,
    tp_rank: int,
    pp_rank: int = 0,
    dp_rank: int | None = None,
    moe_ep_rank: int = 0,
    numa_id: int = 0,
) -> dict:
    cpu_id = 12 + numa_id
    return {
        "pid": 1000 + pp_rank * 10 + tp_rank,
        "gpu_id": pp_rank + tp_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": dp_rank,
        "moe_ep_rank": moe_ep_rank,
        "moe_dp_rank": 0,
        "telemetry": {
            **_native_affinity(numa_id=numa_id, cpu_id=cpu_id),
            "expected_numa_id": numa_id,
            "live_cpu_affinity": [cpu_id],
            "cpu_in_expected_numa": True,
            "singleton_instance_count": 1,
            "registered_configuration_count": 1,
        },
        "validation_error": None,
    }


def _call_server_info(
    server_args: ServerArgs,
    internal_states: list[dict],
) -> dict:
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


class TestKTTaskQueueAffinityNativeReadback(CustomTestCase):
    def setUp(self) -> None:
        self.registrations = kt_ep_wrapper._KT_TASK_QUEUE_AFFINITY_REGISTRATIONS
        self.prior_registrations = dict(self.registrations)
        self.registrations.clear()

    def tearDown(self) -> None:
        self.registrations.clear()
        self.registrations.update(self.prior_registrations)

    def test_exact_native_schema_and_live_pin_are_exported(self):
        cpu_infer = _FakeCPUInfer(_native_affinity())
        kt_ep_wrapper._register_kt_task_queue_affinity_instance(
            cpu_infer,
            threadpool_count=1,
            numa_nodes=[0],
        )

        with (
            patch.dict(os.environ, {"KT_TASK_QUEUE_PIN_FIRST_CORE": "1"}),
            patch.object(os, "sched_getaffinity", return_value={12}),
            patch.object(os.path, "exists", return_value=True),
        ):
            telemetry = kt_ep_wrapper.get_kt_task_queue_affinity_telemetry()

        self.assertEqual(telemetry["expected_numa_id"], 0)
        self.assertEqual(telemetry["live_cpu_affinity"], [12])
        self.assertTrue(telemetry["cpu_in_expected_numa"])
        self.assertEqual(telemetry["singleton_instance_count"], 1)

    def test_missing_singleton_getter_schema_or_pin_fails_closed(self):
        failure_cases = []

        failure_cases.append((object(), [0], {12}, True))
        wrong_schema = _native_affinity()
        wrong_schema.pop("active")
        failure_cases.append((_FakeCPUInfer(wrong_schema), [0], {12}, True))
        inactive = _native_affinity()
        inactive["active"] = False
        failure_cases.append((_FakeCPUInfer(inactive), [0], {12}, True))
        wrong_numa = _native_affinity(numa_id=1)
        failure_cases.append((_FakeCPUInfer(wrong_numa), [0], {12}, True))
        failure_cases.append((_FakeCPUInfer(_native_affinity()), [0], {13}, True))
        failure_cases.append((_FakeCPUInfer(_native_affinity()), [0], {12}, False))

        for cpu_infer, numa_nodes, live_affinity, cpu_in_numa in failure_cases:
            with self.subTest(cpu_infer=type(cpu_infer).__name__):
                self.registrations.clear()
                kt_ep_wrapper._register_kt_task_queue_affinity_instance(
                    cpu_infer,
                    threadpool_count=1,
                    numa_nodes=numa_nodes,
                )
                with (
                    patch.dict(
                        os.environ,
                        {"KT_TASK_QUEUE_PIN_FIRST_CORE": "1"},
                    ),
                    patch.object(
                        os,
                        "sched_getaffinity",
                        return_value=live_affinity,
                    ),
                    patch.object(os.path, "exists", return_value=cpu_in_numa),
                    self.assertRaises(RuntimeError),
                ):
                    kt_ep_wrapper.get_kt_task_queue_affinity_telemetry()

    def test_multiple_singletons_or_non_rank_local_config_fails_closed(self):
        first = _FakeCPUInfer(_native_affinity())
        second = _FakeCPUInfer(_native_affinity())
        kt_ep_wrapper._register_kt_task_queue_affinity_instance(
            first,
            threadpool_count=1,
            numa_nodes=[0],
        )
        kt_ep_wrapper._register_kt_task_queue_affinity_instance(
            second,
            threadpool_count=1,
            numa_nodes=[0],
        )
        with (
            patch.dict(os.environ, {"KT_TASK_QUEUE_PIN_FIRST_CORE": "1"}),
            self.assertRaises(RuntimeError),
        ):
            kt_ep_wrapper.get_kt_task_queue_affinity_telemetry()

        self.registrations.clear()
        kt_ep_wrapper._register_kt_task_queue_affinity_instance(
            first,
            threadpool_count=2,
            numa_nodes=[0, 1],
        )
        with (
            patch.dict(os.environ, {"KT_TASK_QUEUE_PIN_FIRST_CORE": "1"}),
            self.assertRaises(RuntimeError),
        ):
            kt_ep_wrapper.get_kt_task_queue_affinity_telemetry()


class TestKTTaskQueueAffinitySchedulerCollection(CustomTestCase):
    def test_local_getter_failure_is_a_rank_tagged_record(self):
        ps = ParallelState.trivial(
            tp_rank=1,
            tp_size=2,
            moe_ep_rank=1,
            moe_ep_size=2,
            gpu_id=1,
        )
        with patch.object(
            kt_ep_wrapper,
            "get_kt_task_queue_affinity_telemetry",
            side_effect=RuntimeError("missing getter"),
        ):
            record = scheduler._get_kt_task_queue_affinity_worker_record(ps)

        self.assertEqual(record["tp_rank"], 1)
        self.assertEqual(record["moe_ep_rank"], 1)
        self.assertIsNone(record["telemetry"])
        self.assertIn("missing getter", record["validation_error"])

    def test_world_gather_preserves_tp_and_pp_records(self):
        records = [
            _worker_record(tp_rank=0, pp_rank=0),
            _worker_record(tp_rank=0, pp_rank=1, numa_id=1),
        ]
        group = SimpleNamespace(
            world_size=2,
            all_gather_object=lambda local: records,
        )

        gathered = scheduler._gather_kt_task_queue_affinity_workers(
            group,
            records[0],
        )

        self.assertEqual(gathered, records)


class TestKTTaskQueueAffinityHTTPAggregation(CustomTestCase):
    def test_ep2_all_workers_are_active_and_sorted(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            ep_size=2,
            kt_numa_nodes=[0, 1],
        )
        records = [
            _worker_record(tp_rank=1, moe_ep_rank=1, numa_id=1),
            _worker_record(tp_rank=0, moe_ep_rank=0, numa_id=0),
        ]

        summary = http_server._summarize_kt_task_queue_affinity(
            [{"kt_task_queue_affinity_workers": records}],
            True,
            args,
        )

        self.assertEqual(summary["kt_task_queue_affinity_expected_worker_count"], 2)
        self.assertEqual(summary["kt_task_queue_affinity_reporting_worker_count"], 2)
        self.assertEqual(summary["kt_task_queue_affinity_active_worker_count"], 2)
        self.assertEqual(summary["kt_task_queue_affinity_invalid_worker_count"], 0)
        self.assertTrue(summary["kt_task_queue_affinity_rank_coverage_valid"])
        self.assertTrue(summary["kt_task_queue_affinity_all_workers_active"])
        self.assertEqual(
            [
                record["tp_rank"]
                for record in summary["kt_task_queue_affinity_worker_telemetry"]
            ],
            [0, 1],
        )

    def test_pp2_uses_pipeline_rank_for_expected_numa(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=1,
            pp_size=2,
            ep_size=1,
            kt_numa_nodes=[0, 1],
        )
        records = [
            _worker_record(tp_rank=0, pp_rank=0, numa_id=0),
            _worker_record(tp_rank=0, pp_rank=1, numa_id=1),
        ]

        summary = http_server._summarize_kt_task_queue_affinity(
            [{"kt_task_queue_affinity_workers": records}],
            True,
            args,
        )

        self.assertEqual(summary["kt_task_queue_affinity_active_worker_count"], 2)
        self.assertTrue(summary["kt_task_queue_affinity_all_workers_active"])

    def test_wrong_numa_missing_getter_and_duplicate_rank_fail_closed(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            ep_size=2,
            kt_numa_nodes=[0, 1],
        )
        good = _worker_record(tp_rank=0, moe_ep_rank=0, numa_id=0)
        wrong_numa = _worker_record(tp_rank=1, moe_ep_rank=1, numa_id=0)
        summary = http_server._summarize_kt_task_queue_affinity(
            [{"kt_task_queue_affinity_workers": [good, wrong_numa]}],
            True,
            args,
        )
        self.assertEqual(summary["kt_task_queue_affinity_active_worker_count"], 1)
        self.assertFalse(summary["kt_task_queue_affinity_all_workers_active"])

        getter_error = _worker_record(tp_rank=1, moe_ep_rank=1, numa_id=1)
        getter_error["telemetry"] = None
        getter_error["validation_error"] = "RuntimeError: missing getter"
        summary = http_server._summarize_kt_task_queue_affinity(
            [{"kt_task_queue_affinity_workers": [good, getter_error]}],
            True,
            args,
        )
        self.assertEqual(summary["kt_task_queue_affinity_invalid_worker_count"], 1)
        self.assertFalse(summary["kt_task_queue_affinity_all_workers_active"])

        duplicate = copy.deepcopy(good)
        summary = http_server._summarize_kt_task_queue_affinity(
            [{"kt_task_queue_affinity_workers": [good, duplicate]}],
            True,
            args,
        )
        self.assertEqual(summary["kt_task_queue_affinity_duplicate_worker_count"], 1)
        self.assertFalse(summary["kt_task_queue_affinity_rank_coverage_valid"])
        self.assertFalse(summary["kt_task_queue_affinity_all_workers_active"])

    def test_default_off_is_empty_and_endpoint_exports_summary(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            ep_size=2,
            kt_numa_nodes=[0, 1],
        )
        off = http_server._summarize_kt_task_queue_affinity(
            [{"unrelated": True}],
            False,
            args,
        )
        self.assertEqual(off["kt_task_queue_affinity_reporting_worker_count"], 0)
        self.assertFalse(off["kt_task_queue_affinity_all_workers_active"])

        records = [
            _worker_record(tp_rank=0, moe_ep_rank=0, numa_id=0),
            _worker_record(tp_rank=1, moe_ep_rank=1, numa_id=1),
        ]
        with patch.dict(
            os.environ,
            {"KT_TASK_QUEUE_PIN_FIRST_CORE": "1"},
            clear=True,
        ):
            info = _call_server_info(
                args,
                [{"kt_task_queue_affinity_workers": records}],
            )

        self.assertTrue(info["kt_task_queue_affinity_configured"])
        self.assertEqual(info["kt_task_queue_affinity_expected_worker_count"], 2)
        self.assertEqual(info["kt_task_queue_affinity_active_worker_count"], 2)
        self.assertTrue(info["kt_task_queue_affinity_all_workers_active"])


if __name__ == "__main__":
    unittest.main()
