"""CPU-only tests for exact single-NUMA inline-dispatch server proof."""

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

_WORKER_COUNT = 56


class _FakeCPUInfer:
    def __init__(
        self,
        *,
        numa_id: int = 0,
        cpu_base: int = 0,
        native_thread_id_base: int = 10000,
        configured_worker_count: int = _WORKER_COUNT,
        exception_count: int = 0,
        advance_dispatch_count: bool = True,
        wrong_worker_zero_role: bool = False,
    ) -> None:
        self.numa_id = numa_id
        self.worker_cpu_ids = list(range(cpu_base, cpu_base + _WORKER_COUNT))
        self.worker_native_thread_ids = list(
            range(native_thread_id_base, native_thread_id_base + _WORKER_COUNT)
        )
        self.configured_worker_count = configured_worker_count
        self.exception_count = exception_count
        self.advance_dispatch_count = advance_dispatch_count
        self.wrong_worker_zero_role = wrong_worker_zero_role
        self.dispatch_count = 0

    def task_queue_affinity(self) -> dict:
        return {
            "environment_enabled": True,
            "eligible_single_numa_subpool": True,
            "requested": True,
            "active": True,
            "status": "active",
            "numa_id": self.numa_id,
            "cpu_id": self.worker_cpu_ids[0],
            "native_thread_id": self.worker_native_thread_ids[0],
        }

    def single_numa_inline_dispatch(self) -> dict:
        has_dispatched = self.dispatch_count > 0
        return {
            "environment_enabled": True,
            "eligible_single_numa_subpool": True,
            "requested": True,
            "active": True,
            "status": "active",
            "physical_numa_id": self.numa_id,
            "configured_worker_count": self.configured_worker_count,
            "distributor_worker_count": 0,
            "distributor_thread_elided": True,
            "dispatch_count": self.dispatch_count,
            "exception_count": self.exception_count,
            "last_native_thread_id": (
                self.worker_native_thread_ids[0] if has_dispatched else -1
            ),
            "last_cpu_id": self.worker_cpu_ids[0] if has_dispatched else -1,
            "last_worker_pool_thread_id": 0 if has_dispatched else -1,
            "task_queue_native_thread_id": self.worker_native_thread_ids[0],
            "task_queue_cpu_id": self.worker_cpu_ids[0],
            "task_queue_affinity_active": True,
            "last_dispatch_on_task_queue_thread": has_dispatched,
            "last_dispatch_on_task_queue_cpu": has_dispatched,
            "logical_worker_zero_proven": has_dispatched,
            "collision_free_worker_zero": True,
        }

    def worker_pool_affinity(self) -> dict:
        roles = ["inline_task_queue_worker0"] + ["background_worker"] * (
            _WORKER_COUNT - 1
        )
        if self.wrong_worker_zero_role:
            roles[0] = "background_worker"
        return {
            "subpool_count": 1,
            "configured_worker_count": self.configured_worker_count,
            "subpools": [
                {
                    "logical_subpool_index": 0,
                    "physical_numa_id": self.numa_id,
                    "configured_worker_count": self.configured_worker_count,
                    "active_worker_count": _WORKER_COUNT,
                    "worker_cpu_ids": list(self.worker_cpu_ids),
                    "worker_native_thread_ids": list(self.worker_native_thread_ids),
                    "worker_affinity_statuses": ["active"] * _WORKER_COUNT,
                    "worker_roles": roles,
                    "last_caller_native_thread_id": self.worker_native_thread_ids[0],
                    "last_caller_cpu_id": self.worker_cpu_ids[0],
                }
            ],
            "all_worker_bindings_active": True,
            "all_worker_cpu_ids_unique": True,
            "all_workers_on_expected_numa": True,
        }

    def probe_single_numa_inline_dispatch(self, task_count: int) -> dict:
        if self.advance_dispatch_count:
            self.dispatch_count += 1
        return {
            "executed_task_count": task_count,
            "worker_native_thread_ids": list(self.worker_native_thread_ids),
            "worker_cpu_ids": list(self.worker_cpu_ids),
        }


def _telemetry(
    *,
    numa_id: int,
    cpu_base: int,
    native_thread_id_base: int,
) -> dict:
    fake = _FakeCPUInfer(
        numa_id=numa_id,
        cpu_base=cpu_base,
        native_thread_id_base=native_thread_id_base,
    )
    fake.dispatch_count = 1
    task_queue = fake.task_queue_affinity()
    inline = fake.single_numa_inline_dispatch()
    worker_pool = fake.worker_pool_affinity()
    startup_probe = {
        "executed_task_count": _WORKER_COUNT,
        "worker_native_thread_ids": list(fake.worker_native_thread_ids),
        "worker_cpu_ids": list(fake.worker_cpu_ids),
    }
    return {
        "environment_enabled": True,
        "expected_numa_id": numa_id,
        "required_worker_count": _WORKER_COUNT,
        "singleton_instance_count": 1,
        "registered_configuration_count": 1,
        "dispatch_count_before_startup_probe": 0,
        "dispatch_count_after_startup_probe": 1,
        "startup_probe_advanced_dispatch_count": True,
        "task_queue_affinity": task_queue,
        "single_numa_inline_dispatch": inline,
        "worker_pool_affinity": worker_pool,
        "startup_probe": startup_probe,
        "task_queue_live_cpu_affinity": [fake.worker_cpu_ids[0]],
        "worker_live_cpu_affinities": [[cpu_id] for cpu_id in fake.worker_cpu_ids],
        "all_live_worker_affinities_exact": True,
        "all_worker_cpus_in_expected_numa": True,
    }


def _worker_record(
    *,
    tp_rank: int,
    numa_id: int,
    pp_rank: int = 0,
    moe_ep_rank: int | None = None,
    gpu_id: int | None = None,
) -> dict:
    effective_gpu_id = tp_rank if gpu_id is None else gpu_id
    return {
        "pid": 2000 + effective_gpu_id,
        "gpu_id": effective_gpu_id,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": None,
        "moe_ep_rank": tp_rank if moe_ep_rank is None else moe_ep_rank,
        "moe_dp_rank": 0,
        "telemetry": _telemetry(
            numa_id=numa_id,
            cpu_base=numa_id * 64,
            native_thread_id_base=10000 + numa_id * 100,
        ),
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


class TestKTSingleNumaInlineDispatchNativeReadback(CustomTestCase):
    def setUp(self) -> None:
        self.registrations = kt_ep_wrapper._KT_TASK_QUEUE_AFFINITY_REGISTRATIONS
        self.prior_registrations = dict(self.registrations)
        self.registrations.clear()

    def tearDown(self) -> None:
        self.registrations.clear()
        self.registrations.update(self.prior_registrations)

    def _register(self, cpu_infer: object) -> None:
        kt_ep_wrapper._register_kt_task_queue_affinity_instance(
            cpu_infer,
            threadpool_count=1,
            numa_nodes=[0],
        )

    def test_startup_probe_exports_exact_56_worker_live_proof(self):
        cpu_infer = _FakeCPUInfer()
        self._register(cpu_infer)
        affinity_by_thread_id = dict(
            zip(
                cpu_infer.worker_native_thread_ids,
                ({cpu_id} for cpu_id in cpu_infer.worker_cpu_ids),
                strict=True,
            )
        )
        with (
            patch.dict(os.environ, {"KT_SINGLE_NUMA_INLINE_DISPATCH": "1"}),
            patch.object(
                os,
                "sched_getaffinity",
                side_effect=lambda native_thread_id: affinity_by_thread_id[
                    native_thread_id
                ],
            ),
            patch.object(os.path, "exists", return_value=True),
        ):
            telemetry = kt_ep_wrapper.get_kt_single_numa_inline_dispatch_telemetry()

        self.assertEqual(telemetry["dispatch_count_before_startup_probe"], 0)
        self.assertEqual(telemetry["dispatch_count_after_startup_probe"], 1)
        self.assertEqual(
            telemetry["startup_probe"]["executed_task_count"], _WORKER_COUNT
        )
        self.assertEqual(
            telemetry["worker_pool_affinity"]["subpools"][0]["worker_roles"][0],
            "inline_task_queue_worker0",
        )
        self.assertTrue(telemetry["all_live_worker_affinities_exact"])

    def test_wrong_count_exception_probe_role_and_live_pin_fail_closed(self):
        failure_cases = (
            (_FakeCPUInfer(configured_worker_count=55), False),
            (_FakeCPUInfer(exception_count=1), False),
            (_FakeCPUInfer(advance_dispatch_count=False), False),
            (_FakeCPUInfer(wrong_worker_zero_role=True), False),
            (_FakeCPUInfer(), True),
        )
        for cpu_infer, break_live_pin in failure_cases:
            with self.subTest(
                count=cpu_infer.configured_worker_count,
                exceptions=cpu_infer.exception_count,
                advance=cpu_infer.advance_dispatch_count,
                role=cpu_infer.wrong_worker_zero_role,
                live_pin=break_live_pin,
            ):
                self.registrations.clear()
                self._register(cpu_infer)
                affinity_by_thread_id = dict(
                    zip(
                        cpu_infer.worker_native_thread_ids,
                        ({cpu_id} for cpu_id in cpu_infer.worker_cpu_ids),
                        strict=True,
                    )
                )
                if break_live_pin:
                    affinity_by_thread_id[cpu_infer.worker_native_thread_ids[-1]] = {
                        cpu_infer.worker_cpu_ids[-1] + 1
                    }
                with (
                    patch.dict(
                        os.environ,
                        {"KT_SINGLE_NUMA_INLINE_DISPATCH": "1"},
                    ),
                    patch.object(
                        os,
                        "sched_getaffinity",
                        side_effect=affinity_by_thread_id.__getitem__,
                    ),
                    patch.object(os.path, "exists", return_value=True),
                    self.assertRaises(RuntimeError),
                ):
                    kt_ep_wrapper.get_kt_single_numa_inline_dispatch_telemetry()


class TestKTSingleNumaInlineDispatchSchedulerCollection(CustomTestCase):
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
            "get_kt_single_numa_inline_dispatch_telemetry",
            side_effect=RuntimeError("native probe unavailable"),
        ):
            failure = scheduler._get_kt_single_numa_inline_dispatch_worker_record(ps)
        self.assertEqual(failure["tp_rank"], 1)
        self.assertIsNone(failure["telemetry"])
        self.assertIn("native probe unavailable", failure["validation_error"])

        records = [
            _worker_record(tp_rank=0, numa_id=0),
            _worker_record(tp_rank=1, numa_id=1),
        ]
        group = SimpleNamespace(
            world_size=2,
            all_gather_object=lambda local: records,
        )
        gathered = scheduler._gather_kt_single_numa_inline_dispatch_workers(
            group,
            records[0],
        )
        self.assertEqual(gathered, records)


class TestKTSingleNumaInlineDispatchHTTPAggregation(CustomTestCase):
    def setUp(self) -> None:
        self.args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            ep_size=2,
            kt_numa_nodes=[0, 1],
        )
        self.records = [
            _worker_record(tp_rank=1, numa_id=1),
            _worker_record(tp_rank=0, numa_id=0),
        ]

    def test_exact_ep2_workers_are_active_and_sorted(self):
        summary = http_server._summarize_kt_single_numa_inline_dispatch(
            [{"kt_single_numa_inline_dispatch_workers": self.records}],
            True,
            self.args,
        )

        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_expected_worker_count"], 2
        )
        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_reporting_worker_count"], 2
        )
        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_active_worker_count"], 2
        )
        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_invalid_worker_count"], 0
        )
        self.assertTrue(summary["kt_single_numa_inline_dispatch_rank_coverage_valid"])
        self.assertTrue(summary["kt_single_numa_inline_dispatch_ep2_topology_valid"])
        self.assertTrue(summary["kt_single_numa_inline_dispatch_all_workers_active"])
        self.assertEqual(
            [
                record["tp_rank"]
                for record in summary["kt_single_numa_inline_dispatch_worker_telemetry"]
            ],
            [0, 1],
        )

    def test_nonzero_dispatch_zero_exceptions_and_exact_roles_are_required(self):
        mutation_paths = (
            (
                "zero dispatch",
                lambda record: record["telemetry"][
                    "single_numa_inline_dispatch"
                ].update(dispatch_count=0),
            ),
            (
                "exception",
                lambda record: record["telemetry"][
                    "single_numa_inline_dispatch"
                ].update(exception_count=1),
            ),
            (
                "distributor",
                lambda record: record["telemetry"][
                    "single_numa_inline_dispatch"
                ].update(distributor_worker_count=1),
            ),
            (
                "worker zero role",
                lambda record: record["telemetry"]["worker_pool_affinity"]["subpools"][
                    0
                ]["worker_roles"].__setitem__(0, "background_worker"),
            ),
        )
        for label, mutate in mutation_paths:
            with self.subTest(label=label):
                records = copy.deepcopy(self.records)
                mutate(records[0])
                summary = http_server._summarize_kt_single_numa_inline_dispatch(
                    [{"kt_single_numa_inline_dispatch_workers": records}],
                    True,
                    self.args,
                )
                self.assertEqual(
                    summary["kt_single_numa_inline_dispatch_active_worker_count"],
                    1,
                )
                self.assertFalse(
                    summary["kt_single_numa_inline_dispatch_all_workers_active"]
                )

    def test_wrong_numa_duplicate_rank_and_pp2_topology(self):
        wrong_numa = copy.deepcopy(self.records)
        wrong_numa[0]["telemetry"]["expected_numa_id"] = 0
        summary = http_server._summarize_kt_single_numa_inline_dispatch(
            [{"kt_single_numa_inline_dispatch_workers": wrong_numa}],
            True,
            self.args,
        )
        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_active_worker_count"], 1
        )
        self.assertFalse(summary["kt_single_numa_inline_dispatch_all_workers_active"])

        duplicate = copy.deepcopy(self.records[0])
        summary = http_server._summarize_kt_single_numa_inline_dispatch(
            [
                {
                    "kt_single_numa_inline_dispatch_workers": [
                        self.records[0],
                        duplicate,
                    ]
                }
            ],
            True,
            self.args,
        )
        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_duplicate_worker_count"], 1
        )
        self.assertFalse(summary["kt_single_numa_inline_dispatch_rank_coverage_valid"])

        pp2_args = ServerArgs(
            model_path="dummy",
            tp_size=1,
            pp_size=2,
            ep_size=1,
            kt_numa_nodes=[0, 1],
        )
        pp2_records = [
            _worker_record(
                tp_rank=0,
                pp_rank=0,
                moe_ep_rank=0,
                gpu_id=0,
                numa_id=0,
            ),
            _worker_record(
                tp_rank=0,
                pp_rank=1,
                moe_ep_rank=0,
                gpu_id=1,
                numa_id=1,
            ),
        ]
        summary = http_server._summarize_kt_single_numa_inline_dispatch(
            [{"kt_single_numa_inline_dispatch_workers": pp2_records}],
            True,
            pp2_args,
        )
        self.assertFalse(summary["kt_single_numa_inline_dispatch_ep2_topology_valid"])
        self.assertEqual(
            summary["kt_single_numa_inline_dispatch_topology"], "pp2-ep1"
        )
        self.assertTrue(
            summary[
                "kt_single_numa_inline_dispatch_supported_topology_valid"
            ]
        )
        self.assertTrue(summary["kt_single_numa_inline_dispatch_all_workers_active"])

    def test_default_off_and_server_info_export(self):
        off = http_server._summarize_kt_single_numa_inline_dispatch(
            [{"unrelated": True}],
            False,
            self.args,
        )
        self.assertEqual(
            off["kt_single_numa_inline_dispatch_reporting_worker_count"], 0
        )
        self.assertFalse(off["kt_single_numa_inline_dispatch_all_workers_active"])

        with patch.dict(
            os.environ,
            {"KT_SINGLE_NUMA_INLINE_DISPATCH": "1"},
            clear=True,
        ):
            info = _call_server_info(
                self.args,
                [{"kt_single_numa_inline_dispatch_workers": self.records}],
            )
        self.assertTrue(info["kt_single_numa_inline_dispatch_configured"])
        self.assertEqual(info["kt_single_numa_inline_dispatch_active_worker_count"], 2)
        self.assertTrue(info["kt_single_numa_inline_dispatch_all_workers_active"])


if __name__ == "__main__":
    unittest.main()
