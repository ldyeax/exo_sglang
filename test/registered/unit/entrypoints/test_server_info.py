"""Endpoint-level tests for `/server_info`.

`/server_info` is the introspection surface that external consumers
(SGLang's own deprecated `/get_server_info` alias, monitoring tools,
KV-aware routers) scrape to learn about the running server's
configuration. New `/server_info` behaviours should add their test
classes to this file as the surface grows.

Current coverage:

* `TestServerInfoKvEventsField` — the `kv_events` publisher descriptor
  surfaced by `_build_kv_events_block`. Covers the full input matrix
  end-to-end (happy path / disabled / malformed JSON / inproc endpoint /
  port edge cases / missing-or-non-positive page_size) because the
  helper has no separate test target; the handler is its only caller.

* `TestServerInfoExistingFieldsPreserved` — regression guard that no
  field existing consumers depend on is silently dropped: every
  `ServerArgs` dataclass field, `internal_states`, `version`, and the
  pre-existing flat `kv_events_config` string all remain visible.
"""

import asyncio
import dataclasses
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _call_server_info_with(
    server_args: ServerArgs,
    internal_states: list[dict] | None = None,
    scheduler_info: dict | None = None,
) -> dict:
    """Invoke `http_server.server_info()` against a stub global state.

    Bypasses the FastAPI HTTP layer (no TestClient): the handler is an
    `async def` that reads module-level `_global_state`, so wiring a
    `SimpleNamespace` stub via `set_global_state` and awaiting the
    coroutine directly is enough to exercise the handler logic without
    booting a model server.
    """

    async def _fake_internal_state():
        return internal_states or [{"max_req_input_len": 1024}]

    stub_state = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            server_args=server_args,
            get_internal_state=_fake_internal_state,
        ),
        scheduler_info=(
            {"max_req_input_len": 1024} if scheduler_info is None else scheduler_info
        ),
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(stub_state)
    try:
        return asyncio.run(http_server.server_info())
    finally:
        # Restore so a later test in the same process isn't surprised.
        http_server._global_state = prior_state


class TestServerInfoKvEventsField(CustomTestCase):
    """The new `kv_events` field is wired correctly across the full
    `_build_kv_events_block` input matrix.
    """

    # ----- happy path --------------------------------------------------

    def test_kv_events_key_present_when_publishing_enabled(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": "kv"}'
            ),
            page_size=64,
            dp_size=2,
        )

        info = _call_server_info_with(args)

        self.assertIn("kv_events", info)
        self.assertEqual(
            info["kv_events"],
            {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 2,
            },
        )

    def test_kv_events_descriptor_carries_specific_host_and_topic(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://0.0.0.0:7777", "topic": "kv"}'
            ),
            page_size=128,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIsNotNone(info["kv_events"])
        self.assertEqual(info["kv_events"]["endpoint_host"], "0.0.0.0")
        self.assertEqual(info["kv_events"]["endpoint_port_base"], 7777)
        self.assertEqual(info["kv_events"]["topic"], "kv")
        self.assertEqual(info["kv_events"]["block_size"], 128)
        self.assertEqual(info["kv_events"]["dp_size"], 1)

    # ----- disabled / unconfigured -------------------------------------

    def test_kv_events_is_null_when_no_publisher_configured(self):
        args = ServerArgs(model_path="dummy")  # no --kv-events-config

        info = _call_server_info_with(args)

        # The key must still be present so consumers can detect
        # "publishing disabled" via a single shape check.
        self.assertIn("kv_events", info)
        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_when_publisher_explicitly_null(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "null"}',
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    # ----- malformed config --------------------------------------------

    def test_kv_events_is_null_for_malformed_json(self):
        # Not JSON — the publisher would have failed at server startup,
        # but /server_info must keep working.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config="not-json",
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    # ----- unreachable endpoints ---------------------------------------

    def test_kv_events_is_null_for_inproc_endpoint(self):
        # `inproc://` is not reachable across process boundaries, so the
        # descriptor must hide it from external routers.
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "inproc://cache", "topic": ""}'
            ),
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_when_endpoint_missing_port(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://0.0.0.0", "topic": ""}'
            ),
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_when_port_not_integer(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=(
                '{"publisher": "zmq", "endpoint": "tcp://0.0.0.0:abc", "topic": ""}'
            ),
            page_size=64,
        )

        info = _call_server_info_with(args)

        self.assertIsNone(info["kv_events"])

    def test_kv_events_is_null_for_port_out_of_range(self):
        # TCP ports are 1..65535; values outside the range can't bind, so
        # the descriptor refuses to advertise them rather than handing
        # subscribers a non-dialable address.
        for bad_port in (0, -1, 65536, 1_000_000):
            with self.subTest(port=bad_port):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=(
                        f'{{"publisher": "zmq", "endpoint": "tcp://0.0.0.0:{bad_port}", "topic": ""}}'
                    ),
                    page_size=64,
                )
                info = _call_server_info_with(args)
                self.assertIsNone(info["kv_events"])

    # ----- bad scheduler context ---------------------------------------

    def test_kv_events_is_null_when_page_size_missing_or_non_positive(self):
        # Without a real positive `page_size` the descriptor's
        # `block_size` would be a misleading placeholder; subscribers
        # would hash prompts at the wrong granularity and miss every
        # cache entry. Refuse to advertise instead.
        good_cfg = '{"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": ""}'
        for bad_page_size in (None, 0, -1):
            with self.subTest(page_size=bad_page_size):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=good_cfg,
                    page_size=bad_page_size,
                )
                info = _call_server_info_with(args)
                self.assertIsNone(info["kv_events"])


class TestServerInfoExistingFieldsPreserved(CustomTestCase):
    """Regression guard: the new `kv_events` field is additive — none of
    the fields existing consumers depend on may be silently dropped.

    Existing `/server_info` consumers in the wild include:
      * SGLang's own deprecated `/get_server_info` (forwards to the
        same handler).
      * External monitoring tools that scrape the full ServerArgs.
      * KV-aware routers reading `kv_events_config`, `page_size`,
        `dp_size` directly to derive subscription info (this is the
        path the new `kv_events` block enriches but does not replace).
    """

    def test_every_server_args_field_appears_in_response(self):
        # `dataclasses.asdict(server_args)` is spread into the response;
        # asserting every dataclass field surfaces is the strongest
        # backward-compat guarantee that's still implementation-agnostic.
        args = ServerArgs(model_path="dummy")

        info = _call_server_info_with(args)

        for field in dataclasses.fields(ServerArgs):
            self.assertIn(
                field.name,
                info,
                f"existing ServerArgs field '{field.name}' missing from "
                f"/server_info response — kv_events patch must not "
                f"shadow or drop ServerArgs fields",
            )

    def test_internal_states_and_version_keys_preserved(self):
        # These two top-level keys predate the kv_events patch and are
        # named individually (not spread from a dataclass), so a stray
        # edit could remove them without breaking syntax. Lock them down.
        args = ServerArgs(model_path="dummy")

        info = _call_server_info_with(args)

        self.assertIn("internal_states", info)
        self.assertIn("version", info)

    def test_kv_events_config_raw_field_still_surfaced(self):
        # The new structured `kv_events` block sits alongside the
        # pre-existing flat `kv_events_config` field (the raw CLI string
        # already on ServerArgs). Both must remain visible so
        # consumers that hand-parse the raw config keep working.
        raw_cfg = '{"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": ""}'
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=raw_cfg,
            page_size=64,
            dp_size=1,
        )

        info = _call_server_info_with(args)

        self.assertIn("kv_events_config", info)
        self.assertEqual(info["kv_events_config"], raw_cfg)
        # And the new structured block is separately present:
        self.assertIn("kv_events", info)
        self.assertIsNotNone(info["kv_events"])

    def test_lora_refs_are_json_serializable_dicts(self):
        lora_ref = LoRARef(
            lora_id="lora-id",
            lora_name="adapter",
            lora_path="/tmp/adapter",
            pinned=True,
        )
        args = ServerArgs(model_path="dummy")
        args.lora_paths = [lora_ref]

        info = _call_server_info_with(
            args,
            internal_states=[{"lora_paths": [lora_ref]}],
        )

        expected = {
            "lora_id": "lora-id",
            "lora_name": "adapter",
            "lora_path": "/tmp/adapter",
            "pinned": True,
        }
        self.assertEqual(info["lora_paths"], [expected])
        self.assertEqual(info["internal_states"][0]["lora_paths"], [expected])
        json.dumps(info)


class TestServerInfoDsv4Sm86SmallBatchTelemetry(CustomTestCase):
    @staticmethod
    def _worker_telemetry(
        tp_rank: int, selection_count: int, *, fused_t5: bool = False
    ) -> dict:
        return {
            "dsv4_sm86_small_batch_gemm": {
                "configured": True,
                "patch_state": "installed",
                "patch_installed": True,
                "patch_error": None,
                "selection_count": selection_count,
                "fused_t5_moe_configured": fused_t5,
                "fused_t5_moe_conversion_count": 43 if fused_t5 else 0,
                "fused_t5_moe_apply_count": 6 if fused_t5 else 0,
                "fused_t5_kt_routing_apply_count": 6 if fused_t5 else 0,
                "observed_signatures": [],
                "selected_config": {
                    "block_n": 128,
                    "split_k": 2,
                    "num_stages": 4,
                    "num_warps": 4,
                },
                "pid": 1000 + tp_rank,
                "gpu_id": tp_rank,
                "tp_rank": tp_rank,
                "pp_rank": 0,
                "dp_rank": 0,
            }
        }

    def test_two_rank_selection_is_proven_and_sorted(self):
        args = ServerArgs(model_path="dummy", tp_size=2)
        states = [
            {
                "dsv4_sm86_small_batch_gemm_workers": [
                    self._worker_telemetry(1, 7)["dsv4_sm86_small_batch_gemm"],
                    self._worker_telemetry(0, 5)["dsv4_sm86_small_batch_gemm"],
                ]
            }
        ]

        with patch.dict(
            os.environ,
            {"SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM": "1"},
        ):
            info = _call_server_info_with(args, internal_states=states)

        self.assertTrue(info["dsv4_sm86_small_batch_gemm_configured"])
        self.assertEqual(info["dsv4_sm86_small_batch_gemm_expected_worker_count"], 2)
        self.assertEqual(info["dsv4_sm86_small_batch_gemm_reporting_worker_count"], 2)
        self.assertEqual(info["dsv4_sm86_small_batch_gemm_active_worker_count"], 2)
        self.assertTrue(info["dsv4_sm86_small_batch_gemm_all_workers_active"])
        self.assertEqual(
            [
                worker["tp_rank"]
                for worker in info["dsv4_sm86_small_batch_gemm_worker_telemetry"]
            ],
            [0, 1],
        )

    def test_missing_or_unselected_rank_fails_activation_proof(self):
        args = ServerArgs(model_path="dummy", tp_size=2)
        states = [
            {
                "dsv4_sm86_small_batch_gemm_workers": [
                    self._worker_telemetry(0, 5)["dsv4_sm86_small_batch_gemm"],
                    self._worker_telemetry(1, 0)["dsv4_sm86_small_batch_gemm"],
                ]
            }
        ]

        with patch.dict(
            os.environ,
            {"SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM": "1"},
        ):
            info = _call_server_info_with(args, internal_states=states)

        self.assertEqual(info["dsv4_sm86_small_batch_gemm_active_worker_count"], 1)
        self.assertFalse(info["dsv4_sm86_small_batch_gemm_all_workers_active"])

        with patch.dict(
            os.environ,
            {"SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM": "1"},
        ):
            missing = _call_server_info_with(
                args,
                internal_states=[
                    {
                        "dsv4_sm86_small_batch_gemm_workers": [
                            self._worker_telemetry(0, 5)["dsv4_sm86_small_batch_gemm"]
                        ]
                    }
                ],
            )
        self.assertEqual(
            missing["dsv4_sm86_small_batch_gemm_reporting_worker_count"], 1
        )
        self.assertFalse(missing["dsv4_sm86_small_batch_gemm_all_workers_active"])

    def test_fused_t5_requires_intent_and_rank_local_use(self):
        args = ServerArgs(model_path="dummy", tp_size=2)
        states = [
            {
                "dsv4_sm86_small_batch_gemm_workers": [
                    self._worker_telemetry(0, 5, fused_t5=True)[
                        "dsv4_sm86_small_batch_gemm"
                    ],
                    self._worker_telemetry(1, 7, fused_t5=True)[
                        "dsv4_sm86_small_batch_gemm"
                    ],
                ]
            }
        ]
        with patch.dict(
            os.environ,
            {
                "SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM": "1",
                "SGLANG_V4_MXFP4_FUSED_T5_MOE": "1",
                "SGLANG_DSV4_OSCAR_FUSED_C4_PIPELINE": "1",
            },
        ):
            info = _call_server_info_with(args, internal_states=states)

        self.assertTrue(info["dsv4_fused_t5_moe_configured"])
        self.assertEqual(info["dsv4_fused_t5_moe_active_worker_count"], 2)
        self.assertTrue(info["dsv4_fused_t5_moe_all_workers_active"])
        self.assertTrue(info["dsv4_oscar_fused_c4_pipeline_configured"])


class TestServerInfoDsv4OscarKernelTelemetry(CustomTestCase):
    def test_scheduler_oscar_kernel_contract_reaches_endpoint_unchanged(self):
        args = ServerArgs(model_path="dummy")
        scheduler_info = {
            "max_req_input_len": 524_287,
            "dsv4_oscar_int2_kv_storage": True,
            "dsv4_oscar_algorithm": "oscar-int2-asym-g64-v1",
            "dsv4_oscar_masked_writer_execution": ("device-uniform-live-mask-row-v1"),
            "dsv4_oscar_wo_a_absorption_state": {
                "enabled": True,
                "consumer_role": "target_compressed",
                "target_only": True,
                "applied": True,
                "apply_count": 1,
                "artifact_sha256": "a" * 64,
                "admission_sha256": "b" * 64,
                "expected_local_compressed_layer_ids": [2],
                "absorbed_local_layer_ids": [2],
                "runtime_restore_skipped_layer_ids": [2],
                "all_local_target_compressed_layers_absorbed": True,
                "all_local_target_compressed_layers_skip_runtime_restore": True,
                "weight_dtype": "bfloat16",
                "head_layout": "per-head-nope448-rope64",
                "fold_orientation": "wo_a_nope@rotation",
                "rope_columns_unchanged": True,
            },
            "dsv4_oscar_c4_scorer": True,
            "dsv4_oscar_c4_scorer_algorithm": (
                "oscar-int2-c4-asym-c128-fp32-adjacent4-v1"
            ),
            "dsv4_oscar_c4_masked_writer_execution": (
                "device-uniform-live-mask-row-v1"
            ),
            "dsv4_oscar_c4_query_rotation_execution": (
                "once-per-query-stable-workspace-v1"
            ),
        }

        info = _call_server_info_with(args, scheduler_info=scheduler_info)

        for key, value in scheduler_info.items():
            self.assertEqual(info[key], value)
        json.dumps(info)


if __name__ == "__main__":
    unittest.main()
