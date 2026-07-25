import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

with (
    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    patch.object(torch.cuda, "current_device", return_value=0),
):
    from sglang.srt.batch_overlap import operations
    from sglang.srt.batch_overlap.operations_strategy import (
        OperationsStrategy,
        _compute_moe_deepseek_kt_tbo,
    )
    from sglang.srt.batch_overlap.two_batch_overlap import (
        TboCudaGraphRunnerPlugin,
        _refresh_tbo_child_token_state,
        compute_split_indices_for_cuda_graph_replay,
        compute_split_seq_index,
    )
    from sglang.srt.layers.moe import kt_ep_wrapper as kt
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )


class TestSharedStagingBufferLease(unittest.TestCase):
    def setUp(self):
        self.buffer = kt.SharedStagingBuffer(
            max_tokens=8,
            hidden_size=4,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

    def test_one_generation_can_own_buffer(self):
        first = self.buffer.acquire(3, owner="layer=3,generation=1")
        self.assertEqual(tuple(first.tensor.shape), (3, 4))

        with self.assertRaisesRegex(RuntimeError, "layer=3,generation=1"):
            self.buffer.acquire(2, owner="layer=4,generation=1")
        with self.assertRaisesRegex(RuntimeError, "synchronous reuse is unsafe"):
            self.buffer.get_slice(2)

        first.release()
        second = self.buffer.acquire(2, owner="layer=4,generation=1")
        self.assertGreater(second.generation, first.generation)
        second.release()

    def test_stale_release_does_not_unlock_active_generation(self):
        active = self.buffer.acquire(2, owner="active")
        stale = kt.SharedStagingLease(
            owner=self.buffer,
            token=object(),
            generation=active.generation,
            owner_label="stale",
            tensor=active.tensor,
        )

        with self.assertRaisesRegex(RuntimeError, "does not own"):
            stale.release()
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.buffer.acquire(1, owner="next")

        active.release()
        self.buffer.acquire(1, owner="next").release()

    def test_incompatible_global_spec_fails_closed(self):
        self.buffer.validate_spec(
            max_tokens=8,
            hidden_size=4,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible specification"):
            self.buffer.validate_spec(
                max_tokens=16,
                hidden_size=4,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

    def test_handle_identity_and_generation_are_validated(self):
        wrapper = object.__new__(kt.KTEPWrapperMethod)
        wrapper.kt_config = SimpleNamespace(layer_idx=3)
        wrapper._tbo_apply_in_flight = True
        wrapper._tbo_apply_generation = 4
        handle = kt.KTAsyncApplyHandle(
            owner=wrapper,
            generation=4,
            reference=torch.zeros((1, 2)),
            staging_buffer=None,
            staging_lease=None,
            gpu_output=torch.zeros((1, 2)),
            cpu_submitted=False,
        )
        wrapper._tbo_apply_handle = handle

        wrapper._validate_tbo_handle(handle)

        stale = kt.KTAsyncApplyHandle(
            owner=wrapper,
            generation=3,
            reference=handle.reference,
            staging_buffer=None,
            staging_lease=None,
            gpu_output=handle.gpu_output,
            cpu_submitted=False,
        )
        with self.assertRaisesRegex(RuntimeError, "stale or non-active"):
            wrapper._validate_tbo_handle(stale)

        wrapper._release_tbo_handle(handle)
        self.assertTrue(handle.closed)
        self.assertFalse(wrapper._tbo_apply_in_flight)
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            wrapper._validate_tbo_handle(handle)


class TestTboExceptionCleanup(unittest.TestCase):
    def test_executor_aborts_resource_and_preserves_original_error(self):
        calls = []

        class Resource:
            def abort_on_error(self):
                calls.append("abort")

        resource = Resource()
        forward_batch = SimpleNamespace(
            global_dp_buffer_len=None,
            tbo_padded_len=1,
            global_num_tokens_cpu=None,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        )

        def acquire(state, **_kwargs):
            state.resource = resource

        def fail(state):
            raise ValueError("model failure")

        with (
            patch.object(operations, "set_dp_buffer_len"),
            self.assertRaisesRegex(ValueError, "model failure"),
        ):
            operations.execute_operations(
                {"forward_batch": forward_batch},
                [acquire, fail],
            )

        self.assertEqual(calls, ["abort"])


class TestKtTboOperationSchedule(unittest.TestCase):
    def test_next_cpu_submit_waits_for_previous_sync(self):
        events = []
        active_cpu_job = [None]

        def make_operation(name, layer_index):
            def operation(state, **values):
                child = values["child"]
                if name == "kt_submit":
                    self.assertIsNone(active_cpu_job[0])
                    active_cpu_job[0] = (layer_index, child)
                elif name == "kt_sync":
                    self.assertEqual(active_cpu_job[0], (layer_index, child))
                    active_cpu_job[0] = None
                events.append((layer_index, name, child))
                return values

            operation.__name__ = name
            return operation

        def make_layer(layer_index):
            return SimpleNamespace(
                op_comm_prepare_attn=make_operation(
                    "comm_prepare_attn", layer_index
                ),
                op_comm_prepare_mlp=make_operation(
                    "comm_prepare_mlp", layer_index
                ),
                op_comm_postprocess_layer=make_operation(
                    "comm_postprocess_layer", layer_index
                ),
                self_attn=SimpleNamespace(
                    op_prepare=make_operation("attn_prepare", layer_index),
                    op_core=make_operation("attn_core", layer_index),
                ),
                mlp=SimpleNamespace(
                    op_gate=make_operation("gate", layer_index),
                    op_select_experts=make_operation(
                        "select_experts", layer_index
                    ),
                    op_kt_dispatch=make_operation("kt_dispatch", layer_index),
                    op_kt_submit=make_operation("kt_submit", layer_index),
                    op_shared_experts=make_operation(
                        "shared_experts", layer_index
                    ),
                    op_kt_overlap_window=make_operation(
                        "kt_overlap_window", layer_index
                    ),
                    op_kt_sync=make_operation("kt_sync", layer_index),
                    op_kt_combine=make_operation("kt_combine", layer_index),
                    op_kt_output=make_operation("kt_output", layer_index),
                ),
            )

        strategy = OperationsStrategy.concat(
            [
                _compute_moe_deepseek_kt_tbo(
                    make_layer(layer_index), ForwardMode.DECODE
                )
                for layer_index in range(2)
            ]
        )

        def make_inputs(child):
            return {
                "child": child,
                "forward_batch": SimpleNamespace(
                    global_dp_buffer_len=None,
                    tbo_padded_len=1,
                    global_num_tokens_cpu=None,
                    dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
                ),
            }

        with (
            patch.object(operations, "set_dp_buffer_len"),
            patch.object(
                operations,
                "_resolve_tbo_child_contexts",
                return_value=(None, None),
            ),
        ):
            operations.execute_overlapped_operations(
                inputs_arr=[make_inputs("a"), make_inputs("b")],
                operations_arr=[strategy.operations, strategy.operations],
                delta_stages=[0, strategy.tbo_delta_stages],
            )

        self.assertIsNone(active_cpu_job[0])
        for layer_index in range(2):
            self.assertLess(
                events.index((layer_index, "attn_core", "b")),
                events.index((layer_index, "kt_sync", "a")),
            )
            self.assertLess(
                events.index((layer_index, "kt_sync", "a")),
                events.index((layer_index, "kt_submit", "b")),
            )


class TestTboIndexShareBoundary(unittest.TestCase):
    def _batch(self, topk_indices):
        children = [
            SimpleNamespace(
                tbo_parent_token_range=(0, 2),
                topk_indices=torch.full((2, 3), -1),
            ),
            SimpleNamespace(
                tbo_parent_token_range=(2, 5),
                topk_indices=torch.full((3, 3), -1),
            ),
        ]
        return SimpleNamespace(
            input_ids=torch.arange(5),
            topk_indices=topk_indices,
            tbo_children=children,
        )

    def test_dense_layer_indices_are_sliced_at_tbo_entry(self):
        topk_indices = torch.arange(15, dtype=torch.int32).view(5, 3)
        batch = self._batch(topk_indices)

        _refresh_tbo_child_token_state(batch)

        torch.testing.assert_close(
            batch.tbo_children[0].topk_indices, topk_indices[:2]
        )
        torch.testing.assert_close(
            batch.tbo_children[1].topk_indices, topk_indices[2:]
        )

    def test_mismatched_index_rows_fail_closed(self):
        batch = self._batch(torch.zeros((4, 3), dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, "one row per parent token"):
            _refresh_tbo_child_token_state(batch)

    def test_single_stream_decode_does_not_build_empty_child(self):
        self.assertIsNone(
            compute_split_seq_index(
                forward_mode=ForwardMode.DECODE,
                num_tokens=1,
                extend_lens=None,
                token_num_per_seq=1,
            )
        )
        self.assertEqual(
            compute_split_seq_index(
                forward_mode=ForwardMode.DECODE,
                num_tokens=2,
                extend_lens=None,
                token_num_per_seq=1,
            ),
            1,
        )

    def test_single_stream_cuda_graph_uses_capture_only_placeholder(self):
        self.assertEqual(
            compute_split_indices_for_cuda_graph_replay(
                forward_mode=ForwardMode.DECODE,
                cuda_graph_num_tokens=1,
                spec_info=None,
            ),
            (0, 0),
        )

    def test_cuda_graph_runner_passes_only_supported_replay_arguments(self):
        replay_source = textwrap.dedent(
            inspect.getsource(DecodeCudaGraphRunner.load_batch)
        )
        replay_tree = ast.parse(replay_source)
        plugin_calls = [
            node
            for node in ast.walk(replay_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replay_prepare"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "tbo_plugin"
        ]
        self.assertEqual(len(plugin_calls), 1)

        call_keywords = {
            keyword.arg
            for keyword in plugin_calls[0].keywords
            if keyword.arg is not None
        }
        supported_keywords = set(
            inspect.signature(TboCudaGraphRunnerPlugin.replay_prepare).parameters
        ) - {"self"}
        self.assertEqual(call_keywords, supported_keywords)


if __name__ == "__main__":
    unittest.main()
