import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor import memory_profiler
from sglang.srt.model_executor.memory_profiler import DSv4MemoryCalculator
from sglang.srt.mem_cache import deepseekv4_memory_pool
from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4SingleKVPool


class TestDeepSeekV4PipelineKVPool(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = object.__new__(DeepSeekV4SingleKVPool)
        self.pool.start_layer = 21
        self.pool.kv_buffer = [object(), object()]
        self.pool.use_bf16_cache = True
        self.pool.store_dtype = torch.uint8
        self.pool.dtype = torch.bfloat16
        self.pool._page_size = 256
        self.pool.is_swa_pool = True

    def test_global_layer_ids_map_to_stage_local_buffers(self) -> None:
        self.assertIs(self.pool.get_key_buffer(21), self.pool.kv_buffer[0])
        self.assertIs(self.pool.get_key_buffer(22), self.pool.kv_buffer[1])

        with self.assertRaisesRegex(IndexError, r"stage=\[21, 23\)"):
            self.pool.get_key_buffer(20)
        with self.assertRaisesRegex(IndexError, r"stage=\[21, 23\)"):
            self.pool.get_key_buffer(23)

    def test_setters_use_the_same_stage_local_mapping(self) -> None:
        location = torch.tensor([0], dtype=torch.int32)
        bf16_pack = object()

        with patch.object(
            deepseekv4_memory_pool.index_buf_accessor_v4.SetBf16KAndS,
            "execute",
        ) as set_bf16:
            self.pool.set_key_buffer(
                layer_id=21,
                loc=location,
                cache_nope_fp8_rope_bf16_pack=None,
                cache_bf16_pack=bf16_pack,
            )
        self.assertIs(set_bf16.call_args.kwargs["buf"], self.pool.kv_buffer[0])

        cache_k = torch.empty(1)
        with patch.object(
            deepseekv4_memory_pool,
            "fused_store_cache",
        ) as fused_store:
            self.pool.set_key_buffer_fused(22, location, cache_k)
        self.assertIs(fused_store.call_args.kwargs["cache"], self.pool.kv_buffer[1])

    def test_distributed_memory_profile_reduces_token_capacity(self) -> None:
        calculator = object.__new__(DSv4MemoryCalculator)
        calculator.is_speculative = False
        calculator.bytes_per_full_token = 10.0
        calculator.page_size = 256
        calculator.swa_ratio = 0.1
        calculator.swa_page_size = 128
        calculator.c4_ring_size = 8
        calculator.c128_ring_size = 128
        calculator.c4_shrink_factor = 1
        calculator.use_bf16 = True

        model_runner = SimpleNamespace(
            device="cuda",
            gpu_id=0,
            total_gpu_memory=24.0,
            mem_fraction_static=0.9,
        )
        world_group = SimpleNamespace(world_size=2, cpu_group=object())
        pp_group = SimpleNamespace(world_size=1)

        def reduce_to_remote_capacity(
            token_capacity: torch.Tensor,
            **_: object,
        ) -> None:
            token_capacity.fill_(1024)

        with (
            patch.object(
                memory_profiler,
                "profile_available_bytes",
                return_value=20480,
            ) as profile_available_bytes,
            patch.object(
                memory_profiler,
                "get_world_group",
                return_value=world_group,
            ),
            patch.object(
                memory_profiler,
                "get_pp_group",
                return_value=pp_group,
            ),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=reduce_to_remote_capacity,
            ),
        ):
            pool_sizes = calculator.get_pool_sizes_by_profiling(model_runner)

        self.assertEqual(pool_sizes.full_max_total_num_tokens, 1024)
        self.assertFalse(profile_available_bytes.call_args.kwargs["distributed"])

    def test_pipeline_communicators_are_initialized_on_forward_edges(self) -> None:
        device_group = object()
        pp_group = SimpleNamespace(
            world_size=3,
            rank_in_group=1,
            ranks=[10, 11, 12],
            device_group=device_group,
        )
        warmup = Mock()

        with (
            patch.object(
                memory_profiler,
                "get_pp_group",
                return_value=pp_group,
            ),
            patch.object(torch, "empty", return_value=warmup) as empty,
            patch.object(torch.distributed, "recv") as recv,
            patch.object(torch.distributed, "send") as send,
            patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            memory_profiler._preinitialize_pipeline_communicators("cuda")

        empty.assert_called_once_with(1, dtype=torch.uint8, device="cuda")
        recv.assert_called_once_with(warmup, src=10, group=device_group)
        send.assert_called_once_with(warmup, dst=12, group=device_group)
        synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
