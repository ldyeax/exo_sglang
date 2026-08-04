"""Unit tests for SWA ChunkCache chunk-boundary release semantics."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeSwaAllocator:
    def __init__(self, available: int):
        self.available = available
        self.freed_swa = []

    def free_swa(self, indices):
        freed = indices.detach().cpu().clone()
        self.freed_swa.append(freed)
        self.available += len(freed)

    def swa_available_size(self):
        return self.available


class TestSWAChunkCache(CustomTestCase):
    def test_chunk_stash_reclaims_before_next_chunk_admission(self):
        # Live DSV4 geometry at the failure point: a 1024-token SWA pool has
        # admitted 512 + 256 tokens, leaving 256.  The next chunk must reserve a
        # 256-token allocator page, so it cannot be admitted until the completed
        # prefix releases a page.  Chunk N starts at 512 and window=128, making
        # [0, 256) the page-aligned range that is safe to release.
        allocator = _FakeSwaAllocator(available=256)
        cache = SWAChunkCache.__new__(SWAChunkCache)
        cache.page_size = 256
        cache.sliding_window_size = 128
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(1, 1025, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = allocator

        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
            swa_evict_floor=0,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
            extend_range=SimpleNamespace(start=512, end=768),
        )

        cache.cache_unfinished_req(req, chunked=True)

        self.assertEqual(len(req.prefix_indices), 768)
        self.assertEqual(req.kv.swa_evicted_seqlen, 256)
        self.assertEqual(allocator.swa_available_size(), 512)
        self.assertEqual(len(allocator.freed_swa), 1)
        self.assertTrue(
            torch.equal(
                allocator.freed_swa[0], torch.arange(1, 257, dtype=torch.int64)
            )
        )

        # A parked request can be reconsidered repeatedly.  Re-stashing the same
        # boundary must not free the page twice.
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(allocator.swa_available_size(), 512)
        self.assertEqual(len(allocator.freed_swa), 1)

    def test_non_chunked_stash_leaves_swa_release_to_decode(self):
        allocator = _FakeSwaAllocator(available=256)
        cache = SWAChunkCache.__new__(SWAChunkCache)
        cache.page_size = 256
        cache.sliding_window_size = 128
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(1, 1025, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = allocator
        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
            swa_evict_floor=0,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
            extend_range=SimpleNamespace(start=512, end=768),
        )

        cache.cache_unfinished_req(req, chunked=False)

        self.assertEqual(req.kv.swa_evicted_seqlen, 0)
        self.assertEqual(allocator.freed_swa, [])


if __name__ == "__main__":
    unittest.main()
