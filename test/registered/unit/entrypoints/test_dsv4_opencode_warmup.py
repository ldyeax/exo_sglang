from __future__ import annotations

import asyncio
from typing import Any

from sglang.srt.entrypoints.warmup import (
    DSV4_OPENCODE_WARMUP_INPUT_TOKENS,
    DSV4_OPENCODE_WARMUP_OUTPUT_TOKENS,
    dsv4_opencode_2694,
)


class RecordingTokenizerManager:
    def __init__(self) -> None:
        self.request: Any = None
        self.raw_request: Any = object()
        self.yields = 0

    async def generate_request(self, request: Any, raw_request: Any):
        self.request = request
        self.raw_request = raw_request
        for index in range(3):
            self.yields += 1
            yield {"index": index}


def test_dsv4_opencode_warmup_uses_realistic_shape_and_drains() -> None:
    manager = RecordingTokenizerManager()

    asyncio.run(dsv4_opencode_2694("null", manager))  # type: ignore[arg-type]

    assert len(manager.request.input_ids) == DSV4_OPENCODE_WARMUP_INPUT_TOKENS
    assert len(set(manager.request.input_ids)) > 1000
    assert manager.request.sampling_params == {
        "max_new_tokens": DSV4_OPENCODE_WARMUP_OUTPUT_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    assert manager.raw_request is None
    assert manager.yields == 3


def test_dsv4_opencode_warmup_sets_disaggregation_bootstrap() -> None:
    manager = RecordingTokenizerManager()

    asyncio.run(dsv4_opencode_2694("prefill", manager))  # type: ignore[arg-type]

    assert manager.request.bootstrap_room == 0
    assert manager.request.bootstrap_host
