from __future__ import annotations

import threading

import pytest
import torch

from sglang.srt.layers.moe.kt_stream_prefill import (
    KTStreamPrefillAdmissionError,
    KTStreamPrefillConfig,
    KTStreamPrefillFacts,
    ReusableAsyncRing,
    admit_kt_stream_prefill,
    iter_expert_chunks,
)


def _glm52_facts(**overrides) -> KTStreamPrefillFacts:
    values = {
        "architecture": "GlmMoeDsaForCausalLM",
        "method": "AMXINT4",
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 2,
        "threadpool_count": 2,
        "numa_nodes": (0, 1),
        "num_gpu_experts": 0,
        "max_deferred_experts_per_token": 0,
        "dynamic_expert_update": False,
        "expert_lora_enabled": False,
        "prefill_token_threshold": 4096,
        "num_experts": 256,
        "top_k": 8,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "parameter_dtype": torch.bfloat16,
        "available_device_bytes": 2 * 1024**3,
    }
    values.update(overrides)
    return KTStreamPrefillFacts(**values)


def test_glm52_tp2_four_expert_ring_sizing() -> None:
    plan = admit_kt_stream_prefill(
        KTStreamPrefillConfig(enabled=True, experts_per_chunk=4),
        _glm52_facts(),
    )

    assert plan.per_expert_device_bytes == 36 * 1024**2
    assert plan.device_ring_bytes == 288 * 1024**2
    assert plan.host_ring_bytes_per_rank == 288 * 1024**2
    assert plan.chunks_per_layer == 64


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"pipeline_parallel_size": 2}, "pipeline parallel size must be 1"),
        ({"tensor_parallel_size": 1}, "tensor parallel size must be 2"),
        ({"method": "BF16"}, "KT method must be AMXINT4"),
        ({"num_gpu_experts": 1}, "--kt-num-gpu-experts 0"),
        ({"max_deferred_experts_per_token": 1}, "deferred experts 0"),
        ({"parameter_dtype": torch.float16}, "must be BF16"),
    ],
)
def test_admission_rejects_unproved_runtime_combinations(
    overrides,
    expected: str,
) -> None:
    with pytest.raises(KTStreamPrefillAdmissionError, match=expected):
        admit_kt_stream_prefill(
            KTStreamPrefillConfig(enabled=True),
            _glm52_facts(**overrides),
        )


def test_admission_rejects_ring_that_consumes_vram_margin() -> None:
    with pytest.raises(KTStreamPrefillAdmissionError, match="insufficient free VRAM"):
        admit_kt_stream_prefill(
            KTStreamPrefillConfig(enabled=True, experts_per_chunk=8),
            _glm52_facts(available_device_bytes=1024**3),
        )


def test_expert_chunks_are_contiguous_exact_partition() -> None:
    chunks = tuple(iter_expert_chunks(10, 4))
    assert chunks == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9))
    assert tuple(expert for chunk in chunks for expert in chunk) == tuple(range(10))


def test_reusable_ring_never_overwrites_unconsumed_generation() -> None:
    ring = ReusableAsyncRing[int, tuple[int, int], tuple[int, int]](slots=2)
    state_lock = threading.Lock()
    loaded_generation = [0, 0]
    consumed_generation = [0, 0]
    operations: list[tuple[str, int, int]] = []

    def load(slot: int, generation: int, chunk: int) -> tuple[int, int]:
        with state_lock:
            assert consumed_generation[slot] == loaded_generation[slot]
            loaded_generation[slot] = generation
            operations.append(("load", slot, chunk))
        return slot, chunk

    def compute(ticket) -> tuple[int, int]:
        with state_lock:
            assert loaded_generation[ticket.slot_index] == ticket.generation
            consumed_generation[ticket.slot_index] = ticket.generation
            operations.append(("compute", ticket.slot_index, ticket.chunk))
        return ticket.loaded

    try:
        assert ring.run(tuple(range(6)), load=load, compute=compute) == [
            (0, 0),
            (1, 1),
            (0, 2),
            (1, 3),
            (0, 4),
            (1, 5),
        ]
    finally:
        ring.close()

    for slot in (0, 1):
        assert consumed_generation[slot] == loaded_generation[slot]
    assert [entry for entry in operations if entry[0] == "compute"] == [
        ("compute", 0, 0),
        ("compute", 1, 1),
        ("compute", 0, 2),
        ("compute", 1, 3),
        ("compute", 0, 4),
        ("compute", 1, 5),
    ]


def test_reusable_ring_releases_run_lock_after_loader_failure() -> None:
    ring = ReusableAsyncRing[int, int, int](slots=2)

    def failing_load(slot: int, generation: int, chunk: int) -> int:
        del slot, generation
        if chunk == 1:
            raise RuntimeError("injected load failure")
        return chunk

    try:
        with pytest.raises(RuntimeError, match="injected load failure"):
            ring.run((0, 1, 2), load=failing_load, compute=lambda ticket: ticket.loaded)

        assert ring.run(
            (3, 4),
            load=lambda slot, generation, chunk: chunk,
            compute=lambda ticket: ticket.loaded,
        ) == [3, 4]
    finally:
        ring.close()
