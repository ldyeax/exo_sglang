# SPDX-License-Identifier: Apache-2.0
"""Bounded stream-loading prefill for KTransformers AMXINT4 experts.

KTransformers' existing ``kt_gpu_prefill_token_threshold`` path constructs one
complete temporary GPU expert layer, then synchronously writes every expert
into it before compute starts.  That is useful on GPUs with enough spare
memory, but it is not the sub-layer stream-loading pipeline described by Wang
et al. (OSDI'26):

* the complete AMXINT4 expert allocation remains authoritative in host DRAM;
* two fixed GPU expert chunks form a reusable ring;
* a loader thread dequantizes the next chunk into pinned shared memory and
  enqueues H2D on a transfer stream;
* the model stream waits on a per-slot ready event, computes a weighted partial
  MoE result, and records a consumed event before the slot is reused.

This module intentionally admits only the first hardware/model configuration
we can reason about exactly: GLM-5.2, PP=1/TP=2 on one host, two KT NUMA pools,
AMXINT4 CPU experts, no resident/deferred experts, and BF16 temporary GPU
weights.  Unsupported combinations fail before ring allocation instead of
silently falling back to a numerically or topologically different path.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

import torch
import torch.distributed as dist
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.kt_ep_wrapper import (
        KTEPWrapperMethod,
        SharedFullContext,
    )
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)

_GLM52_ARCHITECTURES = frozenset(
    {
        "GlmMoeDsaForCausalLM",
        "GlmMoeDsaForConditionalGeneration",
    }
)
_BF16_BYTES = 2
_DEFAULT_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024


class KTStreamPrefillAdmissionError(ValueError):
    """Raised when stream-prefill's exact execution contract is not met."""


class _KTWeightExporter(Protocol):
    def submit_write_weight_scale_to_buffer(
        self,
        gpu_tp_count: int,
        expert_id: int,
        w13_weight_ptrs: Sequence[int],
        w13_scale_ptrs: Sequence[int],
        w2_weight_ptrs: Sequence[int],
        w2_scale_ptrs: Sequence[int],
    ) -> None: ...

    def sync_write_weight_scale_to_buffer(self) -> None: ...


@dataclass(frozen=True)
class KTStreamPrefillConfig:
    """User-controlled bounded-ring settings."""

    enabled: bool = False
    experts_per_chunk: int = 4
    ring_slots: int = 2
    safety_margin_bytes: int = _DEFAULT_SAFETY_MARGIN_BYTES


@dataclass(frozen=True)
class KTStreamPrefillFacts:
    """Runtime facts required for fail-closed admission."""

    architecture: str
    method: str
    pipeline_parallel_size: int
    tensor_parallel_size: int
    threadpool_count: int
    numa_nodes: tuple[int, ...]
    num_gpu_experts: int
    max_deferred_experts_per_token: int
    dynamic_expert_update: bool
    expert_lora_enabled: bool
    prefill_token_threshold: int
    num_experts: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: int
    parameter_dtype: torch.dtype
    available_device_bytes: int


@dataclass(frozen=True)
class KTStreamPrefillPlan:
    """Admitted immutable execution and memory plan."""

    experts_per_chunk: int
    ring_slots: int
    num_experts: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: int
    tensor_parallel_size: int
    per_expert_device_bytes: int
    device_ring_bytes: int
    host_ring_bytes_per_rank: int
    safety_margin_bytes: int

    @property
    def chunks_per_layer(self) -> int:
        return self.num_experts // self.experts_per_chunk


def admit_kt_stream_prefill(
    config: KTStreamPrefillConfig,
    facts: KTStreamPrefillFacts,
) -> KTStreamPrefillPlan:
    """Validate the prototype contract and return exact ring sizing."""

    if not config.enabled:
        raise KTStreamPrefillAdmissionError("KT stream-prefill is not enabled")

    failures: list[str] = []
    if facts.architecture not in _GLM52_ARCHITECTURES:
        failures.append(
            f"architecture must be GLM-5.2 ({sorted(_GLM52_ARCHITECTURES)}), "
            f"got {facts.architecture!r}"
        )
    if facts.method.upper() != "AMXINT4":
        failures.append(f"KT method must be AMXINT4, got {facts.method!r}")
    if facts.pipeline_parallel_size != 1:
        failures.append(
            f"pipeline parallel size must be 1, got {facts.pipeline_parallel_size}"
        )
    if facts.tensor_parallel_size != 2:
        failures.append(
            f"tensor parallel size must be 2, got {facts.tensor_parallel_size}"
        )
    if facts.threadpool_count != facts.tensor_parallel_size:
        failures.append(
            "KT threadpool count must equal tensor parallel size "
            f"({facts.tensor_parallel_size}), got {facts.threadpool_count}"
        )
    if len(facts.numa_nodes) != facts.threadpool_count:
        failures.append(
            "one explicit NUMA node is required per KT threadpool, got "
            f"{facts.numa_nodes!r}"
        )
    if len(set(facts.numa_nodes)) != len(facts.numa_nodes):
        failures.append(f"KT NUMA nodes must be distinct, got {facts.numa_nodes!r}")
    if facts.num_gpu_experts != 0:
        failures.append(
            "stream-prefill currently requires --kt-num-gpu-experts 0, got "
            f"{facts.num_gpu_experts}"
        )
    if facts.max_deferred_experts_per_token != 0:
        failures.append(
            "stream-prefill requires exact routing with deferred experts 0, got "
            f"{facts.max_deferred_experts_per_token}"
        )
    if facts.dynamic_expert_update:
        failures.append("dynamic expert update is incompatible with the reusable ring")
    if facts.expert_lora_enabled:
        failures.append("KT expert LoRA is not admitted by the first stream-prefill path")
    if facts.prefill_token_threshold <= 0:
        failures.append(
            "--kt-gpu-prefill-token-threshold must be positive when "
            "--kt-stream-prefill is enabled"
        )
    if facts.parameter_dtype != torch.bfloat16:
        failures.append(
            f"temporary GPU experts must be BF16, got {facts.parameter_dtype}"
        )
    if facts.num_experts != 256:
        failures.append(f"GLM-5.2 must expose 256 routed experts, got {facts.num_experts}")
    if facts.top_k != 8:
        failures.append(f"GLM-5.2 must route top-8 experts, got top-{facts.top_k}")
    if facts.hidden_size != 6144:
        failures.append(f"GLM-5.2 hidden size must be 6144, got {facts.hidden_size}")
    if facts.moe_intermediate_size != 2048:
        failures.append(
            "GLM-5.2 routed intermediate size must be 2048, got "
            f"{facts.moe_intermediate_size}"
        )
    if config.ring_slots != 2:
        failures.append(
            f"the first event scheduler requires exactly 2 ring slots, got {config.ring_slots}"
        )
    if config.experts_per_chunk <= 0:
        failures.append("experts per chunk must be positive")
    elif config.experts_per_chunk > 16:
        failures.append(
            "experts per chunk must not exceed 16 on a 24 GiB RTX 3090"
        )
    elif config.experts_per_chunk & (config.experts_per_chunk - 1):
        failures.append("experts per chunk must be a power of two")
    elif facts.num_experts % config.experts_per_chunk != 0:
        failures.append(
            f"{facts.num_experts} experts must divide evenly into "
            f"chunks of {config.experts_per_chunk}"
        )
    if config.safety_margin_bytes < 0:
        failures.append("stream-prefill safety margin must not be negative")

    if failures:
        raise KTStreamPrefillAdmissionError(
            "KT stream-prefill admission failed: " + "; ".join(failures)
        )

    intermediate_per_rank = (
        facts.moe_intermediate_size // facts.tensor_parallel_size
    )
    if intermediate_per_rank * facts.tensor_parallel_size != facts.moe_intermediate_size:
        raise KTStreamPrefillAdmissionError(
            "KT stream-prefill requires an even intermediate-size TP partition"
        )

    # Per rank: gate [I/TP,H] + up [I/TP,H] + down [H,I/TP].
    per_expert_device_bytes = (
        3
        * facts.hidden_size
        * intermediate_per_rank
        * _BF16_BYTES
    )
    device_ring_bytes = (
        per_expert_device_bytes
        * config.experts_per_chunk
        * config.ring_slots
    )
    required_device_bytes = device_ring_bytes + config.safety_margin_bytes
    if facts.available_device_bytes < required_device_bytes:
        raise KTStreamPrefillAdmissionError(
            "insufficient free VRAM for KT stream-prefill ring: "
            f"available={facts.available_device_bytes} bytes, "
            f"ring={device_ring_bytes} bytes, "
            f"safety_margin={config.safety_margin_bytes} bytes"
        )

    return KTStreamPrefillPlan(
        experts_per_chunk=config.experts_per_chunk,
        ring_slots=config.ring_slots,
        num_experts=facts.num_experts,
        top_k=facts.top_k,
        hidden_size=facts.hidden_size,
        moe_intermediate_size=facts.moe_intermediate_size,
        tensor_parallel_size=facts.tensor_parallel_size,
        per_expert_device_bytes=per_expert_device_bytes,
        device_ring_bytes=device_ring_bytes,
        # Each rank owns one pinned host chunk for each GPU ring slot.  Rank
        # zero writes both ranks' POSIX-shared chunks before publishing them.
        host_ring_bytes_per_rank=device_ring_bytes,
        safety_margin_bytes=config.safety_margin_bytes,
    )


def collectively_admit_kt_stream_prefill(
    config: KTStreamPrefillConfig,
    facts: KTStreamPrefillFacts,
) -> KTStreamPrefillPlan:
    """Make local admission failure visible before any rank allocates a ring."""

    local_plan: Optional[KTStreamPrefillPlan] = None
    local_error: Optional[str] = None
    try:
        local_plan = admit_kt_stream_prefill(config, facts)
    except KTStreamPrefillAdmissionError as error:
        local_error = str(error)

    if dist.is_initialized() and get_tensor_model_parallel_world_size() > 1:
        errors: list[Optional[str]] = [
            None
            for _ in range(get_tensor_model_parallel_world_size())
        ]
        dist.all_gather_object(
            errors,
            local_error,
            group=get_tp_group().cpu_group,
        )
        failures = [
            f"TP rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if failures:
            raise KTStreamPrefillAdmissionError(
                "collective KT stream-prefill admission failed: "
                + " | ".join(failures)
            )
    elif local_error is not None:
        raise KTStreamPrefillAdmissionError(local_error)

    if local_plan is None:
        raise AssertionError("collective admission succeeded without a local plan")
    return local_plan


ChunkT = TypeVar("ChunkT")
LoadedT = TypeVar("LoadedT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class RingTicket(Generic[ChunkT, LoadedT]):
    """One generation-bound load result."""

    slot_index: int
    generation: int
    chunk: ChunkT
    loaded: LoadedT


class ReusableAsyncRing(Generic[ChunkT, LoadedT, OutputT]):
    """Single-loader reusable ring with explicit load/consume ownership.

    ``load`` must wait for the selected slot's previous consumed event before
    overwriting it, and must record a ready event before returning.
    ``compute`` must wait for that ready event and record the new consumed
    event before returning.  These callback contracts keep the generic
    scheduler independent of CUDA and make the ordering unit-testable.
    """

    def __init__(self, slots: int):
        if slots < 2:
            raise ValueError("a reusable async ring needs at least two slots")
        self._slots = slots
        self._loader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="kt-stream-prefill-loader",
        )
        self._closed = False
        self._run_lock = threading.Lock()

    def run(
        self,
        chunks: Sequence[ChunkT],
        *,
        load: Callable[[int, int, ChunkT], LoadedT],
        compute: Callable[[RingTicket[ChunkT, LoadedT]], OutputT],
    ) -> list[OutputT]:
        if self._closed:
            raise RuntimeError("KT stream-prefill ring is closed")
        if not chunks:
            return []
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("KT stream-prefill ring already has an active run")

        futures: dict[int, Future[LoadedT]] = {}
        generations = [0] * self._slots

        def submit(chunk_index: int) -> None:
            slot_index = chunk_index % self._slots
            generations[slot_index] += 1
            generation = generations[slot_index]
            futures[chunk_index] = self._loader.submit(
                load,
                slot_index,
                generation,
                chunks[chunk_index],
            )

        try:
            for chunk_index in range(min(self._slots, len(chunks))):
                submit(chunk_index)

            outputs: list[OutputT] = []
            for chunk_index, chunk in enumerate(chunks):
                slot_index = chunk_index % self._slots
                loaded = futures.pop(chunk_index).result()
                ticket = RingTicket(
                    slot_index=slot_index,
                    generation=generations[slot_index],
                    chunk=chunk,
                    loaded=loaded,
                )
                outputs.append(compute(ticket))

                next_index = chunk_index + self._slots
                if next_index < len(chunks):
                    # compute() has recorded the reuse fence before returning.
                    submit(next_index)
            return outputs
        except BaseException:
            for future in futures.values():
                future.cancel()
            # A running loader cannot be cancelled.  Drain it so a later
            # request cannot observe a half-published host/GPU slot.
            for future in futures.values():
                if not future.cancelled():
                    with contextlib.suppress(BaseException):
                        future.result()
            raise
        finally:
            self._run_lock.release()

    def close(self) -> None:
        if not self._closed:
            self._loader.shutdown(wait=True, cancel_futures=True)
            self._closed = True


@dataclass
class _CudaRingSlot:
    context: "SharedFullContext"
    transfer_stream: torch.cuda.Stream
    ready_event: torch.cuda.Event
    consumed_event: torch.cuda.Event
    loaded_generation: int = 0
    consumed_generation: int = 0


@dataclass(frozen=True)
class _LoadedExpertChunk:
    logical_expert_ids: tuple[int, ...]


class KTStreamPrefillExecutor:
    """Two-slot BF16 expert-chunk executor shared by every GLM MoE layer."""

    def __init__(
        self,
        plan: KTStreamPrefillPlan,
        owner: "KTEPWrapperMethod",
        layer: torch.nn.Module,
    ):
        from sglang.srt.layers.moe.kt_ep_wrapper import SharedFullContext

        self.plan = plan
        self._device = next(layer.parameters()).device
        self._ring = ReusableAsyncRing[
            tuple[int, ...], _LoadedExpertChunk, None
        ](plan.ring_slots)
        self._slots: list[_CudaRingSlot] = []

        init_args = owner._full_init_args
        if init_args is None:
            raise KTStreamPrefillAdmissionError(
                "KT stream-prefill cannot allocate before MoE weights are created"
            )
        for _ in range(plan.ring_slots):
            context = SharedFullContext(
                layer=layer,
                init_args=init_args,
                global_num_experts=plan.experts_per_chunk,
                moe_runner_config=owner.moe_runner_config,
                host_buffer_experts=plan.experts_per_chunk,
            )
            if not getattr(context, "is_bf16_quant", False):
                raise KTStreamPrefillAdmissionError(
                    "AMXINT4 stream-prefill requires an unquantized BF16 GPU "
                    "shadow layer"
                )
            self._slots.append(
                _CudaRingSlot(
                    context=context,
                    transfer_stream=torch.cuda.Stream(device=self._device),
                    ready_event=torch.cuda.Event(),
                    consumed_event=torch.cuda.Event(),
                )
            )

        logger.info(
            "KT stream-prefill admitted: experts_per_chunk=%d ring_slots=%d "
            "chunks_per_layer=%d device_ring=%.2f MiB pinned_host_per_rank=%.2f MiB",
            plan.experts_per_chunk,
            plan.ring_slots,
            plan.chunks_per_layer,
            plan.device_ring_bytes / (1024 * 1024),
            plan.host_ring_bytes_per_rank / (1024 * 1024),
        )

    def _publish_rank_zero_status(self, error: Optional[BaseException]) -> None:
        if not dist.is_initialized() or get_tensor_model_parallel_world_size() == 1:
            if error is not None:
                raise error
            return

        tp_group = get_tp_group()
        status: list[Optional[str]] = [
            None if error is None else f"{type(error).__name__}: {error}"
        ]
        dist.broadcast_object_list(
            status,
            src=tp_group.first_rank,
            group=tp_group.cpu_group,
        )
        if status[0] is not None:
            raise RuntimeError(
                "KT stream-prefill rank-0 AMXINT4 export failed: " + status[0]
            )

    def _load_chunk(
        self,
        source_wrapper: Optional[_KTWeightExporter],
        slot_index: int,
        generation: int,
        logical_expert_ids: tuple[int, ...],
    ) -> _LoadedExpertChunk:
        slot = self._slots[slot_index]
        if slot.loaded_generation != slot.consumed_generation:
            slot.consumed_event.synchronize()
            slot.consumed_generation = slot.loaded_generation

        torch.cuda.set_device(self._device)
        context = slot.context
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        rank_zero_error: Optional[BaseException] = None

        if tp_rank == 0:
            try:
                if source_wrapper is None:
                    raise RuntimeError(
                        "KT stream-prefill source wrapper is not initialized"
                    )
                w13_cpu = context.cpu_buffers["w13_weight"]
                w2_cpu = context.cpu_buffers["w2_weight"]
                w13_expert_bytes = w13_cpu[0].numel() * w13_cpu.element_size()
                w2_expert_bytes = w2_cpu[0].numel() * w2_cpu.element_size()
                for destination_index, logical_expert_id in enumerate(
                    logical_expert_ids
                ):
                    w13_pointers = [
                        pointer + destination_index * w13_expert_bytes
                        for pointer in context.all_rank_buffer_ptrs["w13_weight"]
                    ]
                    w2_pointers = [
                        pointer + destination_index * w2_expert_bytes
                        for pointer in context.all_rank_buffer_ptrs["w2_weight"]
                    ]
                    source_wrapper.submit_write_weight_scale_to_buffer(
                        tp_world_size,
                        logical_expert_id,
                        w13_pointers,
                        [0] * tp_world_size,
                        w2_pointers,
                        [0] * tp_world_size,
                    )
                source_wrapper.sync_write_weight_scale_to_buffer()
            except BaseException as error:
                rank_zero_error = error

        self._publish_rank_zero_status(rank_zero_error)

        with torch.cuda.stream(slot.transfer_stream):
            count = len(logical_expert_ids)
            context.gpu_layer.w13_weight[:count].copy_(
                context.cpu_buffers["w13_weight"][:count],
                non_blocking=True,
            )
            context.gpu_layer.w2_weight[:count].copy_(
                context.cpu_buffers["w2_weight"][:count],
                non_blocking=True,
            )
            slot.ready_event.record(slot.transfer_stream)

        slot.loaded_generation = generation
        return _LoadedExpertChunk(logical_expert_ids=logical_expert_ids)

    def _compute_chunk(
        self,
        dispatch_output: "StandardDispatchOutput",
        ticket: RingTicket[tuple[int, ...], _LoadedExpertChunk],
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import CombineInputChecker
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        slot = self._slots[ticket.slot_index]
        if ticket.generation != slot.loaded_generation:
            raise RuntimeError(
                "KT stream-prefill received a stale ring ticket: "
                f"ticket={ticket.generation}, loaded={slot.loaded_generation}"
            )

        x = dispatch_output.hidden_states
        current_stream = torch.cuda.current_stream(x.device)
        current_stream.wait_event(slot.ready_event)

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError(
                "KT stream-prefill requires StandardTopKOutput routing"
            )
        logical_to_slot = torch.full(
            (self.plan.num_experts,),
            -1,
            dtype=topk_output.topk_ids.dtype,
            device=topk_output.topk_ids.device,
        )
        logical_ids = torch.tensor(
            ticket.loaded.logical_expert_ids,
            dtype=topk_output.topk_ids.dtype,
            device=topk_output.topk_ids.device,
        )
        logical_to_slot[logical_ids] = torch.arange(
            len(ticket.loaded.logical_expert_ids),
            dtype=topk_output.topk_ids.dtype,
            device=topk_output.topk_ids.device,
        )
        remapped_topk_ids = logical_to_slot[topk_output.topk_ids]
        chunk_dispatch = dispatch_output._replace(
            topk_output=topk_output._replace(topk_ids=remapped_topk_ids)
        )
        combine_input = slot.context.gpu_method.apply(
            slot.context.gpu_layer,
            chunk_dispatch,
        )
        if not CombineInputChecker.format_is_standard(combine_input):
            raise RuntimeError(
                "KT stream-prefill GPU runner returned a non-standard combine input"
            )
        partial = combine_input.hidden_states
        slot.consumed_event.record(current_stream)
        return partial

    def apply(
        self,
        owner: "KTEPWrapperMethod",
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        chunks = tuple(
            tuple(
                range(
                    start,
                    min(start + self.plan.experts_per_chunk, self.plan.num_experts),
                )
            )
            for start in range(
                0,
                self.plan.num_experts,
                self.plan.experts_per_chunk,
            )
        )
        source_wrapper = owner.wrapper
        if source_wrapper is None and get_tensor_model_parallel_rank() == 0:
            raise RuntimeError("KT stream-prefill source wrapper is not initialized")

        try:
            output = torch.zeros_like(dispatch_output.hidden_states)

            def compute_and_accumulate(
                ticket: RingTicket[tuple[int, ...], _LoadedExpertChunk],
            ) -> None:
                partial = self._compute_chunk(dispatch_output, ticket)
                output.add_(partial)

            self._ring.run(
                chunks,
                load=lambda slot_index, generation, chunk: self._load_chunk(
                    source_wrapper,
                    slot_index,
                    generation,
                    chunk,
                ),
                compute=compute_and_accumulate,
            )
            return StandardCombineInput(hidden_states=output)
        except BaseException:
            # Fail closed: a model-process restart is preferable to reusing a
            # slot whose CUDA completion state is unknown.
            torch.cuda.synchronize(self._device)
            raise


_shared_stream_prefill_executor: Optional[KTStreamPrefillExecutor] = None
_SHARED_STREAM_PREFILL_LOCK = threading.Lock()


def get_existing_kt_stream_prefill_executor() -> Optional[KTStreamPrefillExecutor]:
    """Return an initialized ring without performing another VRAM admission."""

    return _shared_stream_prefill_executor


def get_or_create_kt_stream_prefill_executor(
    plan: KTStreamPrefillPlan,
    owner: "KTEPWrapperMethod",
    layer: torch.nn.Module,
) -> KTStreamPrefillExecutor:
    """Return the process-global ring shared by all homogeneous GLM MoE layers."""

    global _shared_stream_prefill_executor
    with _SHARED_STREAM_PREFILL_LOCK:
        if _shared_stream_prefill_executor is None:
            _shared_stream_prefill_executor = KTStreamPrefillExecutor(
                plan,
                owner,
                layer,
            )
        elif _shared_stream_prefill_executor.plan != plan:
            raise RuntimeError(
                "KT stream-prefill ring was initialized with an incompatible plan: "
                f"existing={_shared_stream_prefill_executor.plan}, requested={plan}"
            )
        return _shared_stream_prefill_executor


def iter_expert_chunks(
    num_experts: int,
    experts_per_chunk: int,
) -> Iterable[tuple[int, ...]]:
    """Pure helper used by sizing/contract tests and offline planners."""

    if num_experts <= 0 or experts_per_chunk <= 0:
        raise ValueError("num_experts and experts_per_chunk must be positive")
    for start in range(0, num_experts, experts_per_chunk):
        yield tuple(range(start, min(start + experts_per_chunk, num_experts)))
