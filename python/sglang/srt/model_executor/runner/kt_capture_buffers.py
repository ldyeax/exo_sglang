from __future__ import annotations

from collections.abc import Iterable

try:
    from kt_kernel import KTMoEWrapper
    from kt_kernel.experts_base import KExpertsCPUBuffer

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


def register_kt_capture_batch_sizes(capture_batch_sizes: Iterable[int]) -> None:
    """Retain KT pinned buffers for every CUDA-graph runner in the process."""
    if not KTRANSFORMERS_AVAILABLE:
        return

    requested_sizes = {int(batch_size) for batch_size in capture_batch_sizes}
    registered_sizes = {
        int(batch_size) for batch_size in KTMoEWrapper.get_capture_batch_sizes()
    }
    all_registered_sizes = registered_sizes.union(requested_sizes)
    KTMoEWrapper.set_capture_batch_sizes(sorted(all_registered_sizes))

    # Older kt-kernel builds keep an unregistered shape in one replaceable
    # temp slot. Registering that shape after its first allocation does not
    # retroactively move the tuple into capture_buffers, so a later shape can
    # free storage whose raw pointers were baked into a CUDA graph. Promote
    # the live temp tuple here as well as in newer kt-kernel implementations.
    temp_batch_size = int(KExpertsCPUBuffer.temp_bs)
    if temp_batch_size in all_registered_sizes and KExpertsCPUBuffer.temp_buffer:
        KExpertsCPUBuffer.capture_buffers[temp_batch_size] = (
            KExpertsCPUBuffer.temp_buffer
        )
