# SPDX-License-Identifier: Apache-2.0
"""Low-dependency host-buffer ownership contract for KT stream prefill."""

from typing import Literal

SharedFullContextHostBufferMode = Literal[
    "legacy_double_buffer",
    "stream_prefill_ring_slot",
]


def validate_shared_full_context_host_buffer(
    *,
    host_buffer_experts: int,
    global_num_experts: int,
    host_buffer_mode: SharedFullContextHostBufferMode,
) -> None:
    """Admit a single host entry only when an outer stream ring owns reuse."""

    if host_buffer_experts < 1:
        raise ValueError("host_buffer_experts must be positive")
    if host_buffer_mode == "legacy_double_buffer":
        if host_buffer_experts < 2:
            raise ValueError(
                "legacy SharedFullContext host buffers require at least "
                "2 experts for per-expert double buffering"
            )
        return
    if host_buffer_mode == "stream_prefill_ring_slot":
        if host_buffer_experts != global_num_experts:
            raise ValueError(
                "a stream-prefill ring context requires one host-buffer "
                "entry for every expert in its GPU chunk"
            )
        return
    raise ValueError(
        f"unsupported SharedFullContext host buffer mode: {host_buffer_mode}"
    )
