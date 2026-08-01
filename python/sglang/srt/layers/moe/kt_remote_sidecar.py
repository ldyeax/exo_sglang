# SPDX-License-Identifier: Apache-2.0
"""Persistent native-MXFP4 CPU-expert sidecar transport.

The owner submits its local CPU shard first, then performs this request while
the local AMX/AVX-512 pool is running.  The sidecar returns only its partial
MoE sum, so the checkpoint weights remain lossless and the owner can add the
two contributions directly.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import threading
import time
from typing import ClassVar

import torch

_REQUEST_HEADER = struct.Struct("!4sIIII")
_RESPONSE_HEADER = struct.Struct("!4sII")
_REQUEST_MAGIC = b"KTR1"
_RESPONSE_MAGIC = b"KTO1"

logger = logging.getLogger(__name__)


def _recv_exact(connection: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    _recv_into(connection, memoryview(data))
    return data


def _recv_into(connection: socket.socket, view: memoryview) -> None:
    if view.format != "B" or view.ndim != 1:
        view = view.cast("B")
    offset = 0
    while offset < len(view):
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("KTransformers expert sidecar closed the connection")
        offset += received


class KTExpertSidecarClient:
    """One persistent, serialized IB connection per scheduler process."""

    _instances: ClassVar[dict[str, "KTExpertSidecarClient"]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, endpoint: str):
        host, separator, port_text = endpoint.rpartition(":")
        if not separator or not host:
            raise ValueError(
                "KTransformers expert-sidecar endpoint must be HOST:PORT, "
                f"got {endpoint!r}"
            )
        self.endpoint = endpoint
        self._host = host
        self._port = int(port_text)
        self._connection: socket.socket | None = None
        self._request_lock = threading.Lock()
        self._request_counts: dict[int, int] = {}

    @classmethod
    def get(cls, endpoint: str) -> "KTExpertSidecarClient":
        with cls._instances_lock:
            instance = cls._instances.get(endpoint)
            if instance is None:
                instance = cls(endpoint)
                cls._instances[endpoint] = instance
            return instance

    def _connect(self) -> socket.socket:
        if self._connection is None:
            connection = socket.create_connection(
                (self._host, self._port), timeout=120.0
            )
            connection.settimeout(1200.0)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
            self._connection = connection
        return self._connection

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def forward(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        return_cpu: bool = False,
    ) -> torch.Tensor:
        """Return the sidecar's partial weighted expert sum.

        ``return_cpu`` receives directly into pinned host storage so callers
        can stream the result through an existing GPU staging tensor.
        """
        timing_enabled = os.environ.get("SGLANG_KT_REMOTE_TIMING") == "1"
        start_time = time.perf_counter() if timing_enabled else 0.0
        flat_hidden_states = hidden_states.detach().view(
            -1, hidden_states.shape[-1]
        )
        batch_size, hidden_size = flat_hidden_states.shape
        topk = topk_ids.shape[-1]
        if topk_ids.shape != (batch_size, topk):
            raise ValueError("KTransformers sidecar top-k IDs have invalid shape")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("KTransformers sidecar top-k weights have invalid shape")

        hidden_cpu = flat_hidden_states.to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        ids_cpu = topk_ids.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        weights_cpu = topk_weights.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        payload_ready_time = time.perf_counter() if timing_enabled else 0.0
        expected_output_bytes = batch_size * hidden_size * 2

        with self._request_lock:
            try:
                connection = self._connect()
                connection.sendall(
                    _REQUEST_HEADER.pack(
                        _REQUEST_MAGIC,
                        int(layer_idx),
                        int(batch_size),
                        int(hidden_size),
                        int(topk),
                    )
                )
                connection.sendall(
                    memoryview(hidden_cpu.view(torch.uint8).numpy())
                )
                connection.sendall(memoryview(ids_cpu.view(torch.uint8).numpy()))
                connection.sendall(
                    memoryview(weights_cpu.view(torch.uint8).numpy())
                )
                response_header = _recv_exact(connection, _RESPONSE_HEADER.size)
                magic, status, payload_size = _RESPONSE_HEADER.unpack(
                    response_header
                )
                if magic != _RESPONSE_MAGIC:
                    raise RuntimeError(
                        "KTransformers expert sidecar returned an invalid magic"
                    )
                if status != 0:
                    payload = _recv_exact(connection, payload_size)
                    raise RuntimeError(
                        "KTransformers expert sidecar failed: "
                        + payload.decode("utf-8", errors="replace")
                    )
                if payload_size != expected_output_bytes:
                    raise RuntimeError(
                        "KTransformers expert sidecar returned an invalid "
                        f"payload size: expected={expected_output_bytes}, "
                        f"actual={payload_size}"
                    )
                output_cpu = torch.empty(
                    (batch_size, hidden_size),
                    device="cpu",
                    dtype=torch.bfloat16,
                    pin_memory=True,
                )
                _recv_into(
                    connection,
                    memoryview(output_cpu.view(torch.uint8).numpy()),
                )
            except Exception:
                self._close()
                raise

        response_ready_time = time.perf_counter() if timing_enabled else 0.0
        output = (
            output_cpu
            if return_cpu
            else output_cpu.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).view_as(hidden_states)
        )
        if timing_enabled:
            completed_time = time.perf_counter()
            request_count = self._request_counts.get(layer_idx, 0) + 1
            self._request_counts[layer_idx] = request_count
            if request_count == 1 or request_count % 128 == 0:
                active_ids = ids_cpu[ids_cpu >= 0]
                logger.info(
                    "KTransformers sidecar timing endpoint=%s layer=%d "
                    "requests=%d batch=%d active_routes=%d "
                    "active_experts=%d prepare_ms=%.3f wire_ms=%.3f "
                    "return_ms=%.3f total_ms=%.3f",
                    self.endpoint,
                    layer_idx,
                    request_count,
                    batch_size,
                    active_ids.numel(),
                    torch.unique(active_ids).numel(),
                    (payload_ready_time - start_time) * 1000,
                    (response_ready_time - payload_ready_time) * 1000,
                    (completed_time - response_ready_time) * 1000,
                    (completed_time - start_time) * 1000,
                )
        return output
