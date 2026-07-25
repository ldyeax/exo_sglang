# SPDX-License-Identifier: Apache-2.0
"""Executable GLM-5.2 prefill/decode split for a two-GPU host.

The normal SGLang PD path already has the right KV-cache protocol: the model
gateway dual-dispatches a request, assigns an unpredictable bootstrap room,
and passes the selected prefill worker's bootstrap endpoint to both workers.
The decode worker preallocates its KV pages and the prefill worker transfers
the populated pages before decode is admitted.

This module turns that protocol into a fail-closed ``dwagon`` topology:

* one PP=1/TP=1 non-SmallEP stream-prefill worker is pinned by UUID to the
  first GPU;
* one PP=1/TP=1 decode worker is pinned by UUID to the second GPU;
* both workers attach the same read-only AMXINT4 safetensors generation;
* each worker has its own process tree, runtime directory, scratch directory,
  CUDA context, KTransformers task queues, and staging buffers;
* the native SGLang model gateway is the only public request endpoint; and
* the gateway starts only after both workers are healthy and their live
  KTransformers leases prove that they attached the same generation.

The launcher intentionally does not hash the multi-hundred-gigabyte checkpoint
on every start.  It authenticates the small manifest, verifies its content
table and immutable inode metadata, then relies on the artifact manager's
recorded per-file SHA-256 digests.  Any identity, NUMA, device-capacity,
process-ownership, lease, or KV-contract mismatch aborts before the public
router is exposed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import ipaddress
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, Literal, cast

from sglang.srt.disaggregation.glm52_intra_node_policy import (
    DWAGON_HARDWARE_PROFILE,
    GLM52CapacityContract,
    GLM52ExecutorCompatibility,
    GLM52IntraNodeContracts,
    GLM52SharedHostWeights,
)

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID: Final = re.compile(r"^GPU-[0-9a-fA-F-]{16,64}$")
_MANIFEST_KIND: Final = "kt_shared_host_weights_manifest"
_CONTENT_KIND: Final = "kt_shared_host_weights_content"
_MANIFEST_SCHEMA_VERSION: Final = 1
_MAXIMUM_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_MIB: Final = 1024 * 1024
_PROC_ROOT: Final = Path("/proc")

GLM52PDRole = Literal["prefill", "decode"]
GLM52PDTransferBackend = Literal["mooncake", "nixl", "mori"]


class GLM52PDRuntimeContractError(RuntimeError):
    """Raised before exposure when the exact split contract cannot be proven."""


@dataclass(frozen=True)
class GLM52PDDeviceContract:
    """Stable identity, NUMA placement, and launch capacity for one worker."""

    gpu_uuid: str
    numa_node: int
    required_free_bytes: int

    def __post_init__(self) -> None:
        if _GPU_UUID.fullmatch(self.gpu_uuid) is None:
            raise GLM52PDRuntimeContractError(
                f"GPU identity must be a full NVIDIA UUID, got {self.gpu_uuid!r}"
            )
        if self.numa_node < 0:
            raise GLM52PDRuntimeContractError("GPU NUMA node must be non-negative")
        if self.required_free_bytes <= 0:
            raise GLM52PDRuntimeContractError(
                "required GPU free capacity must be positive"
            )


@dataclass(frozen=True)
class GLM52PDSharedWeightContract:
    """Expected immutable AMXINT4 artifact and common lease generation."""

    checkpoint_root: Path
    manifest_path: Path
    state_directory: Path
    content_id: str
    manifest_sha256: str
    weight_bytes: int
    host_safety_margin_bytes: int

    def __post_init__(self) -> None:
        for description, path in (
            ("checkpoint root", self.checkpoint_root),
            ("manifest", self.manifest_path),
            ("state directory", self.state_directory),
        ):
            if not path.is_absolute():
                raise GLM52PDRuntimeContractError(
                    f"{description} path must be absolute: {path}"
                )
        for description, digest in (
            ("content ID", self.content_id),
            ("manifest SHA-256", self.manifest_sha256),
        ):
            if _SHA256.fullmatch(digest) is None:
                raise GLM52PDRuntimeContractError(
                    f"shared-weight {description} is not a SHA-256 digest"
                )
        if self.weight_bytes <= 0:
            raise GLM52PDRuntimeContractError(
                "shared-host-weight byte count must be positive"
            )
        if self.host_safety_margin_bytes < 0:
            raise GLM52PDRuntimeContractError(
                "shared-host-weight safety margin must not be negative"
            )


@dataclass(frozen=True)
class GLM52PDKVHandoffContract:
    """Parameters that must agree across prefill, decode, and the gateway."""

    transfer_backend: GLM52PDTransferBackend
    bootstrap_port: int
    context_length: int
    maximum_prefill_tokens: int
    maximum_total_tokens: int
    kv_cache_dtype: str = "bfloat16"
    page_size: int = 1
    ib_device: str | None = None

    def __post_init__(self) -> None:
        if self.transfer_backend not in {"mooncake", "nixl", "mori"}:
            raise GLM52PDRuntimeContractError(
                "intra-node KV transfer backend must be mooncake, nixl, or mori"
            )
        _validate_port(self.bootstrap_port, "bootstrap")
        if self.context_length <= 0:
            raise GLM52PDRuntimeContractError("context length must be positive")
        if self.maximum_prefill_tokens <= 0:
            raise GLM52PDRuntimeContractError("maximum prefill tokens must be positive")
        if self.maximum_total_tokens < self.context_length:
            raise GLM52PDRuntimeContractError(
                "maximum total tokens must cover one full context"
            )
        if self.maximum_prefill_tokens > self.maximum_total_tokens:
            raise GLM52PDRuntimeContractError(
                "maximum prefill tokens must not exceed maximum total tokens"
            )
        if self.kv_cache_dtype not in {"bfloat16", "bf16", "auto"}:
            raise GLM52PDRuntimeContractError(
                "split KV cache dtype must be bfloat16/bf16 or auto"
            )
        if self.page_size <= 0:
            raise GLM52PDRuntimeContractError("KV page size must be positive")
        if self.ib_device is not None and (
            not self.ib_device
            or any(character.isspace() for character in self.ib_device)
        ):
            raise GLM52PDRuntimeContractError(
                "IB device list must be a non-empty comma-separated token"
            )


@dataclass(frozen=True)
class GLM52PDLaunchConfig:
    """Complete immutable launch contract for one local PD pair."""

    python_executable: Path
    model_path: Path
    model_identity: str
    served_model_name: str
    architecture: str
    runtime_root: Path
    shared_weights: GLM52PDSharedWeightContract
    prefill_device: GLM52PDDeviceContract
    decode_device: GLM52PDDeviceContract
    kv_handoff: GLM52PDKVHandoffContract
    prefill_host: str
    prefill_port: int
    decode_host: str
    decode_port: int
    router_host: str
    router_port: int
    cpu_infer_threads: int
    threadpool_numa_nodes: tuple[int, int]
    chunked_prefill_size: int
    stream_prefill_token_threshold: int
    stream_prefill_experts_per_chunk: int = 4
    stream_prefill_ring_slots: int = 2
    stream_prefill_safety_margin_mib: int = 512
    maximum_prefill_requests: int = 1
    maximum_decode_requests: int = 2
    static_memory_fraction: float = 0.92
    attention_backend: str = "flashinfer"
    startup_timeout_seconds: int = 3600
    health_poll_seconds: float = 1.0
    shutdown_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        for description, path in (
            ("Python executable", self.python_executable),
            ("model", self.model_path),
            ("runtime root", self.runtime_root),
        ):
            if not path.is_absolute():
                raise GLM52PDRuntimeContractError(
                    f"{description} path must be absolute: {path}"
                )
        if _SHA256.fullmatch(self.model_identity) is None:
            raise GLM52PDRuntimeContractError("model identity must be a SHA-256 digest")
        if not self.served_model_name:
            raise GLM52PDRuntimeContractError("served model name must not be empty")
        if self.architecture not in {
            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForConditionalGeneration",
        }:
            raise GLM52PDRuntimeContractError(
                f"unsupported GLM-5.2 architecture {self.architecture!r}"
            )
        if self.prefill_device.gpu_uuid == self.decode_device.gpu_uuid:
            raise GLM52PDRuntimeContractError(
                "prefill and decode must use distinct GPU UUIDs"
            )
        if self.prefill_device.numa_node == self.decode_device.numa_node:
            raise GLM52PDRuntimeContractError(
                "dwagon split requires one GPU on each distinct NUMA node"
            )
        expected_nodes = {
            self.prefill_device.numa_node,
            self.decode_device.numa_node,
        }
        if set(self.threadpool_numa_nodes) != expected_nodes:
            raise GLM52PDRuntimeContractError(
                "KT NUMA pools must exactly cover the two GPU NUMA nodes"
            )
        if len(set(self.threadpool_numa_nodes)) != 2:
            raise GLM52PDRuntimeContractError(
                "KT threadpool NUMA nodes must be distinct"
            )
        for description, host in (
            ("prefill", self.prefill_host),
            ("decode", self.decode_host),
            ("router", self.router_host),
        ):
            _validate_loopback_host(host, description)
        ports = (
            self.prefill_port,
            self.decode_port,
            self.router_port,
            self.kv_handoff.bootstrap_port,
        )
        for description, port in zip(
            ("prefill", "decode", "router", "bootstrap"),
            ports,
            strict=True,
        ):
            _validate_port(port, description)
        if len(set(ports)) != len(ports):
            raise GLM52PDRuntimeContractError(
                "prefill, decode, router, and bootstrap ports must be distinct"
            )
        positive_values = (
            ("CPU inference threads", self.cpu_infer_threads),
            ("chunked prefill size", self.chunked_prefill_size),
            ("stream-prefill threshold", self.stream_prefill_token_threshold),
            ("prefill request capacity", self.maximum_prefill_requests),
            ("decode request capacity", self.maximum_decode_requests),
            ("startup timeout", self.startup_timeout_seconds),
            ("shutdown timeout", self.shutdown_timeout_seconds),
        )
        for description, value in positive_values:
            if value <= 0:
                raise GLM52PDRuntimeContractError(f"{description} must be positive")
        if self.stream_prefill_token_threshold > self.kv_handoff.maximum_prefill_tokens:
            raise GLM52PDRuntimeContractError(
                "stream-prefill threshold exceeds maximum prefill tokens"
            )
        if self.stream_prefill_experts_per_chunk not in {1, 2, 4, 8, 16}:
            raise GLM52PDRuntimeContractError(
                "stream-prefill experts per chunk must be a power of two in [1, 16]"
            )
        if self.stream_prefill_ring_slots != 2:
            raise GLM52PDRuntimeContractError(
                "stream-prefill handoff currently requires exactly two ring slots"
            )
        if self.stream_prefill_safety_margin_mib < 0:
            raise GLM52PDRuntimeContractError(
                "stream-prefill safety margin must not be negative"
            )
        tp1_expert_bytes = 3 * 6144 * 2048 * 2
        stream_prefill_reserve = (
            tp1_expert_bytes
            * self.stream_prefill_experts_per_chunk
            * self.stream_prefill_ring_slots
            + self.stream_prefill_safety_margin_mib * _MIB
        )
        if self.prefill_device.required_free_bytes < stream_prefill_reserve:
            capacity_message = " ".join(
                (
                    "prefill GPU capacity contract does not cover its TP1 BF16",
                    "expert ring plus safety margin:",
                )
            )
            raise GLM52PDRuntimeContractError(
                "{} {}<{}".format(
                    capacity_message,
                    self.prefill_device.required_free_bytes,
                    stream_prefill_reserve,
                )
            )
        if not 0.0 < self.static_memory_fraction < 1.0:
            raise GLM52PDRuntimeContractError(
                "static memory fraction must be strictly between zero and one"
            )
        if self.health_poll_seconds <= 0:
            raise GLM52PDRuntimeContractError("health poll interval must be positive")


@dataclass(frozen=True)
class GLM52PDGPUObservation:
    """One ``nvidia-smi`` row resolved to its current PCI/NUMA placement."""

    gpu_uuid: str
    index: int
    pci_bus_id: str
    numa_node: int
    total_bytes: int
    free_bytes: int
    compute_process_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class GLM52PDCommand:
    """Auditable process command with only contract-owned environment entries."""

    role: str
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    working_directory: Path
    log_path: Path

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True)
class GLM52PDLaunchPlan:
    """Validated commands and policy contracts, before any process is started."""

    config: GLM52PDLaunchConfig
    prefill: GLM52PDCommand
    decode: GLM52PDCommand
    router: GLM52PDCommand
    policy_contracts: GLM52IntraNodeContracts
    shared_manifest_numa_nodes: tuple[int, ...]

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "glm52_intra_node_pd_launch_plan",
            "hardware_profile": DWAGON_HARDWARE_PROFILE,
            "model_identity": self.config.model_identity,
            "served_model_name": self.config.served_model_name,
            "architecture": self.config.architecture,
            "shared_weight_content_id": self.config.shared_weights.content_id,
            "shared_weight_manifest_sha256": (
                self.config.shared_weights.manifest_sha256
            ),
            "shared_weight_bytes": self.config.shared_weights.weight_bytes,
            "shared_manifest_numa_nodes": list(self.shared_manifest_numa_nodes),
            "parallelism": {
                "prefill": {"pipeline_parallel_size": 1, "tensor_parallel_size": 1},
                "decode": {"pipeline_parallel_size": 1, "tensor_parallel_size": 1},
            },
            "process_isolation": {
                "separate_process_trees": True,
                "private_cuda_contexts": True,
                "private_kt_task_queues": True,
                "private_staging_and_scratch": True,
                "shared_objects": [
                    "read_only_file_backed_amxint4_pages",
                    "shared_weight_generation_leases",
                ],
            },
            "kv_handoff": asdict(self.config.kv_handoff),
            "commands": {
                command.role: {
                    "argv": list(command.argv),
                    "environment": dict(command.environment),
                    "working_directory": str(command.working_directory),
                    "log_path": str(command.log_path),
                }
                for command in (self.prefill, self.decode, self.router)
            },
        }


@dataclass(frozen=True)
class GLM52PDProcessIdentity:
    role: str
    process_id: int
    start_time_ticks: int


@dataclass(frozen=True)
class GLM52PDLeaseProof:
    generation_id: str
    prefill_lease_process_ids: tuple[int, ...]
    decode_lease_process_ids: tuple[int, ...]


def _validate_port(port: int, description: str) -> None:
    if not 1 <= port <= 65535:
        raise GLM52PDRuntimeContractError(f"{description} port must be in [1, 65535]")


def _validate_loopback_host(host: str, description: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise GLM52PDRuntimeContractError(
            f"{description} host must be a numeric loopback address"
        ) from error
    if not address.is_loopback:
        raise GLM52PDRuntimeContractError(
            f"{description} host must be loopback for an intra-node split"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest path must be a non-empty string"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
        or "\x00" in value
    ):
        raise GLM52PDRuntimeContractError(
            f"unsafe shared-weight manifest path {value!r}"
        )
    return value


def _open_regular_nofollow(path: Path, description: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GLM52PDRuntimeContractError(
            f"cannot open {description}: {path}"
        ) from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise GLM52PDRuntimeContractError(
            f"{description} is not a regular file: {path}"
        )
    return descriptor, metadata


def _read_manifest_bytes(path: Path) -> bytes:
    descriptor, metadata = _open_regular_nofollow(
        path,
        "shared-weight manifest",
    )
    try:
        if metadata.st_size > _MAXIMUM_MANIFEST_BYTES:
            raise GLM52PDRuntimeContractError(
                "shared-weight manifest exceeds the size limit"
            )
        output = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            output.extend(chunk)
            if len(output) > _MAXIMUM_MANIFEST_BYTES:
                raise GLM52PDRuntimeContractError(
                    "shared-weight manifest exceeds the size limit"
                )
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(metadata) != identity(after):
            raise GLM52PDRuntimeContractError(
                "shared-weight manifest changed while it was read"
            )
        return bytes(output)
    finally:
        os.close(descriptor)


def verify_shared_weight_contract(
    contract: GLM52PDSharedWeightContract,
    *,
    host_capacity_bytes: int | None = None,
) -> tuple[int, ...]:
    """Authenticate the manifest and immutable file metadata without rehashing."""

    try:
        checkpoint_root = contract.checkpoint_root.resolve(strict=True)
        manifest_path = contract.manifest_path.resolve(strict=True)
    except OSError as error:
        raise GLM52PDRuntimeContractError(
            "shared-weight checkpoint or manifest does not exist"
        ) from error
    if checkpoint_root != contract.checkpoint_root:
        raise GLM52PDRuntimeContractError(
            "shared-weight checkpoint root must be canonical, not a symlink"
        )
    if manifest_path != contract.manifest_path:
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest path must be canonical, not a symlink"
        )
    if not checkpoint_root.is_dir():
        raise GLM52PDRuntimeContractError(
            "shared-weight checkpoint root is not a directory"
        )

    raw = _read_manifest_bytes(manifest_path)
    if _sha256_bytes(raw) != contract.manifest_sha256:
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest SHA-256 does not match the launch contract"
        )
    try:
        value = cast(object, json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest must be a JSON object"
        )
    manifest = cast(dict[str, object], value)
    if (
        manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != _MANIFEST_KIND
    ):
        raise GLM52PDRuntimeContractError("unsupported shared-weight manifest schema")
    if manifest.get("content_id") != contract.content_id:
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest content ID does not match the launch contract"
        )
    raw_nodes_value = manifest.get("numa_nodes")
    if not isinstance(raw_nodes_value, list) or not raw_nodes_value:
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest has an invalid NUMA contract"
        )
    raw_nodes = cast(list[object], raw_nodes_value)
    parsed_nodes: list[int] = []
    for node in raw_nodes:
        if isinstance(node, bool) or not isinstance(node, int) or node < 0:
            raise GLM52PDRuntimeContractError(
                "shared-weight manifest has an invalid NUMA contract"
            )
        parsed_nodes.append(node)
    numa_nodes = tuple(parsed_nodes)
    if len(set(numa_nodes)) != len(numa_nodes):
        raise GLM52PDRuntimeContractError(
            "shared-weight manifest NUMA nodes are not unique"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise GLM52PDRuntimeContractError("shared-weight manifest has no file table")
    content_entries: list[dict[str, object]] = []
    total_bytes = 0
    previous_path: str | None = None
    for raw_entry in cast(list[object], raw_files):
        if not isinstance(raw_entry, dict):
            raise GLM52PDRuntimeContractError(
                "shared-weight manifest file entry is not an object"
            )
        entry = cast(dict[str, object], raw_entry)
        relative_path = _safe_relative_path(entry.get("path"))
        size_bytes = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise GLM52PDRuntimeContractError(
                f"invalid byte count for shared-weight file {relative_path}"
            )
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise GLM52PDRuntimeContractError(
                f"invalid digest for shared-weight file {relative_path}"
            )
        if previous_path is not None and relative_path <= previous_path:
            raise GLM52PDRuntimeContractError(
                "shared-weight manifest file table is not strictly sorted"
            )
        previous_path = relative_path
        candidate = checkpoint_root.joinpath(*PurePosixPath(relative_path).parts)
        current = checkpoint_root
        for part in PurePosixPath(relative_path).parts:
            current /= part
            try:
                if stat.S_ISLNK(os.lstat(current).st_mode):
                    raise GLM52PDRuntimeContractError(
                        f"shared-weight file traverses a symlink: {relative_path}"
                    )
            except OSError as error:
                raise GLM52PDRuntimeContractError(
                    f"cannot inspect shared-weight file {relative_path}"
                ) from error
        descriptor, metadata = _open_regular_nofollow(
            candidate,
            f"shared-weight file {relative_path}",
        )
        os.close(descriptor)
        if metadata.st_size != size_bytes:
            raise GLM52PDRuntimeContractError(
                f"shared-weight file size changed: {relative_path}"
            )
        mount_read_only = bool(os.statvfs(candidate).f_flag & os.ST_RDONLY)
        mode_read_only = metadata.st_mode & 0o222 == 0
        if not mount_read_only and not mode_read_only:
            raise GLM52PDRuntimeContractError(
                f"shared-weight file is writable: {relative_path}"
            )
        total_bytes += size_bytes
        content_entries.append(
            {
                "path": relative_path,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )

    content_descriptor = {
        "files": content_entries,
        "kind": _CONTENT_KIND,
        "schema_version": _MANIFEST_SCHEMA_VERSION,
    }
    if _sha256_bytes(_canonical_json_bytes(content_descriptor)) != contract.content_id:
        raise GLM52PDRuntimeContractError(
            "shared-weight content ID does not match its file table"
        )
    if total_bytes != contract.weight_bytes:
        byte_count_message = " ".join(
            (
                "shared-weight manifest byte count does not match",
                "the launch contract:",
            )
        )
        raise GLM52PDRuntimeContractError(
            "{} {}!={}".format(
                byte_count_message,
                total_bytes,
                contract.weight_bytes,
            )
        )
    capacity = (
        host_capacity_bytes
        if host_capacity_bytes is not None
        else os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    )
    if capacity < total_bytes + contract.host_safety_margin_bytes:
        host_capacity_message = " ".join(
            (
                "host capacity is insufficient for shared weights",
                "plus safety margin:",
            )
        )
        raise GLM52PDRuntimeContractError(
            "{} {}<{}+{}".format(
                host_capacity_message,
                capacity,
                total_bytes,
                contract.host_safety_margin_bytes,
            )
        )
    return numa_nodes


def model_metadata_identity(model_path: Path) -> str:
    """Hash the small HF topology/index files that identify GPU-side weights."""

    entries: list[dict[str, object]] = []
    for relative_path in ("config.json", "model.safetensors.index.json"):
        path = model_path / relative_path
        descriptor, metadata = _open_regular_nofollow(
            path,
            f"model identity file {relative_path}",
        )
        try:
            if metadata.st_size > _MAXIMUM_MANIFEST_BYTES:
                raise GLM52PDRuntimeContractError(
                    f"model identity file is too large: {relative_path}"
                )
            raw = bytearray()
            while chunk := os.read(descriptor, 1024 * 1024):
                raw.extend(chunk)
                if len(raw) > _MAXIMUM_MANIFEST_BYTES:
                    raise GLM52PDRuntimeContractError(
                        f"model identity file is too large: {relative_path}"
                    )
            after = os.fstat(descriptor)
            before_identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity:
                raise GLM52PDRuntimeContractError(
                    f"model identity file changed while read: {relative_path}"
                )
            entries.append(
                {
                    "path": relative_path,
                    "sha256": _sha256_bytes(bytes(raw)),
                    "size_bytes": metadata.st_size,
                }
            )
        finally:
            os.close(descriptor)
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "files": entries,
                "kind": "glm52_model_metadata_identity",
                "schema_version": 1,
            }
        )
    )


def validate_gpu_contracts(
    config: GLM52PDLaunchConfig,
    observations: Sequence[GLM52PDGPUObservation],
) -> tuple[GLM52PDGPUObservation, GLM52PDGPUObservation]:
    """Resolve UUIDs against current PCI slots and reject stale owners/capacity."""

    by_uuid = {observation.gpu_uuid: observation for observation in observations}
    if len(by_uuid) != len(observations):
        raise GLM52PDRuntimeContractError("GPU inventory contains duplicate UUIDs")
    resolved: list[GLM52PDGPUObservation] = []
    for role, contract in (
        ("prefill", config.prefill_device),
        ("decode", config.decode_device),
    ):
        observation = by_uuid.get(contract.gpu_uuid)
        if observation is None:
            raise GLM52PDRuntimeContractError(
                f"{role} GPU UUID is absent from the current inventory"
            )
        if observation.numa_node != contract.numa_node:
            raise GLM52PDRuntimeContractError(
                "{} GPU {} moved to NUMA {}, expected {}".format(
                    role,
                    contract.gpu_uuid,
                    observation.numa_node,
                    contract.numa_node,
                )
            )
        if observation.free_bytes < contract.required_free_bytes:
            raise GLM52PDRuntimeContractError(
                "{} GPU has {} free bytes but {} are required".format(
                    role,
                    observation.free_bytes,
                    contract.required_free_bytes,
                )
            )
        if observation.compute_process_ids:
            raise GLM52PDRuntimeContractError(
                "{} GPU already has CUDA compute owners {}".format(
                    role,
                    observation.compute_process_ids,
                )
            )
        resolved.append(observation)
    return resolved[0], resolved[1]


def _normalize_pci_bus_id(value: str) -> str:
    normalized = value.strip().lower()
    fields = normalized.split(":")
    if len(fields) == 3 and len(fields[0]) == 8:
        normalized = f"{fields[0][-4:]}:{fields[1]}:{fields[2]}"
    return normalized


def collect_gpu_observations() -> tuple[GLM52PDGPUObservation, ...]:
    """Collect stable UUID, current PCI/NUMA, capacity, and process ownership."""

    inventory_command = (
        "nvidia-smi",
        "--query-gpu=uuid,index,pci.bus_id,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    process_command = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    )
    try:
        inventory = subprocess.run(
            inventory_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        processes = subprocess.run(
            process_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GLM52PDRuntimeContractError(
            "could not collect bounded NVIDIA GPU inventory"
        ) from error

    process_ids: dict[str, list[int]] = {}
    try:
        for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True):
            normalized = tuple(field.strip() for field in row)
            if not normalized or all(not field for field in normalized):
                continue
            if len(normalized) != 2:
                raise ValueError("invalid compute process row")
            process_ids.setdefault(normalized[0], []).append(int(normalized[1]))

        observations: list[GLM52PDGPUObservation] = []
        for row in csv.reader(inventory.stdout.splitlines(), skipinitialspace=True):
            normalized = tuple(field.strip() for field in row)
            if not normalized or all(not field for field in normalized):
                continue
            if len(normalized) != 5:
                raise ValueError("invalid GPU inventory row")
            gpu_uuid, index, raw_bus_id, total_mib, free_mib = normalized
            pci_bus_id = _normalize_pci_bus_id(raw_bus_id)
            numa_path = Path("/sys/bus/pci/devices") / pci_bus_id / "numa_node"
            numa_node = int(numa_path.read_text().strip())
            if numa_node < 0:
                raise ValueError(f"GPU {gpu_uuid} has no resolved NUMA node")
            observations.append(
                GLM52PDGPUObservation(
                    gpu_uuid=gpu_uuid,
                    index=int(index),
                    pci_bus_id=pci_bus_id,
                    numa_node=numa_node,
                    total_bytes=int(total_mib) * _MIB,
                    free_bytes=int(free_mib) * _MIB,
                    compute_process_ids=tuple(sorted(process_ids.get(gpu_uuid, ()))),
                )
            )
    except (OSError, ValueError) as error:
        raise GLM52PDRuntimeContractError(
            "NVIDIA inventory or current PCI/NUMA placement is invalid"
        ) from error
    return tuple(sorted(observations, key=lambda observation: observation.index))


def _require_safe_directory(path: Path, *, create: bool) -> Path:
    if not path.is_absolute():
        raise GLM52PDRuntimeContractError(f"runtime directory must be absolute: {path}")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GLM52PDRuntimeContractError(
            f"cannot inspect runtime directory: {path}"
        ) from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise GLM52PDRuntimeContractError(
            f"runtime directory is not a safe owned directory: {path}"
        )
    os.chmod(path, 0o700)
    return path


def prepare_runtime_directories(config: GLM52PDLaunchConfig) -> None:
    """Create role-private runtime roots and the common lease directory."""

    _ = _require_safe_directory(config.runtime_root, create=True)
    _ = _require_safe_directory(
        config.shared_weights.state_directory,
        create=True,
    )
    for role in ("prefill", "decode", "router"):
        role_root = _require_safe_directory(
            config.runtime_root / role,
            create=True,
        )
        _ = _require_safe_directory(role_root / "scratch", create=True)
        _ = _require_safe_directory(role_root / "cache", create=True)
        _ = _require_safe_directory(role_root / "logs", create=True)


def _worker_command(
    config: GLM52PDLaunchConfig,
    role: GLM52PDRole,
) -> GLM52PDCommand:
    device = config.prefill_device if role == "prefill" else config.decode_device
    host = config.prefill_host if role == "prefill" else config.decode_host
    port = config.prefill_port if role == "prefill" else config.decode_port
    maximum_requests = (
        config.maximum_prefill_requests
        if role == "prefill"
        else config.maximum_decode_requests
    )
    role_root = config.runtime_root / role
    argv = (
        str(config.python_executable),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config.model_path),
        "--load-format",
        "safetensors",
        "--served-model-name",
        config.served_model_name,
        "--kt-weight-path",
        str(config.shared_weights.checkpoint_root),
        "--kt-method",
        "AMXINT4",
        "--kt-cpuinfer",
        str(config.cpu_infer_threads),
        "--kt-threadpool-count",
        "2",
        "--kt-numa-nodes",
        *(str(node) for node in config.threadpool_numa_nodes),
        "--kt-num-gpu-experts",
        "0",
        "--kt-max-deferred-experts-per-token",
        "0",
        "--kt-expert-placement-strategy",
        "uniform",
        "--kt-gpu-prefill-token-threshold",
        str(config.stream_prefill_token_threshold),
        "--kt-stream-prefill",
        "--kt-stream-prefill-experts-per-chunk",
        str(config.stream_prefill_experts_per_chunk),
        "--kt-stream-prefill-ring-slots",
        str(config.stream_prefill_ring_slots),
        "--kt-stream-prefill-safety-margin-mb",
        str(config.stream_prefill_safety_margin_mib),
        "--pp-size",
        "1",
        "--tp-size",
        "1",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--base-gpu-id",
        "0",
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(config.kv_handoff.context_length),
        "--max-prefill-tokens",
        str(config.kv_handoff.maximum_prefill_tokens),
        "--max-total-tokens",
        str(config.kv_handoff.maximum_total_tokens),
        "--max-running-requests",
        str(maximum_requests),
        "--chunked-prefill-size",
        str(config.chunked_prefill_size),
        "--page-size",
        str(config.kv_handoff.page_size),
        "--attention-backend",
        config.attention_backend,
        "--kv-cache-dtype",
        config.kv_handoff.kv_cache_dtype,
        "--mem-fraction-static",
        str(config.static_memory_fraction),
        "--moe-a2a-backend",
        "none",
        "--disaggregation-mode",
        role,
        "--disaggregation-transfer-backend",
        config.kv_handoff.transfer_backend,
        "--disaggregation-bootstrap-port",
        str(config.kv_handoff.bootstrap_port),
        *(
            (
                "--disaggregation-ib-device",
                config.kv_handoff.ib_device,
            )
            if config.kv_handoff.ib_device is not None
            else ()
        ),
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-custom-all-reduce",
        "--disable-shared-experts-fusion",
        "--trust-remote-code",
        "--watchdog-timeout",
        "3000",
    )
    environment = (
        ("CUDA_VISIBLE_DEVICES", device.gpu_uuid),
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
        ("SGLANG_KT_PD_ROLE", role),
        ("TMPDIR", str(role_root / "scratch")),
        ("XDG_CACHE_HOME", str(role_root / "cache")),
        ("KT_SHARED_HOST_WEIGHTS", "1"),
        (
            "KT_SHARED_HOST_WEIGHTS_MANIFEST",
            str(config.shared_weights.manifest_path),
        ),
        ("KT_SHARED_HOST_WEIGHTS_CONTENT_ID", config.shared_weights.content_id),
        (
            "KT_SHARED_HOST_WEIGHTS_STATE_DIR",
            str(config.shared_weights.state_directory),
        ),
        *((("KT_AMX_FINE_GRAINED_DECODE", "1"),) if role == "decode" else ()),
    )
    return GLM52PDCommand(
        role=role,
        argv=argv,
        environment=environment,
        working_directory=role_root,
        log_path=role_root / "logs" / "server.log",
    )


def _router_command(config: GLM52PDLaunchConfig) -> GLM52PDCommand:
    role_root = config.runtime_root / "router"
    return GLM52PDCommand(
        role="router",
        argv=(
            str(config.python_executable),
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            f"http://{config.prefill_host}:{config.prefill_port}",
            str(config.kv_handoff.bootstrap_port),
            "--decode",
            f"http://{config.decode_host}:{config.decode_port}",
            "--host",
            config.router_host,
            "--port",
            str(config.router_port),
        ),
        environment=(
            ("SGLANG_KT_PD_ROLE", "router"),
            ("TMPDIR", str(role_root / "scratch")),
            ("XDG_CACHE_HOME", str(role_root / "cache")),
        ),
        working_directory=role_root,
        log_path=role_root / "logs" / "router.log",
    )


def build_glm52_pd_launch_plan(
    config: GLM52PDLaunchConfig,
    *,
    observations: Sequence[GLM52PDGPUObservation] | None = None,
    host_capacity_bytes: int | None = None,
    prepare_directories: bool = False,
) -> GLM52PDLaunchPlan:
    """Validate every static gate and construct exact executable commands."""

    try:
        python_path = config.python_executable.resolve(strict=True)
        model_path = config.model_path.resolve(strict=True)
    except OSError as error:
        raise GLM52PDRuntimeContractError(
            "Python executable or model path does not exist"
        ) from error
    if not python_path.is_file() or not os.access(config.python_executable, os.X_OK):
        raise GLM52PDRuntimeContractError(
            "Python executable must resolve to an executable regular file"
        )
    if model_path != config.model_path or not model_path.is_dir():
        raise GLM52PDRuntimeContractError("model path must be a canonical directory")
    if model_metadata_identity(model_path) != config.model_identity:
        raise GLM52PDRuntimeContractError(
            "model config/index identity does not match the launch contract"
        )

    manifest_nodes = verify_shared_weight_contract(
        config.shared_weights,
        host_capacity_bytes=host_capacity_bytes,
    )
    if manifest_nodes != config.threadpool_numa_nodes:
        numa_message = " ".join(
            (
                "shared-weight NUMA order does not match",
                "KT threadpool order:",
            )
        )
        raise GLM52PDRuntimeContractError(
            "{} {}!={}".format(
                numa_message,
                manifest_nodes,
                config.threadpool_numa_nodes,
            )
        )
    resolved = validate_gpu_contracts(
        config,
        collect_gpu_observations() if observations is None else observations,
    )
    if prepare_directories:
        prepare_runtime_directories(config)

    prefill_capacity = GLM52CapacityContract(
        gpu_ids=(0,),
        max_concurrent_requests=config.maximum_prefill_requests,
        max_inflight_tokens=config.kv_handoff.maximum_prefill_tokens,
        available_device_bytes=(resolved[0].free_bytes,),
        required_device_bytes=(config.prefill_device.required_free_bytes,),
    )
    decode_capacity = GLM52CapacityContract(
        gpu_ids=(1,),
        max_concurrent_requests=config.maximum_decode_requests,
        max_inflight_tokens=config.kv_handoff.maximum_total_tokens,
        available_device_bytes=(resolved[1].free_bytes,),
        required_device_bytes=(config.decode_device.required_free_bytes,),
    )
    policy_contracts = GLM52IntraNodeContracts(
        executors=GLM52ExecutorCompatibility(
            architecture=config.architecture,
            hardware_profile=DWAGON_HARDWARE_PROFILE,
            gpu_ids=(0, 1),
            pipeline_parallel_size=1,
            chunked_prefill_ready=False,
            chunked_prefill_tensor_parallel_size=2,
            distributed_slp_ready=False,
            distributed_slp_tensor_parallel_size=2,
            distributed_slp_admission=None,
            split_slp_ready=True,
            split_slp_tensor_parallel_size=1,
            split_decode_ready=True,
            split_decode_tensor_parallel_size=1,
            split_kv_transfer_ready=True,
        ),
        split_slp_capacity=prefill_capacity,
        split_decode_capacity=decode_capacity,
        shared_host_weights=GLM52SharedHostWeights(
            weight_identity=(
                "glm52-amxint4-sha256:" + config.shared_weights.content_id
            ),
            allocation_identity=(
                "kt-generation-state:"
                + _sha256_bytes(str(config.shared_weights.state_directory).encode())
            ),
            weight_bytes=config.shared_weights.weight_bytes,
            host_capacity_bytes=(
                host_capacity_bytes
                if host_capacity_bytes is not None
                else os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            ),
            safety_margin_bytes=config.shared_weights.host_safety_margin_bytes,
            immutable=True,
            # A static launch plan is deliberately not routable.  The gateway
            # is withheld until live leases independently prove both readers.
            sharing_verified=False,
            prefill_reader_ready=False,
            decode_reader_ready=False,
        ),
    )
    return GLM52PDLaunchPlan(
        config=config,
        prefill=_worker_command(config, "prefill"),
        decode=_worker_command(config, "decode"),
        router=_router_command(config),
        policy_contracts=policy_contracts,
        shared_manifest_numa_nodes=manifest_nodes,
    )


def live_policy_contracts(
    plan: GLM52PDLaunchPlan,
    lease_proof: GLM52PDLeaseProof,
) -> GLM52IntraNodeContracts:
    """Promote static contracts only after one live shared-generation proof."""

    if (
        not lease_proof.generation_id
        or not lease_proof.prefill_lease_process_ids
        or not lease_proof.decode_lease_process_ids
    ):
        raise GLM52PDRuntimeContractError(
            "shared-generation proof does not contain both live readers"
        )
    shared_weights = plan.policy_contracts.shared_host_weights
    if shared_weights is None:
        raise GLM52PDRuntimeContractError(
            "launch plan has no shared-host-weight contract"
        )
    return replace(
        plan.policy_contracts,
        shared_host_weights=replace(
            shared_weights,
            allocation_identity=("kt-generation:" + lease_proof.generation_id),
            sharing_verified=True,
            prefill_reader_ready=True,
            decode_reader_ready=True,
        ),
    )


def _read_process_stat(
    process_id: int,
    *,
    proc_root: Path = _PROC_ROOT,
) -> tuple[int, int]:
    try:
        raw = (proc_root / str(process_id) / "stat").read_text()
        fields = raw.rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[19])
    except (OSError, IndexError, ValueError) as error:
        raise GLM52PDRuntimeContractError(
            f"cannot read process identity for PID {process_id}"
        ) from error


def process_identity(
    role: str,
    process_id: int,
    *,
    proc_root: Path = _PROC_ROOT,
) -> GLM52PDProcessIdentity:
    _, start_time_ticks = _read_process_stat(process_id, proc_root=proc_root)
    return GLM52PDProcessIdentity(role, process_id, start_time_ticks)


def _is_descendant(
    process_id: int,
    ancestor_process_id: int,
    *,
    proc_root: Path,
) -> bool:
    current = process_id
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current == ancestor_process_id:
            return True
        visited.add(current)
        try:
            parent, _ = _read_process_stat(current, proc_root=proc_root)
        except GLM52PDRuntimeContractError:
            return False
        current = parent
    return current == ancestor_process_id


def verify_shared_generation_leases(
    state_directory: Path,
    *,
    content_id: str,
    prefill_root_process_id: int,
    decode_root_process_id: int,
    proc_root: Path = _PROC_ROOT,
) -> GLM52PDLeaseProof:
    """Prove live readers from both process trees share one KT generation."""

    leases_directory = state_directory / "leases"
    try:
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
        lease_paths = tuple(sorted(leases_directory.glob("*.json")))
    except OSError as error:
        raise GLM52PDRuntimeContractError(
            "cannot inspect shared-host-weight leases"
        ) from error
    if not boot_id or not lease_paths:
        raise GLM52PDRuntimeContractError("shared-host-weight lease evidence is empty")

    by_generation: dict[str, dict[str, set[int]]] = {}
    for lease_path in lease_paths:
        try:
            if lease_path.is_symlink() or lease_path.stat().st_size > 64 * 1024:
                continue
            value = cast(object, json.loads(lease_path.read_bytes()))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        lease = cast(dict[str, object], value)
        process_id = lease.get("pid")
        start_time_ticks = lease.get("start_time_ticks")
        generation_id = lease.get("generation_id")
        if (
            lease.get("kind") != "kt_shared_host_weights_lease"
            or lease.get("schema_version") != 1
            or lease.get("content_id") != content_id
            or lease.get("boot_id") != boot_id
            or isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id <= 0
            or isinstance(start_time_ticks, bool)
            or not isinstance(start_time_ticks, int)
            or not isinstance(generation_id, str)
            or not generation_id
        ):
            continue
        try:
            _, observed_start = _read_process_stat(
                process_id,
                proc_root=proc_root,
            )
        except GLM52PDRuntimeContractError:
            continue
        if observed_start != start_time_ticks:
            continue
        roles = by_generation.setdefault(
            generation_id,
            {"prefill": set(), "decode": set()},
        )
        if _is_descendant(
            process_id,
            prefill_root_process_id,
            proc_root=proc_root,
        ):
            roles["prefill"].add(process_id)
        if _is_descendant(
            process_id,
            decode_root_process_id,
            proc_root=proc_root,
        ):
            roles["decode"].add(process_id)

    complete = [
        (generation_id, roles)
        for generation_id, roles in by_generation.items()
        if roles["prefill"] and roles["decode"]
    ]
    if len(complete) != 1:
        raise GLM52PDRuntimeContractError(
            "live prefill and decode readers do not prove one shared generation"
        )
    generation_id, roles = complete[0]
    return GLM52PDLeaseProof(
        generation_id=generation_id,
        prefill_lease_process_ids=tuple(sorted(roles["prefill"])),
        decode_lease_process_ids=tuple(sorted(roles["decode"])),
    )


def _wait_for_health(
    url: str,
    *,
    process: subprocess.Popen[bytes],
    timeout_seconds: int,
    poll_seconds: float,
) -> None:
    parsed_url = urllib.parse.urlsplit(url)
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname is None
        or parsed_url.port is None
    ):
        raise GLM52PDRuntimeContractError(f"invalid health URL: {url}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise GLM52PDRuntimeContractError(
                f"process {process.pid} exited with status {return_code} before health"
            )
        try:
            connection = http.client.HTTPConnection(
                parsed_url.hostname,
                parsed_url.port,
                timeout=5,
            )
            try:
                connection.request("GET", parsed_url.path or "/")
                response = connection.getresponse()
                if 200 <= response.status < 300:
                    return
            finally:
                connection.close()
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(poll_seconds)
    raise GLM52PDRuntimeContractError(
        f"process {process.pid} did not become healthy at {url}"
    )


def _spawn_command(
    command: GLM52PDCommand,
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    environment = os.environ.copy()
    environment.update(command.environment_dict())
    log_file = command.log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command.argv,
            cwd=command.working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return process, log_file


def _stop_processes(
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    timeout_seconds: int,
) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            _ = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            _ = process.wait()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            _ = output.write(_canonical_json_bytes(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_glm52_pd_runtime(
    plan: GLM52PDLaunchPlan,
    *,
    receipt_path: Path,
) -> None:
    """Launch, prove readiness, expose the router, and supervise the topology."""

    prepare_runtime_directories(plan.config)
    processes: list[subprocess.Popen[bytes]] = []
    log_files: list[BinaryIO] = []
    identities: list[GLM52PDProcessIdentity] = []
    try:
        prefill, prefill_log = _spawn_command(plan.prefill)
        processes.append(prefill)
        log_files.append(prefill_log)
        identities.append(process_identity("prefill", prefill.pid))
        _wait_for_health(
            f"http://{plan.config.prefill_host}:{plan.config.prefill_port}/health",
            process=prefill,
            timeout_seconds=plan.config.startup_timeout_seconds,
            poll_seconds=plan.config.health_poll_seconds,
        )

        decode, decode_log = _spawn_command(plan.decode)
        processes.append(decode)
        log_files.append(decode_log)
        identities.append(process_identity("decode", decode.pid))
        _wait_for_health(
            f"http://{plan.config.decode_host}:{plan.config.decode_port}/health",
            process=decode,
            timeout_seconds=plan.config.startup_timeout_seconds,
            poll_seconds=plan.config.health_poll_seconds,
        )

        lease_proof = verify_shared_generation_leases(
            plan.config.shared_weights.state_directory,
            content_id=plan.config.shared_weights.content_id,
            prefill_root_process_id=prefill.pid,
            decode_root_process_id=decode.pid,
        )
        live_contracts = live_policy_contracts(plan, lease_proof)

        router, router_log = _spawn_command(plan.router)
        processes.append(router)
        log_files.append(router_log)
        identities.append(process_identity("router", router.pid))
        _wait_for_health(
            f"http://{plan.config.router_host}:{plan.config.router_port}/health",
            process=router,
            timeout_seconds=plan.config.startup_timeout_seconds,
            poll_seconds=plan.config.health_poll_seconds,
        )

        ready_receipt = {
            **plan.receipt(),
            "status": "ready",
            "processes": [asdict(identity) for identity in identities],
            "shared_generation": asdict(lease_proof),
            "live_split_policy": {
                "split_slp_ready": live_contracts.executors.split_slp_ready,
                "split_decode_ready": live_contracts.executors.split_decode_ready,
                "split_kv_transfer_ready": (
                    live_contracts.executors.split_kv_transfer_ready
                ),
                "shared_weight_readers_verified": (
                    live_contracts.shared_host_weights is not None
                    and live_contracts.shared_host_weights.sharing_verified
                ),
            },
            "router_endpoint": (
                f"http://{plan.config.router_host}:{plan.config.router_port}"
            ),
        }
        _atomic_write_json(receipt_path, ready_receipt)

        while True:
            for process, identity in zip(processes, identities, strict=True):
                return_code = process.poll()
                if return_code is not None:
                    raise GLM52PDRuntimeContractError(
                        f"{identity.role} process exited with status {return_code}"
                    )
                observed = process_identity(identity.role, process.pid)
                if observed != identity:
                    raise GLM52PDRuntimeContractError(
                        f"{identity.role} process identity changed"
                    )
            time.sleep(plan.config.health_poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_processes(
            processes,
            timeout_seconds=plan.config.shutdown_timeout_seconds,
        )
        for log_file in log_files:
            log_file.close()


def _strict_object(
    value: object,
    *,
    description: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GLM52PDRuntimeContractError(f"{description} must be an object")
    result = cast(dict[str, object], value)
    unexpected = set(result) - keys
    missing = keys - set(result)
    if unexpected or missing:
        raise GLM52PDRuntimeContractError(
            "{} keys mismatch: missing={}, unexpected={}".format(
                description,
                sorted(missing),
                sorted(unexpected),
            )
        )
    return result


def _required_str(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise GLM52PDRuntimeContractError(f"{description} must be a non-empty string")
    return value


def _required_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GLM52PDRuntimeContractError(f"{description} must be an integer")
    return value


def _required_float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GLM52PDRuntimeContractError(f"{description} must be numeric")
    return float(value)


def load_glm52_pd_launch_config(path: Path) -> GLM52PDLaunchConfig:
    """Load a strict JSON contract; unknown keys are rejected."""

    try:
        raw = path.read_bytes()
        value = cast(object, json.loads(raw))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GLM52PDRuntimeContractError(
            f"cannot load GLM-5.2 PD launch config: {path}"
        ) from error
    root_keys = {
        "python_executable",
        "model_path",
        "model_identity",
        "served_model_name",
        "architecture",
        "runtime_root",
        "shared_weights",
        "prefill_device",
        "decode_device",
        "kv_handoff",
        "prefill_host",
        "prefill_port",
        "decode_host",
        "decode_port",
        "router_host",
        "router_port",
        "cpu_infer_threads",
        "threadpool_numa_nodes",
        "chunked_prefill_size",
        "stream_prefill_token_threshold",
        "stream_prefill_experts_per_chunk",
        "stream_prefill_ring_slots",
        "stream_prefill_safety_margin_mib",
        "maximum_prefill_requests",
        "maximum_decode_requests",
        "static_memory_fraction",
        "attention_backend",
        "startup_timeout_seconds",
        "health_poll_seconds",
        "shutdown_timeout_seconds",
    }
    root = _strict_object(value, description="launch config", keys=root_keys)

    shared_keys = {
        "checkpoint_root",
        "manifest_path",
        "state_directory",
        "content_id",
        "manifest_sha256",
        "weight_bytes",
        "host_safety_margin_bytes",
    }
    shared = _strict_object(
        root["shared_weights"],
        description="shared_weights",
        keys=shared_keys,
    )
    device_keys = {"gpu_uuid", "numa_node", "required_free_bytes"}

    def parse_device(raw_device: object, description: str) -> GLM52PDDeviceContract:
        device = _strict_object(
            raw_device,
            description=description,
            keys=device_keys,
        )
        return GLM52PDDeviceContract(
            gpu_uuid=_required_str(device["gpu_uuid"], f"{description}.gpu_uuid"),
            numa_node=_required_int(device["numa_node"], f"{description}.numa_node"),
            required_free_bytes=_required_int(
                device["required_free_bytes"],
                f"{description}.required_free_bytes",
            ),
        )

    kv_keys = {
        "transfer_backend",
        "bootstrap_port",
        "context_length",
        "maximum_prefill_tokens",
        "maximum_total_tokens",
        "kv_cache_dtype",
        "page_size",
        "ib_device",
    }
    kv = _strict_object(
        root["kv_handoff"],
        description="kv_handoff",
        keys=kv_keys,
    )
    raw_nodes_value = root["threadpool_numa_nodes"]
    if not isinstance(raw_nodes_value, list):
        raise GLM52PDRuntimeContractError(
            "threadpool_numa_nodes must contain exactly two integers"
        )
    raw_nodes = cast(list[object], raw_nodes_value)
    if len(raw_nodes) != 2:
        raise GLM52PDRuntimeContractError(
            "threadpool_numa_nodes must contain exactly two integers"
        )
    threadpool_numa_nodes = (
        _required_int(raw_nodes[0], "threadpool_numa_nodes[0]"),
        _required_int(raw_nodes[1], "threadpool_numa_nodes[1]"),
    )
    ib_device_value = kv["ib_device"]
    if ib_device_value is not None and not isinstance(ib_device_value, str):
        raise GLM52PDRuntimeContractError(
            "kv_handoff.ib_device must be a string or null"
        )
    transfer_backend = _required_str(
        kv["transfer_backend"],
        "kv_handoff.transfer_backend",
    )
    if transfer_backend not in {"mooncake", "nixl", "mori"}:
        raise GLM52PDRuntimeContractError("kv_handoff.transfer_backend is unsupported")
    return GLM52PDLaunchConfig(
        python_executable=Path(
            _required_str(root["python_executable"], "python_executable")
        ),
        model_path=Path(_required_str(root["model_path"], "model_path")),
        model_identity=_required_str(root["model_identity"], "model_identity"),
        served_model_name=_required_str(
            root["served_model_name"],
            "served_model_name",
        ),
        architecture=_required_str(root["architecture"], "architecture"),
        runtime_root=Path(_required_str(root["runtime_root"], "runtime_root")),
        shared_weights=GLM52PDSharedWeightContract(
            checkpoint_root=Path(
                _required_str(
                    shared["checkpoint_root"],
                    "shared_weights.checkpoint_root",
                )
            ),
            manifest_path=Path(
                _required_str(
                    shared["manifest_path"],
                    "shared_weights.manifest_path",
                )
            ),
            state_directory=Path(
                _required_str(
                    shared["state_directory"],
                    "shared_weights.state_directory",
                )
            ),
            content_id=_required_str(
                shared["content_id"],
                "shared_weights.content_id",
            ),
            manifest_sha256=_required_str(
                shared["manifest_sha256"],
                "shared_weights.manifest_sha256",
            ),
            weight_bytes=_required_int(
                shared["weight_bytes"],
                "shared_weights.weight_bytes",
            ),
            host_safety_margin_bytes=_required_int(
                shared["host_safety_margin_bytes"],
                "shared_weights.host_safety_margin_bytes",
            ),
        ),
        prefill_device=parse_device(root["prefill_device"], "prefill_device"),
        decode_device=parse_device(root["decode_device"], "decode_device"),
        kv_handoff=GLM52PDKVHandoffContract(
            transfer_backend=cast(GLM52PDTransferBackend, transfer_backend),
            bootstrap_port=_required_int(
                kv["bootstrap_port"],
                "kv_handoff.bootstrap_port",
            ),
            context_length=_required_int(
                kv["context_length"],
                "kv_handoff.context_length",
            ),
            maximum_prefill_tokens=_required_int(
                kv["maximum_prefill_tokens"],
                "kv_handoff.maximum_prefill_tokens",
            ),
            maximum_total_tokens=_required_int(
                kv["maximum_total_tokens"],
                "kv_handoff.maximum_total_tokens",
            ),
            kv_cache_dtype=_required_str(
                kv["kv_cache_dtype"],
                "kv_handoff.kv_cache_dtype",
            ),
            page_size=_required_int(kv["page_size"], "kv_handoff.page_size"),
            ib_device=ib_device_value,
        ),
        prefill_host=_required_str(root["prefill_host"], "prefill_host"),
        prefill_port=_required_int(root["prefill_port"], "prefill_port"),
        decode_host=_required_str(root["decode_host"], "decode_host"),
        decode_port=_required_int(root["decode_port"], "decode_port"),
        router_host=_required_str(root["router_host"], "router_host"),
        router_port=_required_int(root["router_port"], "router_port"),
        cpu_infer_threads=_required_int(
            root["cpu_infer_threads"],
            "cpu_infer_threads",
        ),
        threadpool_numa_nodes=threadpool_numa_nodes,
        chunked_prefill_size=_required_int(
            root["chunked_prefill_size"],
            "chunked_prefill_size",
        ),
        stream_prefill_token_threshold=_required_int(
            root["stream_prefill_token_threshold"],
            "stream_prefill_token_threshold",
        ),
        stream_prefill_experts_per_chunk=_required_int(
            root["stream_prefill_experts_per_chunk"],
            "stream_prefill_experts_per_chunk",
        ),
        stream_prefill_ring_slots=_required_int(
            root["stream_prefill_ring_slots"],
            "stream_prefill_ring_slots",
        ),
        stream_prefill_safety_margin_mib=_required_int(
            root["stream_prefill_safety_margin_mib"],
            "stream_prefill_safety_margin_mib",
        ),
        maximum_prefill_requests=_required_int(
            root["maximum_prefill_requests"],
            "maximum_prefill_requests",
        ),
        maximum_decode_requests=_required_int(
            root["maximum_decode_requests"],
            "maximum_decode_requests",
        ),
        static_memory_fraction=_required_float(
            root["static_memory_fraction"],
            "static_memory_fraction",
        ),
        attention_backend=_required_str(
            root["attention_backend"],
            "attention_backend",
        ),
        startup_timeout_seconds=_required_int(
            root["startup_timeout_seconds"],
            "startup_timeout_seconds",
        ),
        health_poll_seconds=_required_float(
            root["health_poll_seconds"],
            "health_poll_seconds",
        ),
        shutdown_timeout_seconds=_required_int(
            root["shutdown_timeout_seconds"],
            "shutdown_timeout_seconds",
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch fail-closed GLM-5.2 TP1 prefill + TP1 decode",
    )
    _ = parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Strict JSON launch contract",
    )
    _ = parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="Ready receipt written only after live shared-generation proof",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts/devices and write a non-ready launch plan",
    )
    return parser


@dataclass
class _CLIArguments:
    config: Path
    receipt: Path
    dry_run: bool


def main(argv: Sequence[str] | None = None) -> int:
    arguments = cast(
        _CLIArguments,
        cast(object, _argument_parser().parse_args(argv)),
    )
    try:
        config = load_glm52_pd_launch_config(arguments.config)
        plan = build_glm52_pd_launch_plan(
            config,
            prepare_directories=not arguments.dry_run,
        )
        if arguments.dry_run:
            _atomic_write_json(
                arguments.receipt,
                {**plan.receipt(), "status": "validated_not_launched"},
            )
            return 0
        run_glm52_pd_runtime(plan, receipt_path=arguments.receipt)
        return 0
    except GLM52PDRuntimeContractError as error:
        print(f"GLM-5.2 PD admission failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
