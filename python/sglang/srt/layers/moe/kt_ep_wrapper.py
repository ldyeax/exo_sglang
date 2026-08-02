# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper, generate_gpu_experts_masks

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)

_KT_SHARED_STAGING_BUFFER: Optional[torch.Tensor] = None
_KT_PROFILE_MASKS: dict[tuple[str, int, int, int], torch.Tensor] = {}
_KT_GPU_EXPERT_MASK_PLANS: dict[str, torch.Tensor] = {}
_KT_CPU_EXPERT_SHARD_PLANS: dict[str, tuple[torch.Tensor, ...]] = {}
_KT_REMOTE_EXPERT_PLANS: dict[str, torch.Tensor] = {}
_KT_REMOTE_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="kt-remote-tier",
)


@dataclass(frozen=True)
class _KTRemotePending:
    prepared_tiers: tuple[tuple, ...]
    futures: tuple[Future[torch.Tensor], ...]
    token_count: int


def _get_hf_config(server_args: "ServerArgs"):
    """Return the HF config across the July and pinned ServerArgs APIs."""
    get_hf_config = getattr(server_args, "get_hf_config", None)
    if get_hf_config is not None:
        return get_hf_config()
    return server_args.get_model_config().hf_config


def get_or_create_shared_staging_buffer(
    *,
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return one persistent GPU staging buffer shared by sequential MoE layers."""
    global _KT_SHARED_STAGING_BUFFER
    requested_device = torch.device(device)
    if _KT_SHARED_STAGING_BUFFER is None:
        _KT_SHARED_STAGING_BUFFER = torch.empty(
            (max_tokens, hidden_size),
            dtype=dtype,
            device=requested_device,
        )
    else:
        buffer = _KT_SHARED_STAGING_BUFFER
        if (
            buffer.shape != (max_tokens, hidden_size)
            or buffer.dtype != dtype
            or buffer.device != requested_device
        ):
            raise RuntimeError(
                "KTransformers staging-buffer specification changed after "
                f"initialization: existing={tuple(buffer.shape)}/{buffer.dtype}/"
                f"{buffer.device}, requested={(max_tokens, hidden_size)}/{dtype}/"
                f"{requested_device}"
            )
    return _KT_SHARED_STAGING_BUFFER


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    gpu_experts_mask: torch.Tensor
    cpuinfer_threads: int
    threadpool_count: int
    numa_nodes: Optional[list[int]]
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None
    weight_key_prefix: Optional[str] = None
    cpu_expert_ids: Optional[torch.Tensor] = None
    global_num_experts: Optional[int] = None
    remote_expert_id_tiers: Optional[tuple[torch.Tensor, ...]] = None
    remote_expert_endpoints: Optional[tuple[str, ...]] = None


def load_remote_expert_plan(
    plan_path: str,
    *,
    num_experts: int,
) -> torch.Tensor:
    """Load and validate the profile-selected sidecar expert IDs."""
    real_path = os.path.realpath(plan_path)
    remote_expert_ids = _KT_REMOTE_EXPERT_PLANS.get(real_path)
    if remote_expert_ids is None:
        loaded_data = torch.load(real_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded_data, dict):
            raise ValueError(
                f"KTransformers remote expert plan must be a dict: {real_path}"
            )
        raw_ids = loaded_data.get("remote_expert_ids")
        if raw_ids is None:
            raise ValueError(
                "KTransformers remote expert plan must contain "
                f"'remote_expert_ids': {real_path}"
            )
        remote_expert_ids = (
            raw_ids.to(device="cpu", dtype=torch.int64).contiguous()
            if isinstance(raw_ids, torch.Tensor)
            else torch.as_tensor(raw_ids, dtype=torch.int64).contiguous()
        )
        if remote_expert_ids.ndim != 2:
            raise ValueError(
                "KTransformers remote expert IDs must have shape "
                f"[layers, experts], got {tuple(remote_expert_ids.shape)}"
            )
        if remote_expert_ids.shape[1] == 0:
            raise ValueError("KTransformers remote expert plan is empty")
        if remote_expert_ids.numel() and (
            int(remote_expert_ids.min().item()) < 0
            or int(remote_expert_ids.max().item()) >= num_experts
        ):
            raise ValueError(
                "KTransformers remote expert plan contains an expert outside "
                f"[0, {num_experts})"
            )
        for layer_idx, layer_ids in enumerate(remote_expert_ids):
            if torch.unique(layer_ids).numel() != layer_ids.numel():
                raise ValueError(
                    "KTransformers remote expert IDs must be unique within "
                    f"layer {layer_idx}"
                )
        _KT_REMOTE_EXPERT_PLANS[real_path] = remote_expert_ids
    return remote_expert_ids


def load_cpu_expert_shard_plan(
    plan_path: str,
    *,
    num_layers: int,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    """Load and validate a lossless, variable-sized KT CPU expert shard."""
    real_path = os.path.realpath(plan_path)
    expert_ids_by_rank = _KT_CPU_EXPERT_SHARD_PLANS.get(real_path)
    if expert_ids_by_rank is None:
        loaded_data = torch.load(real_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded_data, dict):
            raise ValueError(
                f"KTransformers CPU expert shard plan must be a dict: {real_path}"
            )
        raw_shards = loaded_data.get("expert_ids_by_rank")
        if not isinstance(raw_shards, (list, tuple)):
            raise ValueError(
                "KTransformers CPU expert shard plan must contain "
                f"'expert_ids_by_rank': {real_path}"
            )
        expert_ids_by_rank = tuple(
            shard.to(device="cpu", dtype=torch.int64).contiguous()
            if isinstance(shard, torch.Tensor)
            else torch.as_tensor(shard, dtype=torch.int64).contiguous()
            for shard in raw_shards
        )
        _KT_CPU_EXPERT_SHARD_PLANS[real_path] = expert_ids_by_rank

    if len(expert_ids_by_rank) != ep_size:
        raise ValueError(
            "KTransformers CPU expert shard rank count does not match EP: "
            f"plan={len(expert_ids_by_rank)} ep_size={ep_size}"
        )
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"Invalid KTransformers EP rank {ep_rank}/{ep_size}")

    for rank, shard in enumerate(expert_ids_by_rank):
        if shard.ndim != 2 or shard.shape[0] < num_layers:
            raise ValueError(
                "KTransformers CPU expert shard must have shape "
                f"[layers, local_experts]; rank={rank} shape={tuple(shard.shape)} "
                f"required_layers={num_layers}"
            )
        if shard.numel() and (
            int(shard.min().item()) < 0 or int(shard.max().item()) >= num_experts
        ):
            raise ValueError(
                f"KTransformers CPU expert shard rank {rank} contains an "
                f"expert outside [0, {num_experts})"
            )

    expected = torch.arange(num_experts, dtype=torch.int64)
    for layer_idx in range(num_layers):
        assigned = torch.cat(
            [shard[layer_idx] for shard in expert_ids_by_rank], dim=0
        )
        if assigned.numel() != num_experts or not torch.equal(
            torch.sort(assigned).values, expected
        ):
            raise ValueError(
                "KTransformers CPU expert shards must form an exact, disjoint "
                f"partition at layer {layer_idx}"
            )

    selected = expert_ids_by_rank[ep_rank][:num_layers].contiguous()
    logger.info(
        "Loaded KTransformers native CPU expert shard %s: rank=%d/%d "
        "layers=%d local_experts=%d global_experts=%d",
        real_path,
        ep_rank,
        ep_size,
        num_layers,
        selected.shape[1],
        num_experts,
    )
    return selected


def build_logical_to_gpu_index(gpu_experts_mask: torch.Tensor) -> torch.Tensor:
    """Build the compact GPU-weight index for each logical expert."""
    if gpu_experts_mask.ndim != 1 or gpu_experts_mask.dtype != torch.bool:
        raise ValueError(
            "gpu_experts_mask must be a one-dimensional bool tensor, got "
            f"shape={tuple(gpu_experts_mask.shape)} dtype={gpu_experts_mask.dtype}"
        )
    gpu_expert_indices = torch.where(gpu_experts_mask.cpu())[0]
    logical_to_gpu_index = torch.full(
        (gpu_experts_mask.numel(),), -1, dtype=torch.int32, device="cpu"
    )
    logical_to_gpu_index[gpu_expert_indices] = torch.arange(
        gpu_expert_indices.numel(), dtype=torch.int32, device="cpu"
    )
    return logical_to_gpu_index


def partition_remote_local_gpu_experts(
    gpu_experts_mask: torch.Tensor,
    remote_expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint global GPU mask and compact local-CPU expert IDs.

    The sidecar plan and the profile-guided GPU placement both use global
    logical expert IDs. Remote placement wins if the two profiles overlap;
    otherwise an expert would be evaluated twice. The native CPU wrapper gets
    the compact complement and remaps global route IDs before submission.
    """
    if gpu_experts_mask.ndim != 1 or gpu_experts_mask.dtype != torch.bool:
        raise ValueError(
            "gpu_experts_mask must be a one-dimensional bool tensor, got "
            f"shape={tuple(gpu_experts_mask.shape)} dtype={gpu_experts_mask.dtype}"
        )
    remote_expert_ids = remote_expert_ids.to(
        device="cpu", dtype=torch.int64
    ).contiguous()
    if remote_expert_ids.ndim != 1:
        raise ValueError(
            "remote_expert_ids must be one-dimensional, got "
            f"shape={tuple(remote_expert_ids.shape)}"
        )
    num_experts = gpu_experts_mask.numel()
    if remote_expert_ids.numel() and (
        int(remote_expert_ids.min().item()) < 0
        or int(remote_expert_ids.max().item()) >= num_experts
    ):
        raise ValueError(
            "remote_expert_ids contains an expert outside "
            f"[0, {num_experts})"
        )
    if torch.unique(remote_expert_ids).numel() != remote_expert_ids.numel():
        raise ValueError("remote_expert_ids must be unique")

    disjoint_gpu_mask = gpu_experts_mask.to(device="cpu").clone()
    disjoint_gpu_mask[remote_expert_ids] = False
    local_mask = ~(disjoint_gpu_mask.clone())
    local_mask[remote_expert_ids] = False
    local_cpu_expert_ids = torch.arange(
        num_experts, dtype=torch.int64, device="cpu"
    )[local_mask]
    return disjoint_gpu_mask, local_cpu_expert_ids


def combine_remote_expert_tiers(
    remote_expert_id_tiers: tuple[torch.Tensor, ...],
    *,
    num_experts: int,
) -> torch.Tensor:
    """Validate disjoint sidecar tiers and return their global-ID union."""
    normalized_tiers = tuple(
        tier.to(device="cpu", dtype=torch.int64).contiguous()
        for tier in remote_expert_id_tiers
    )
    for tier_idx, tier in enumerate(normalized_tiers):
        if tier.ndim != 1:
            raise ValueError(
                f"remote expert tier {tier_idx} must be one-dimensional, "
                f"got shape={tuple(tier.shape)}"
            )
        if tier.numel() and (
            int(tier.min().item()) < 0
            or int(tier.max().item()) >= num_experts
        ):
            raise ValueError(
                f"remote expert tier {tier_idx} contains an expert outside "
                f"[0, {num_experts})"
            )
        if torch.unique(tier).numel() != tier.numel():
            raise ValueError(
                f"remote expert tier {tier_idx} contains duplicate experts"
            )
    if not normalized_tiers:
        return torch.empty(0, dtype=torch.int64)
    combined = torch.cat(normalized_tiers)
    if torch.unique(combined).numel() != combined.numel():
        raise ValueError("remote expert tiers must be disjoint")
    return combined


def load_profile_guided_gpu_expert_masks(
    profile_path: str,
    *,
    num_layers: int,
    num_experts: int,
    num_gpu_experts_per_layer: int,
) -> torch.Tensor:
    """Select the globally hottest expert slots from an activation profile."""
    cache_key = (
        os.path.realpath(profile_path),
        num_layers,
        num_experts,
        num_gpu_experts_per_layer,
    )
    cached_masks = _KT_PROFILE_MASKS.get(cache_key)
    if cached_masks is not None:
        return cached_masks

    loaded_data = torch.load(profile_path, map_location="cpu", weights_only=True)
    if isinstance(loaded_data, dict):
        if "logical_count" not in loaded_data:
            raise ValueError(
                "KTransformers expert profile does not contain 'logical_count': "
                f"keys={list(loaded_data)}"
            )
        activation_counts = loaded_data["logical_count"]
    else:
        activation_counts = loaded_data
    if not isinstance(activation_counts, torch.Tensor):
        raise TypeError(
            "KTransformers expert profile logical_count must be a tensor, got "
            f"{type(activation_counts).__name__}"
        )
    expected_shape = (num_layers, num_experts)
    if activation_counts.ndim == 3 and tuple(
        activation_counts.shape[1:]
    ) == expected_shape:
        activation_frequency = activation_counts.sum(dim=0).float()
    elif activation_counts.ndim == 2 and tuple(
        activation_counts.shape
    ) == expected_shape:
        activation_frequency = activation_counts.float()
    else:
        raise ValueError(
            "KTransformers expert profile must have shape "
            f"[{num_layers}, {num_experts}] or "
            f"[samples, {num_layers}, {num_experts}], got "
            f"{tuple(activation_counts.shape)}"
        )

    total_gpu_experts = num_gpu_experts_per_layer * num_layers
    if not 0 <= total_gpu_experts <= num_layers * num_experts:
        raise ValueError(
            "Requested GPU expert count is outside the model: "
            f"per_layer={num_gpu_experts_per_layer}, layers={num_layers}, "
            f"experts_per_layer={num_experts}"
        )
    masks = generate_gpu_experts_masks(
        activation_frequency, num_gpu_experts=total_gpu_experts
    ).to(device="cpu", dtype=torch.bool)
    if tuple(masks.shape) != expected_shape:
        raise RuntimeError(
            "kt_kernel returned an invalid expert-mask shape: "
            f"expected={expected_shape}, actual={tuple(masks.shape)}"
        )
    _KT_PROFILE_MASKS[cache_key] = masks
    logger.info(
        "Loaded KTransformers hot-expert profile %s: %d/%d target expert slots on GPU",
        profile_path,
        int(masks.sum().item()),
        masks.numel(),
    )
    return masks


def load_gpu_expert_mask_plan(
    plan_path: str,
    *,
    num_layers: int,
    num_experts: int,
) -> torch.Tensor:
    """Load an explicit, variable-width logical GPU expert mask."""
    real_path = os.path.realpath(plan_path)
    masks = _KT_GPU_EXPERT_MASK_PLANS.get(real_path)
    if masks is None:
        loaded_data = torch.load(real_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded_data, dict):
            raise ValueError(
                f"KTransformers GPU expert mask plan must be a dict: {real_path}"
            )
        raw_masks = loaded_data.get("gpu_experts_mask")
        if raw_masks is None:
            raise ValueError(
                "KTransformers GPU expert mask plan must contain "
                f"'gpu_experts_mask': {real_path}"
            )
        masks = (
            raw_masks.to(device="cpu", dtype=torch.bool).contiguous()
            if isinstance(raw_masks, torch.Tensor)
            else torch.as_tensor(raw_masks, dtype=torch.bool).contiguous()
        )
        expected_shape = (num_layers, num_experts)
        if tuple(masks.shape) != expected_shape:
            raise ValueError(
                "KTransformers GPU expert mask plan must have shape "
                f"{expected_shape}, got {tuple(masks.shape)}"
            )
        _KT_GPU_EXPERT_MASK_PLANS[real_path] = masks
    return masks


def resolve_gpu_experts_mask(
    server_args: "ServerArgs",
    *,
    layer_idx: int,
    weight_key_prefix: Optional[str],
) -> torch.Tensor:
    """Resolve a logical expert mask for a target or DSpark draft layer."""
    hf_config = _get_hf_config(server_args)
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    num_experts = int(
        getattr(
            hf_config,
            "n_routed_experts",
            getattr(hf_config, "num_experts", getattr(hf_config, "num_local_experts", 0)),
        )
    )
    if num_experts <= 0:
        raise ValueError("Unable to determine the model's routed expert count")

    # Draft checkpoint namespaces are independent of target decoder layers, so
    # a target-layer hot-expert profile cannot be applied to them.  Permit an
    # explicit compact prefix of draft experts on GPU; the remaining experts
    # stay native MXFP4 on AMX/AVX-512.  Zero preserves the conservative
    # all-CPU draft default.
    if weight_key_prefix is not None:
        draft_gpu_experts = int(
            os.environ.get("SGLANG_KT_DRAFT_GPU_EXPERTS", "0")
        )
        if not 0 <= draft_gpu_experts <= num_experts:
            raise ValueError(
                "SGLANG_KT_DRAFT_GPU_EXPERTS must be in "
                f"[0, {num_experts}], got {draft_gpu_experts}"
            )
        return (
            torch.arange(num_experts, dtype=torch.int64, device="cpu")
            < draft_gpu_experts
        )

    requested_gpu_experts = server_args.kt_num_gpu_experts
    if requested_gpu_experts is None:
        raise ValueError("--kt-num-gpu-experts is required with --kt-weight-path")
    if requested_gpu_experts == -1:
        return torch.ones(num_experts, dtype=torch.bool, device="cpu")
    if not 0 <= requested_gpu_experts <= num_experts:
        raise ValueError(
            f"--kt-num-gpu-experts must be in [-1, {num_experts}], got "
            f"{requested_gpu_experts}"
        )

    mask_plan_path = os.environ.get("SGLANG_KT_GPU_EXPERT_MASK_PLAN")
    profile_path = os.environ.get("SGLANG_KT_EXPERT_PROFILE")
    if mask_plan_path and profile_path:
        raise ValueError(
            "SGLANG_KT_GPU_EXPERT_MASK_PLAN and SGLANG_KT_EXPERT_PROFILE "
            "are mutually exclusive"
        )
    if mask_plan_path:
        masks = load_gpu_expert_mask_plan(
            mask_plan_path,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        if not 0 <= layer_idx < masks.shape[0]:
            raise ValueError(
                f"Target layer {layer_idx} is outside GPU mask plan with "
                f"{masks.shape[0]} layers"
            )
        return masks[layer_idx].clone()
    if profile_path:
        masks = load_profile_guided_gpu_expert_masks(
            profile_path,
            num_layers=num_layers,
            num_experts=num_experts,
            num_gpu_experts_per_layer=requested_gpu_experts,
        )
        if not 0 <= layer_idx < masks.shape[0]:
            raise ValueError(
                f"Target layer {layer_idx} is outside profile with "
                f"{masks.shape[0]} layers"
            )
        return masks[layer_idx].clone()

    return (
        torch.arange(num_experts, dtype=torch.int64, device="cpu")
        < requested_gpu_experts
    )


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int, prefix: str = ""
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if server_args.kt_weight_path is None:
        return None

    # Try to get num_layers from model config
    num_layers = None
    try:
        hf_config = _get_hf_config(server_args)
        num_layers = getattr(hf_config, "num_hidden_layers", None)
    except Exception:
        # If we can't get the config, num_layers will be None
        pass

    dspark_stage_match = re.search(r"(?:^|\.)stages\.(\d+)(?:\.|$)", prefix)
    weight_key_prefix = (
        f"mtp.{int(dspark_stage_match.group(1))}"
        if dspark_stage_match is not None
        else None
    )
    hf_config = _get_hf_config(server_args)
    global_num_experts = int(
        getattr(
            hf_config,
            "n_routed_experts",
            getattr(hf_config, "num_experts", getattr(hf_config, "num_local_experts", 0)),
        )
    )
    cpu_expert_ids = None
    remote_expert_id_tiers = None
    remote_expert_endpoints = None
    legacy_remote_endpoint = os.environ.get("SGLANG_KT_REMOTE_EXPERT_ENDPOINT")
    legacy_remote_plan_path = os.environ.get("SGLANG_KT_REMOTE_EXPERT_PLAN")
    multi_remote_endpoints = os.environ.get("SGLANG_KT_REMOTE_EXPERT_ENDPOINTS")
    multi_remote_plan_paths = os.environ.get("SGLANG_KT_REMOTE_EXPERT_PLANS")
    if (legacy_remote_endpoint or legacy_remote_plan_path) and (
        multi_remote_endpoints or multi_remote_plan_paths
    ):
        raise ValueError(
            "Use either the singular or plural KTransformers remote expert "
            "environment variables, not both"
        )
    if bool(legacy_remote_plan_path) != bool(legacy_remote_endpoint):
        raise ValueError(
            "SGLANG_KT_REMOTE_EXPERT_PLAN and "
            "SGLANG_KT_REMOTE_EXPERT_ENDPOINT must be set together"
        )
    if bool(multi_remote_plan_paths) != bool(multi_remote_endpoints):
        raise ValueError(
            "SGLANG_KT_REMOTE_EXPERT_PLANS and "
            "SGLANG_KT_REMOTE_EXPERT_ENDPOINTS must be set together"
        )
    if legacy_remote_plan_path:
        remote_plan_paths = (legacy_remote_plan_path,)
        configured_remote_endpoints = (legacy_remote_endpoint,)
    elif multi_remote_plan_paths:
        remote_plan_paths = tuple(
            value.strip()
            for value in multi_remote_plan_paths.split(";")
            if value.strip()
        )
        configured_remote_endpoints = tuple(
            value.strip()
            for value in multi_remote_endpoints.split(";")
            if value.strip()
        )
        if not remote_plan_paths or len(remote_plan_paths) != len(
            configured_remote_endpoints
        ):
            raise ValueError(
                "Plural KTransformers remote plans and endpoints must contain "
                "the same nonzero number of semicolon-separated values"
            )
    else:
        remote_plan_paths = ()
        configured_remote_endpoints = ()
    shard_plan_path = os.environ.get("SGLANG_KT_CPU_EXPERT_SHARD_PLAN")
    if remote_plan_paths and shard_plan_path:
        raise ValueError(
            "KTransformers remote and EP CPU expert plans are mutually exclusive"
        )
    if remote_plan_paths and weight_key_prefix is None:
        selected_tiers = []
        selected_endpoints = []
        for remote_plan_path, remote_endpoint in zip(
            remote_plan_paths, configured_remote_endpoints, strict=True
        ):
            remote_ids_by_layer = load_remote_expert_plan(
                remote_plan_path,
                num_experts=global_num_experts,
            )
            if layer_idx < remote_ids_by_layer.shape[0]:
                selected_tiers.append(remote_ids_by_layer[layer_idx].clone())
                selected_endpoints.append(remote_endpoint)
        if selected_tiers:
            remote_expert_id_tiers = tuple(selected_tiers)
            remote_expert_endpoints = tuple(selected_endpoints)
            combined_remote_expert_ids = combine_remote_expert_tiers(
                remote_expert_id_tiers,
                num_experts=global_num_experts,
            )
            requested_gpu_experts_mask = resolve_gpu_experts_mask(
                server_args,
                layer_idx=layer_idx,
                weight_key_prefix=weight_key_prefix,
            )
            gpu_experts_mask, cpu_expert_ids = partition_remote_local_gpu_experts(
                requested_gpu_experts_mask,
                combined_remote_expert_ids,
            )
            remote_gpu_overlap = int(
                requested_gpu_experts_mask[
                    combined_remote_expert_ids
                ].sum().item()
            )
            logger.info(
                "KTransformers native tiered shard layer=%d local=%d "
                "remote=%d remote_tiers=%d gpu=%d remote_gpu_overlap=%d "
                "endpoints=%s",
                layer_idx,
                cpu_expert_ids.numel(),
                combined_remote_expert_ids.numel(),
                len(remote_expert_id_tiers),
                int(gpu_experts_mask.sum().item()),
                remote_gpu_overlap,
                remote_expert_endpoints,
            )
        else:
            gpu_experts_mask = resolve_gpu_experts_mask(
                server_args,
                layer_idx=layer_idx,
                weight_key_prefix=weight_key_prefix,
            )
    elif shard_plan_path:
        if server_args.kt_num_gpu_experts != 0:
            raise ValueError(
                "SGLANG_KT_CPU_EXPERT_SHARD_PLAN currently requires "
                "--kt-num-gpu-experts 0"
            )
        if int(os.environ.get("SGLANG_KT_DRAFT_GPU_EXPERTS", "0")) != 0:
            raise ValueError(
                "SGLANG_KT_CPU_EXPERT_SHARD_PLAN currently requires "
                "SGLANG_KT_DRAFT_GPU_EXPERTS=0"
            )
        parallel = get_parallel()
        cpu_expert_ids_by_layer = load_cpu_expert_shard_plan(
            shard_plan_path,
            num_layers=int(num_layers),
            num_experts=global_num_experts,
            ep_size=parallel.moe_ep_size,
            ep_rank=parallel.moe_ep_rank,
        )
        cpu_expert_ids = cpu_expert_ids_by_layer[layer_idx].clone()
        gpu_experts_mask = torch.zeros(
            cpu_expert_ids.numel(), dtype=torch.bool, device="cpu"
        )
    else:
        gpu_experts_mask = resolve_gpu_experts_mask(
            server_args,
            layer_idx=layer_idx,
            weight_key_prefix=weight_key_prefix,
        )

    # A GPU-only tier must compact native host storage just like a remote
    # tier. Keeping a full logical-size native wrapper merely to mask hot
    # experts at execution time reserves their AMX/AVX weight buffers anyway
    # and prevents GPU placement from freeing host capacity.
    if cpu_expert_ids is None and bool(gpu_experts_mask.any().item()):
        cpu_gpu_experts_mask = gpu_experts_mask.to(device="cpu")
        cpu_expert_ids = torch.arange(
            global_num_experts, dtype=torch.int64, device="cpu"
        )[~cpu_gpu_experts_mask]

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=int(gpu_experts_mask.sum().item()),
        gpu_experts_mask=gpu_experts_mask,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        numa_nodes=server_args.kt_numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        weight_key_prefix=weight_key_prefix,
        cpu_expert_ids=cpu_expert_ids,
        global_num_experts=global_num_experts,
        remote_expert_id_tiers=remote_expert_id_tiers,
        remote_expert_endpoints=remote_expert_endpoints,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    # Do not mutate topk_ids in place. The asynchronous KTransformers CPU job
    # submitted immediately before this function retains the same tensor. An
    # in-place mask can race with its CUDA-to-host copy and make the CPU worker
    # observe -1 instead of the original logical expert id.
    return torch.where(topk_ids < num_gpu_experts, topk_ids, -1)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """Mask CPU experts and compact arbitrary logical GPU expert IDs."""
    valid = (topk_ids >= 0) & (topk_ids < gpu_experts_mask.numel())
    safe_ids = torch.where(valid, topk_ids, 0)
    is_gpu_expert = valid & gpu_experts_mask[safe_ids]
    return torch.where(is_gpu_expert, logical_to_gpu_index[safe_ids], -1)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def remap_global_expert_ids(
    topk_ids: torch.Tensor,
    global_to_local: torch.Tensor,
) -> torch.Tensor:
    """Map global routed IDs to a sparse native-CPU shard."""
    valid = (topk_ids >= 0) & (topk_ids < global_to_local.numel())
    safe_ids = torch.where(valid, topk_ids, 0)
    return torch.where(valid, global_to_local[safe_ids], -1)


def select_remote_token_rows(
    topk_ids: torch.Tensor,
    remote_expert_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one bool per token indicating ownership by a remote tier."""
    flat_ids = topk_ids.reshape(-1, topk_ids.shape[-1])
    valid = (flat_ids >= 0) & (flat_ids < remote_expert_mask.numel())
    safe_ids = torch.where(valid, flat_ids, 0)
    return (valid & remote_expert_mask[safe_ids]).any(dim=-1)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask
        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())
        self.logical_to_gpu_index = build_logical_to_gpu_index(
            self.gpu_experts_mask
        )
        self.gpu_experts_mask_cuda: Optional[torch.Tensor] = None
        self.logical_to_gpu_index_cuda: Optional[torch.Tensor] = None
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        # CPU experts are replicated across MoE-TP ranks but sharded across
        # MoE-EP ranks.  moe_tp_rank therefore selects the single CPU owner in
        # ordinary TP while allowing every EP rank to execute its local shard.
        self.tp_rank = get_parallel().moe_tp_rank
        self.cpu_expert_ids = kt_config.cpu_expert_ids
        self.remote_expert_id_tiers = kt_config.remote_expert_id_tiers
        self.remote_expert_endpoints = kt_config.remote_expert_endpoints
        self.global_num_experts = (
            kt_config.global_num_experts
            if kt_config.global_num_experts is not None
            else int(kt_config.gpu_experts_mask.numel())
        )

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._sync_done_event: Optional[torch.cuda.Event] = None
        self._shared_staging_buffer: Optional[torch.Tensor] = None
        self.global_to_local_expert_mapping_cuda: Optional[torch.Tensor] = None
        self.remote_expert_masks_cuda: tuple[torch.Tensor, ...] = ()
        self.remote_clients = ()
        self._remote_pending_output: Optional[_KTRemotePending] = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        native_num_experts = num_experts
        native_gpu_experts_mask = self.gpu_experts_mask
        if self.cpu_expert_ids is not None:
            # GPU/remote routing stays in global logical ID space. The native
            # wrapper owns only the compact local-CPU complement and therefore
            # needs a separate all-CPU mask in that compact space.
            native_num_experts = int(self.cpu_expert_ids.numel())
            native_gpu_experts_mask = torch.zeros(
                native_num_experts, dtype=torch.bool, device="cpu"
            )
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        target_device = next(layer.parameters()).device
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(
            device=target_device
        )
        if self.cpu_expert_ids is not None:
            global_to_local = torch.full(
                (self.global_num_experts,), -1, dtype=torch.int32, device="cpu"
            )
            global_to_local[self.cpu_expert_ids] = torch.arange(
                self.cpu_expert_ids.numel(), dtype=torch.int32, device="cpu"
            )
            self.global_to_local_expert_mapping_cuda = global_to_local.to(
                device=target_device, non_blocking=True
            )
        if self.remote_expert_id_tiers is not None:
            if (
                self.cpu_expert_ids is None
                or self.remote_expert_endpoints is None
                or len(self.remote_expert_id_tiers)
                != len(self.remote_expert_endpoints)
            ):
                raise ValueError(
                    "Remote KTransformers expert tiers require a local "
                    "complement and one endpoint per tier"
                )
            self.remote_expert_masks_cuda = tuple(
                torch.zeros(
                    self.global_num_experts, dtype=torch.bool, device="cpu"
                )
                .scatter_(
                    0,
                    remote_expert_ids,
                    torch.ones_like(remote_expert_ids, dtype=torch.bool),
                )
                .to(device=target_device, non_blocking=True)
                for remote_expert_ids in self.remote_expert_id_tiers
            )
            from sglang.srt.layers.moe.kt_remote_sidecar import (
                KTExpertSidecarClient,
            )

            self.remote_clients = tuple(
                KTExpertSidecarClient.get(endpoint)
                for endpoint in self.remote_expert_endpoints
            )

        if self.tp_rank == 0:
            self._cpu_stream = torch.cuda.Stream(device=target_device)
            self._sync_done_event = torch.cuda.Event()
            self._shared_staging_buffer = get_or_create_shared_staging_buffer(
                max_tokens=self.kt_config.chunked_prefill_size,
                hidden_size=hidden_size,
                dtype=params_dtype,
                device=target_device,
            )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            moe_runner_config = getattr(layer, "moe_runner_config", None)
            swiglu_limit = float(
                getattr(moe_runner_config, "swiglu_limit", 0.0) or 0.0
            )
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=native_num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=native_gpu_experts_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
                numa_nodes=(
                    self.kt_config.numa_nodes
                    if self.kt_config.numa_nodes is not None
                    else list(range(self.kt_config.threadpool_count))
                ),
                swiglu_limit=swiglu_limit,
            )
            if self.cpu_expert_ids is not None:
                self.wrapper.weight_expert_ids = self.cpu_expert_ids
            if self.kt_config.weight_key_prefix is not None:
                self.wrapper.weight_key_prefix = self.kt_config.weight_key_prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if self.num_gpu_experts > 0 and hasattr(
            self.gpu_method, "process_weights_after_loading"
        ):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            if self.cpu_expert_ids is not None:
                physical_to_logical_map_cpu = torch.arange(
                    self.cpu_expert_ids.numel(), dtype=torch.int64, device="cpu"
                )
            else:
                physical_to_logical_map_cpu = (
                    get_global_expert_location_metadata()
                    .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                    .contiguous()
                )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        global_topk_ids = topk_ids
        if self.global_to_local_expert_mapping_cuda is not None:
            topk_ids = remap_global_expert_ids(
                topk_ids, self.global_to_local_expert_mapping_cuda
            )

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )
        self._remote_pending_output = self._submit_remote_if_selected(
            x=x,
            global_topk_ids=global_topk_ids,
            topk_weights=topk_weights,
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        output = self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )
        if self._remote_pending_output is not None:
            output = output + self._finish_remote(
                self._remote_pending_output,
                x=x,
            )
            self._remote_pending_output = None
        return output

    def _submit_remote_if_selected(
        self,
        *,
        x: torch.Tensor,
        global_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Optional[_KTRemotePending]:
        if not self.remote_clients:
            return None
        flat_x = x.reshape(-1, x.shape[-1])
        flat_topk_ids = global_topk_ids.reshape(
            -1, global_topk_ids.shape[-1]
        )
        flat_topk_weights = topk_weights.reshape_as(flat_topk_ids)
        prepared_tiers = []
        attn_backend = get_attn_backend()
        dsv4_workspace = getattr(
            attn_backend, "_dsv4_main_q_workspace", None
        )
        workspace_cursor = getattr(
            attn_backend, "_dsv4_mlp_workspace_live_elements", None
        )
        use_dsv4_workspace = (
            flat_x.shape[0] >= 512
            and dsv4_workspace is not None
            and workspace_cursor is not None
            and dsv4_workspace.dtype == flat_x.dtype
            and dsv4_workspace.device == flat_x.device
            and dsv4_workspace.is_contiguous()
        )
        workspace_limit = (
            dsv4_workspace.numel() if use_dsv4_workspace else 0
        )
        if (
            use_dsv4_workspace
            and flat_x.untyped_storage().data_ptr()
            == dsv4_workspace.untyped_storage().data_ptr()
        ):
            # mHC pre can keep the MoE input in the final contiguous suffix
            # of this workspace. Remote compaction owns only the dead prefix;
            # treating the whole allocation as free would corrupt the input
            # before routing and the local/GPU expert paths consume it.
            input_start_bytes = flat_x.data_ptr() - dsv4_workspace.data_ptr()
            input_end_bytes = (
                input_start_bytes + flat_x.numel() * flat_x.element_size()
            )
            workspace_bytes = (
                dsv4_workspace.numel() * dsv4_workspace.element_size()
            )
            if (
                input_start_bytes < 0
                or input_end_bytes > workspace_bytes
                or input_start_bytes % dsv4_workspace.element_size() != 0
                or not flat_x.is_contiguous()
            ):
                raise RuntimeError(
                    "invalid DSV4 MoE input view in shared main-Q workspace: "
                    f"start={input_start_bytes}, end={input_end_bytes}, "
                    f"workspace={workspace_bytes}, "
                    f"contiguous={flat_x.is_contiguous()}"
                )
            workspace_limit = (
                input_start_bytes // dsv4_workspace.element_size()
            )
            if workspace_cursor > workspace_limit:
                raise RuntimeError(
                    "DSV4 MLP workspace cursor already overlaps protected "
                    f"MoE input: cursor={workspace_cursor}, "
                    f"limit={workspace_limit}"
                )
        for remote_client, remote_expert_mask_cuda in zip(
            self.remote_clients, self.remote_expert_masks_cuda, strict=True
        ):
            selected_rows = select_remote_token_rows(
                flat_topk_ids,
                remote_expert_mask_cuda,
            )
            selected_indices = torch.where(selected_rows)[0]
            if selected_indices.numel() == 0:
                continue
            compact_shape = (selected_indices.numel(), flat_x.shape[-1])
            compact_elements = selected_indices.numel() * flat_x.shape[-1]
            if (
                use_dsv4_workspace
                and workspace_cursor + compact_elements
                <= workspace_limit
            ):
                compact_x = dsv4_workspace.view(-1)[
                    workspace_cursor : workspace_cursor + compact_elements
                ].view(compact_shape)
                torch.index_select(
                    flat_x,
                    0,
                    selected_indices,
                    out=compact_x,
                )
                workspace_cursor += compact_elements
            else:
                compact_x = flat_x[selected_indices]
            prepared_tiers.append(
                (
                    remote_client,
                    selected_indices,
                    compact_x,
                    flat_topk_ids[selected_indices],
                    flat_topk_weights[selected_indices],
                )
            )
        if use_dsv4_workspace:
            setattr(
                attn_backend,
                "_dsv4_mlp_workspace_live_elements",
                workspace_cursor,
            )

        if not prepared_tiers:
            return None

        # CUDA's current stream is thread-local. The compact tensors above
        # were produced on the scheduler's inference stream, while executor
        # workers begin on their device's default stream. Record one event so
        # every transport worker observes completed compaction without a
        # process-wide device synchronize. Always use the executor, including
        # for a single selected tier: returning the pending handle here is
        # what overlaps transport and remote AMX/AVX-512 work with the local
        # CPU shard and hot-expert GPU kernel.
        compact_ready_event = None
        if flat_x.is_cuda:
            compact_ready_event = torch.cuda.Event()
            compact_ready_event.record(torch.cuda.current_stream(flat_x.device))

        def run_prepared_tier(prepared_tier):
            if compact_ready_event is not None:
                compact_ready_event.synchronize()
            remote_client, _, compact_x, compact_ids, compact_weights = (
                prepared_tier
            )
            return remote_client.forward(
                layer_idx=self.kt_config.layer_idx,
                hidden_states=compact_x,
                topk_ids=compact_ids,
                topk_weights=compact_weights,
                return_cpu=True,
            )

        futures = tuple(
            _KT_REMOTE_EXECUTOR.submit(run_prepared_tier, prepared_tier)
            for prepared_tier in prepared_tiers
        )
        return _KTRemotePending(
            prepared_tiers=tuple(prepared_tiers),
            futures=futures,
            token_count=flat_x.shape[0],
        )

    def _finish_remote(
        self,
        pending: _KTRemotePending,
        *,
        x: torch.Tensor,
    ) -> torch.Tensor:
        compact_outputs = tuple(future.result() for future in pending.futures)
        combined_output = None
        for prepared_tier, compact_output in zip(
            pending.prepared_tiers,
            compact_outputs,
            strict=True,
        ):
            _, selected_indices, _, _, _ = prepared_tier
            if compact_output.device.type == "cpu":
                compact_output = compact_output.to(
                    device=x.device,
                    dtype=x.dtype,
                )
            if selected_indices.numel() == pending.token_count:
                tier_output = compact_output.view_as(x)
            else:
                tier_output = torch.zeros(
                    (pending.token_count, x.shape[-1]),
                    dtype=x.dtype,
                    device=x.device,
                ).index_copy(
                    0,
                    selected_indices,
                    compact_output.reshape(-1, compact_output.shape[-1]),
                ).view_as(x)
            combined_output = (
                tier_output
                if combined_output is None
                else combined_output + tier_output
            )
        return combined_output

    def _accumulate_remote(
        self,
        pending: _KTRemotePending,
        *,
        output: torch.Tensor,
        staging_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """Stream pinned remote rows through dead staging into the accumulator."""
        flat_output = output.reshape(-1, output.shape[-1])
        flat_staging = staging_buffer.reshape(-1, staging_buffer.shape[-1])
        transfer_rows = int(
            os.environ.get("SGLANG_KT_REMOTE_MERGE_CHUNK_TOKENS", "64")
        )
        if transfer_rows <= 0:
            raise ValueError(
                "SGLANG_KT_REMOTE_MERGE_CHUNK_TOKENS must be positive"
            )
        for prepared_tier, future in zip(
            pending.prepared_tiers,
            pending.futures,
            strict=True,
        ):
            compact_output = future.result()
            _, selected_indices, _, _, _ = prepared_tier
            compact_output = compact_output.reshape(
                -1, compact_output.shape[-1]
            )
            if compact_output.device.type != "cpu":
                raise RuntimeError(
                    "Streamed KTransformers remote merge requires CPU output"
                )
            for token_start in range(
                0, compact_output.shape[0], transfer_rows
            ):
                token_end = min(
                    token_start + transfer_rows,
                    compact_output.shape[0],
                )
                row_count = token_end - token_start
                transfer = flat_staging[:row_count]
                transfer.copy_(
                    compact_output[token_start:token_end],
                    non_blocking=True,
                )
                flat_output.index_add_(
                    0,
                    selected_indices[token_start:token_end],
                    transfer,
                )
        return output

    def _run_remote_if_selected(
        self,
        *,
        x: torch.Tensor,
        global_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compatibility helper that submits and immediately joins remote work."""
        pending = self._submit_remote_if_selected(
            x=x,
            global_topk_ids=global_topk_ids,
            topk_weights=topk_weights,
        )
        if pending is None:
            return None
        return self._finish_remote(pending, x=x)

    @eager_on_graph(True)
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        debug_sync = os.environ.get("SGLANG_KT_GRAPH_DEBUG_SYNC") == "1"
        if debug_sync:
            torch.cuda.synchronize(x.device)
            logger.info(
                "KTransformers graph debug: layer=%d stage=entry synchronized",
                self.kt_config.layer_idx,
            )

        # Step 1: Submit CPU expert computation (non-blocking)
        staging_buffer = None
        if self.tp_rank == 0:
            assert self.wrapper is not None
            assert self._cpu_stream is not None
            assert self._shared_staging_buffer is not None
            if x.shape[0] > self._shared_staging_buffer.shape[0]:
                raise RuntimeError(
                    f"KTransformers batch has {x.shape[0]} tokens, exceeding "
                    f"the staging capacity {self._shared_staging_buffer.shape[0]}"
                )
            staging_buffer = self._shared_staging_buffer[: x.shape[0]]
            staging_buffer.copy_(x, non_blocking=True)
            self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            topk_weights, topk_ids, _ = topk_output
            global_topk_ids = topk_ids
            if self.global_to_local_expert_mapping_cuda is not None:
                topk_ids = remap_global_expert_ids(
                    topk_ids, self.global_to_local_expert_mapping_cuda
                )
            with torch.cuda.stream(self._cpu_stream):
                self.wrapper.submit_forward(
                    staging_buffer,
                    topk_ids,
                    topk_weights,
                    self._cpu_stream.cuda_stream,
                )
            remote_pending = self._submit_remote_if_selected(
                x=x,
                global_topk_ids=global_topk_ids,
                topk_weights=topk_weights,
            )
        else:
            remote_pending = None

        # Step 2: mask CPU expert IDs and map logical GPU IDs to compact weights.
        topk_ids = topk_output.topk_ids
        assert self.gpu_experts_mask_cuda is not None
        assert self.logical_to_gpu_index_cuda is not None
        masked_topk_ids = mask_and_remap_expert_ids(
            topk_ids,
            self.gpu_experts_mask_cuda,
            self.logical_to_gpu_index_cuda,
        )

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation
        if self.num_gpu_experts == 0:
            # A zero-expert GPU weight slice is not a valid input to every
            # wrapped MoE backend. In this mode KTransformers owns every routed
            # expert, so avoid launching a vacuous GPU kernel. For a large
            # DSV4 prefill, the original activation has already been copied to
            # the native staging buffer and every remote compact tensor before
            # this point. Reuse that now-dead input for the zero accumulator,
            # matching the compact GPU path's in-place route reduction.
            dsv4_workspace = getattr(
                get_attn_backend(), "_dsv4_main_q_workspace", None
            )
            reuse_input = (
                os.environ.get("SGLANG_DSV4_INPLACE_ZERO_GPU_MOE") == "1"
                and x.shape[0] >= 512
                and dsv4_workspace is not None
                and getattr(self.moe_runner_config, "inplace", False)
            )
            output = x.zero_() if reuse_input else torch.zeros_like(x)
        else:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states
        if debug_sync:
            torch.cuda.synchronize(x.device)
            logger.info(
                "KTransformers graph debug: layer=%d stage=gpu synchronized",
                self.kt_config.layer_idx,
            )

        # Step 4: Synchronize CPU results and merge with GPU results
        if self.tp_rank == 0:
            assert self.wrapper is not None
            assert self._cpu_stream is not None
            assert self._sync_done_event is not None
            assert staging_buffer is not None
            with torch.cuda.stream(self._cpu_stream):
                cpu_output = self.wrapper.sync_forward(
                    staging_buffer,
                    self._cpu_stream.cuda_stream,
                    output_tensor=staging_buffer,
                )
                self._sync_done_event.record(self._cpu_stream)
            torch.cuda.current_stream(x.device).wait_event(self._sync_done_event)
            output.add_(cpu_output)
        if remote_pending is not None:
            assert staging_buffer is not None
            self._accumulate_remote(
                remote_pending,
                output=output,
                staging_buffer=staging_buffer,
            )
        if debug_sync:
            torch.cuda.synchronize(x.device)
            logger.info(
                "KTransformers graph debug: layer=%d stage=merge synchronized",
                self.kt_config.layer_idx,
            )

        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
