from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Literal, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsv4 import (
    clear_unaccepted_c128_draft_states,
    fused_k_norm_rope_flashmla,
    fused_store_cache,
)
from sglang.kernels.ops.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.dsv4_kv_cache_dtype import (
    dsv4_kv_cache_dtype_name,
    dsv4_supports_int4_kv_storage,
    dsv4_supports_oscar_int2_kv_storage,
    dsv4_supports_selective_c128_bf16_storage,
    dsv4_uses_ampere_fp8_kv_storage,
    format_dsv4_device_capability,
    get_dsv4_device_capability,
    resolve_dsv4_kv_cache_dtype,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.runtime_context import get_exec, get_server_args, get_spec
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def _sha256_regular_file(path: Path, *, label: str) -> str:
    """Hash one provenance file without following an artifact symlink."""

    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_oscar_admission_receipt(
    path: Path,
    *,
    artifact_path: Path,
    artifact_sha256: str,
    checkpoint_path: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Validate the one-per-launch full-checkpoint OSCAR admission receipt."""

    receipt_sha256 = _sha256_regular_file(path, label="OSCAR admission receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OSCAR admission receipt is not valid UTF-8 JSON") from error
    if not isinstance(receipt, dict):
        raise TypeError("OSCAR admission receipt must contain one JSON object")
    expected_keys = {
        "format",
        "format_version",
        "admitted",
        "model_id",
        "artifact_path",
        "artifact_file_sha256",
        "artifact_provenance_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_sha256",
        "checkpoint_fingerprint_path",
        "checkpoint_fingerprint_sha256",
        "validation_policy",
        "admission_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("OSCAR admission receipt fields do not match version 1")
    if (
        receipt.get("format") != "dsv4-oscar-int2-admission"
        or receipt.get("format_version") != 1
        or receipt.get("admitted") is not True
        or receipt.get("validation_policy")
        != "rehash-config-index-and-all-referenced-shards-v1"
    ):
        raise ValueError("OSCAR admission receipt policy/version is not admissible")
    for field in (
        "artifact_file_sha256",
        "artifact_provenance_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "checkpoint_fingerprint_sha256",
        "admission_sha256",
    ):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"OSCAR admission receipt {field} is not SHA-256")
    if not isinstance(receipt.get("model_id"), str) or not receipt["model_id"]:
        raise ValueError("OSCAR admission receipt model_id is empty")
    if receipt.get("artifact_path") != str(artifact_path.resolve()):
        raise ValueError("OSCAR admission receipt names a different artifact")
    if receipt.get("artifact_file_sha256") != artifact_sha256:
        raise ValueError("OSCAR artifact changed after full-checkpoint admission")
    if receipt.get("checkpoint_path") != str(checkpoint_path.resolve()):
        raise ValueError("OSCAR admission receipt names a different checkpoint")
    if receipt.get("config_sha256") != config_sha256:
        raise ValueError("OSCAR model config changed after full-checkpoint admission")
    fingerprint_path = receipt.get("checkpoint_fingerprint_path")
    if (
        not isinstance(fingerprint_path, str)
        or not Path(fingerprint_path).is_absolute()
    ):
        raise ValueError("OSCAR admission fingerprint path must be absolute")
    actual_fingerprint_sha256 = _sha256_regular_file(
        Path(fingerprint_path), label="OSCAR checkpoint fingerprint"
    )
    if actual_fingerprint_sha256 != receipt["checkpoint_fingerprint_sha256"]:
        raise ValueError("OSCAR checkpoint fingerprint changed after admission")
    declared_admission_sha256 = receipt["admission_sha256"]
    admission_payload = {
        key: value for key, value in receipt.items() if key != "admission_sha256"
    }
    actual_admission_sha256 = sha256(
        json.dumps(
            admission_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if actual_admission_sha256 != declared_admission_sha256:
        raise ValueError("OSCAR admission_sha256 does not match its receipt content")
    return receipt, receipt_sha256


class _DSV4OscarKVPoolContract(NamedTuple):
    consumer_role: Literal["target_compressed", "draft_swa_only"]
    compressed_layer_ids: frozenset[int]
    c4_layer_ids: frozenset[int]
    retain_runtime_calibrations: bool


def _resolve_dsv4_oscar_kv_pool_contract(
    *,
    is_draft_worker: bool,
    compression_ratios: list[int],
    swa_size: int,
    c4_size: int,
    c128_size: int,
    c4_state_pool_size: int,
    c128_state_pool_size: int,
) -> _DSV4OscarKVPoolContract:
    """Validate the physical target/draft topology admitted by OSCAR.

    The target owns every calibrated C4/C128 cache and its compression-state
    pools.  A DSV4 speculative worker is a different consumer: NextN attention
    is ratio-0/SWA-only and shares the target allocator, so it must own *zero*
    compressed capacity.  Treating the target artifact's C4 payload as draft
    coverage is both incorrect and what previously made speculative startup
    fail.  This contract keeps the draft under OSCAR admission while making it
    impossible to silently create a generic compressed cache there.
    """

    unsupported_ratios = sorted(set(compression_ratios) - {0, 4, 128})
    if unsupported_ratios:
        raise ValueError(
            f"DSV4 OSCAR pool has unsupported compression ratios {unsupported_ratios}"
        )
    if swa_size <= 0:
        raise ValueError("DSV4 OSCAR requires a positive protected SWA cache")

    compressed_layer_ids = frozenset(
        layer_id for layer_id, ratio in enumerate(compression_ratios) if ratio != 0
    )
    c4_layer_ids = frozenset(
        layer_id for layer_id, ratio in enumerate(compression_ratios) if ratio == 4
    )
    compressed_capacities = {
        "c4_size": c4_size,
        "c128_size": c128_size,
        "c4_state_pool_size": c4_state_pool_size,
        "c128_state_pool_size": c128_state_pool_size,
    }

    if is_draft_worker:
        if compressed_layer_ids:
            raise ValueError(
                "DSV4 OSCAR draft workers must be SWA-only; compressed draft "
                f"layers={sorted(compressed_layer_ids)}"
            )
        nonzero_capacities = {
            name: value for name, value in compressed_capacities.items() if value != 0
        }
        if nonzero_capacities:
            raise ValueError(
                "DSV4 OSCAR draft workers cannot allocate compressed cache/state "
                f"capacity: {nonzero_capacities}"
            )
        return _DSV4OscarKVPoolContract(
            consumer_role="draft_swa_only",
            compressed_layer_ids=frozenset(),
            c4_layer_ids=frozenset(),
            retain_runtime_calibrations=False,
        )

    if not compressed_layer_ids:
        raise ValueError(
            "DSV4 OSCAR target lost its calibrated C4/C128 compression topology"
        )
    required_capacities: dict[str, int] = {}
    if c4_layer_ids:
        required_capacities.update(
            c4_size=c4_size,
            c4_state_pool_size=c4_state_pool_size,
        )
    if 128 in compression_ratios:
        required_capacities.update(
            c128_size=c128_size,
            c128_state_pool_size=c128_state_pool_size,
        )
    missing_capacities = [
        name for name, value in required_capacities.items() if value <= 0
    ]
    if missing_capacities:
        raise ValueError(
            "DSV4 OSCAR target has no capacity for calibrated compressed pools: "
            f"{missing_capacities}"
        )
    return _DSV4OscarKVPoolContract(
        consumer_role="target_compressed",
        compressed_layer_ids=compressed_layer_ids,
        c4_layer_ids=c4_layer_ids,
        retain_runtime_calibrations=True,
    )


def _load_dsv4_oscar_runtime_calibrations(
    *,
    artifact_path: Path,
    device: str,
    config_sha256: str,
    contract: _DSV4OscarKVPoolContract,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Validate one artifact and retain only calibrations this worker consumes.

    Admission of the artifact/checkpoint binding happens before this helper.
    Draft workers still validate the full shared-latent artifact, on CPU, but
    retain no target-layer rotations and never load the C4 scorer payload onto
    their GPU because their physical topology contains no compressed cache.
    """

    from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
        load_dsv4_oscar_int2_calibrations,
    )

    validation_device = device if contract.retain_runtime_calibrations else "cpu"
    shared_calibrations = load_dsv4_oscar_int2_calibrations(
        artifact_path,
        device=validation_device,
        expected_metadata={"config_sha256": config_sha256},
    )
    if not contract.retain_runtime_calibrations:
        return {}, {}

    actual_shared_layers = set(shared_calibrations)
    if actual_shared_layers != contract.compressed_layer_ids:
        raise ValueError(
            "OSCAR artifact shared-latent coverage does not match compressed "
            "model layers: "
            f"missing={sorted(contract.compressed_layer_ids - actual_shared_layers)}, "
            f"extra={sorted(actual_shared_layers - contract.compressed_layer_ids)}"
        )

    from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
        load_dsv4_oscar_int2_c4_calibrations,
    )

    c4_calibrations = load_dsv4_oscar_int2_c4_calibrations(
        artifact_path,
        device=device,
        expected_metadata={"config_sha256": config_sha256},
    )
    actual_c4_layers = set(c4_calibrations)
    if actual_c4_layers != contract.c4_layer_ids:
        raise ValueError(
            "OSCAR artifact C4 scorer coverage does not match the target model: "
            f"missing={sorted(contract.c4_layer_ids - actual_c4_layers)}, "
            f"extra={sorted(actual_c4_layers - contract.c4_layer_ids)}"
        )
    return shared_calibrations, c4_calibrations


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


class DeepSeekV4SingleKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_bf16_cache: bool = False,
        use_int4_cache: bool = False,
        use_oscar_int2_cache: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        selected_layouts = sum((use_bf16_cache, use_int4_cache, use_oscar_int2_cache))
        if selected_layouts > 1:
            raise ValueError(
                "DSV4 BF16, non-OSCAR INT4, and OSCAR-INT2 layouts are "
                "mutually exclusive"
            )
        self.use_bf16_cache = use_bf16_cache
        self.use_int4_cache = use_int4_cache
        self.use_oscar_int2_cache = use_oscar_int2_cache
        self.kv_storage_mode = (
            "bfloat16"
            if use_bf16_cache
            else "oscar_int2_asymmetric"
            if use_oscar_int2_cache
            else "int4_symmetric"
            if use_int4_cache
            else "fp8_e4m3"
        )
        self.use_ampere_fp8_storage = (
            not use_bf16_cache
            and not use_int4_cache
            and not use_oscar_int2_cache
            and dsv4_uses_ampere_fp8_kv_storage(get_dsv4_device_capability(device))
        )
        if use_bf16_cache:
            # BF16 mode: nope stored as bf16 (2B/el), rope as bf16 (2B/el),
            # no per-tile scale section needed.
            self.scale_pad = 0
            self.quantize_block_size = 0
        elif use_int4_cache or use_oscar_int2_cache:
            self.scale_pad = 0
            self.quantize_block_size = 64
        else:
            self.scale_pad = 1
            self.quantize_block_size = 64
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self._create_buffers()

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        if self.use_bf16_cache:
            # nope (bf16, 2B/el) + rope (bf16, 2B/el) = all-bf16, no scale
            nope_bytes = self.qk_nope_head_dim * 2  # 896
            rope_bytes = self.qk_rope_head_dim * 2  # 128
            return nope_bytes + rope_bytes  # 1024
        if self.use_int4_cache:
            from sglang.kernels.ops.attention.dsv4.int4_storage import (
                STORAGE_BYTES_PER_TOKEN,
            )

            return STORAGE_BYTES_PER_TOKEN
        if self.use_oscar_int2_cache:
            from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
                OSCAR_INT2_STORAGE_BYTES_PER_TOKEN,
            )

            return OSCAR_INT2_STORAGE_BYTES_PER_TOKEN
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = (
            bytes_per_page_non_padded
            if self.use_bf16_cache or self.use_int4_cache or self.use_oscar_int2_cache
            else ceil_div(bytes_per_page_non_padded, 576) * 576
        )

        if self.use_bf16_cache:
            assert bytes_per_token == 448 * 2 + 64 * 2
        elif self.use_int4_cache:
            assert bytes_per_token == 368
            assert self.store_dtype == torch.uint8
        elif self.use_oscar_int2_cache:
            assert bytes_per_token == 272
            assert self.store_dtype == torch.uint8
        else:
            assert bytes_per_token == 448 + 64 * 2 + 8, (
                "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
                "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
            )
            assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=(
                torch.uint8
                if self.use_bf16_cache
                or self.use_int4_cache
                or self.use_oscar_int2_cache
                else self.store_dtype
            ),
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack = None,
        cache_bf16_pack: Any = None,
    ):
        if self.use_int4_cache:
            raise RuntimeError(
                "signed-INT4 pages require set_key_buffer_fused with BF16 source keys"
            )
        if self.use_oscar_int2_cache:
            raise RuntimeError("OSCAR-INT2 pages require a calibrated fused writer")
        if self.use_bf16_cache:
            dsv4_index_buf_accessor.SetBf16KAndS.execute(
                pool=self,
                buf=self.kv_buffer[layer_id],
                loc=loc,
                pack=cache_bf16_pack,
            )
        else:
            dsv4_index_buf_accessor.SetKAndS.execute(
                pool=self,
                buf=self.kv_buffer[layer_id],
                loc=loc,
                nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
            )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        *,
        oscar_calibration: Any = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.use_bf16_cache:
            if cache_k.dtype != torch.bfloat16:
                cache_k = cache_k.to(torch.bfloat16)
            cache_bf16_pack = dsv4_index_buf_accessor.NopeBf16RopeBf16Pack(
                k_nope_bf16=cache_k[:, : self.qk_nope_head_dim],
                k_rope_bf16=cache_k[:, self.qk_nope_head_dim :],
            )
            return dsv4_index_buf_accessor.SetBf16KAndS.execute(
                pool=self,
                buf=self.kv_buffer[layer_id],
                loc=loc,
                pack=cache_bf16_pack,
            )
        if self.use_oscar_int2_cache:
            if oscar_calibration is None:
                raise ValueError(
                    "OSCAR-INT2 cache write requires a validated calibration"
                )
            from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
                quantize_dsv4_oscar_int2_cache_paged,
            )

            return quantize_dsv4_oscar_int2_cache_paged(
                cache_k,
                oscar_calibration,
                self.kv_buffer[layer_id],
                loc,
                page_size=self.page_size,
                write_mask=write_mask,
            )
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
            int4_store=self.use_int4_cache,
        )

    def get_key_buffer(self, layer_id: int):
        if self.use_bf16_cache:
            # Return raw uint8 buffer — the dispatch will view as bf16.
            return self.kv_buffer[layer_id - self.start_layer]
        if (
            self.use_int4_cache
            or self.use_oscar_int2_cache
            or self.use_ampere_fp8_storage
        ):
            # Ampere treats E4M3 purely as a byte-storage format.  Never expose
            # a native-float8 pointer to a Triton consumer on SM86.
            return self.kv_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_bf16_cache: bool = False,
        use_int4_cache: bool = False,
        use_oscar_int2_cache: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim
        selected_layouts = sum((use_bf16_cache, use_int4_cache, use_oscar_int2_cache))
        if selected_layouts > 1:
            raise ValueError(
                "DSV4 BF16, non-OSCAR INT4, and OSCAR-INT2 C4 indexer "
                "layouts are mutually exclusive"
            )
        self.use_bf16_cache = use_bf16_cache
        self.use_int4_cache = use_int4_cache
        self.use_oscar_int2_cache = use_oscar_int2_cache
        self.kv_storage_mode = (
            "bfloat16"
            if use_bf16_cache
            else "oscar_int2_c4_asymmetric"
            if use_oscar_int2_cache
            else "int4_symmetric"
            if use_int4_cache
            else "fp8_e4m3"
        )
        self.use_ampere_fp8_storage = (
            not use_bf16_cache
            and not use_int4_cache
            and not use_oscar_int2_cache
            and dsv4_uses_ampere_fp8_kv_storage(get_dsv4_device_capability(device))
        )
        self.use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer
        if (self.use_int4_cache or self.use_oscar_int2_cache) and self.use_fp4_indexer:
            raise ValueError(
                "DSV4 packed C4 indexer storage and the FP4 indexer are "
                "mutually exclusive"
            )

        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        if self.use_bf16_cache:
            return self.index_head_dim * 2
        if self.use_int4_cache:
            from sglang.kernels.ops.attention.dsv4.int4_c4_indexer_poc import (
                INT4_C4_BYTES_PER_TOKEN,
            )

            return INT4_C4_BYTES_PER_TOKEN
        if self.use_oscar_int2_cache:
            from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
                STORAGE_BYTES_PER_TOKEN,
            )

            return STORAGE_BYTES_PER_TOKEN
        if self.use_fp4_indexer:
            return self.index_head_dim // 2 + 4
        return self.index_head_dim + 4

    def _create_buffer(self):
        num_pages = (self.size + self.page_size + 1) // self.page_size
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                if self.use_bf16_cache:
                    # BF16: (num_pages, page_size, head_dim) — no scales
                    self.index_k_bf16_buffer = [
                        torch.zeros(
                            (num_pages, self.page_size, self.index_head_dim),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    num_scales_per_token = self.index_head_dim // self.quant_block_size
                    if self.use_int4_cache or self.use_oscar_int2_cache:
                        page_bytes = self.page_size * self.get_bytes_per_token()
                    else:
                        page_bytes = self.page_size * self.index_head_dim
                        page_bytes += self.page_size * num_scales_per_token * 4
                    self.index_k_with_scale_buffer = [
                        torch.zeros(
                            (num_pages, page_bytes),
                            dtype=self.index_k_with_scale_buffer_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.use_bf16_cache:
            return self.index_k_bf16_buffer[layer_id]
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_bf16_buffer(self, layer_id: int) -> torch.Tensor:
        assert self.use_bf16_cache
        return self.index_k_bf16_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_int4_cache or self.use_oscar_int2_cache:
            raise RuntimeError(
                "packed indexer pages are consumed directly by their paged scorer"
            )
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        if self.use_int4_cache or self.use_oscar_int2_cache:
            raise RuntimeError(
                "packed indexer pages require set_index_fused with BF16 keys"
            )
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        *,
        oscar_calibration: Any = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.use_bf16_cache:
            return self.set_index_k_bf16(layer_id, loc, cache_k.bfloat16())
        if self.use_int4_cache:
            from sglang.kernels.ops.attention.dsv4.int4_c4_indexer_poc import (
                store_int4_c4_indexer_cache,
            )

            return store_int4_c4_indexer_cache(
                cache_k.bfloat16(),
                self.index_k_with_scale_buffer[layer_id - self.start_layer],
                loc,
                page_size=self.page_size,
            )
        if self.use_oscar_int2_cache:
            if oscar_calibration is None:
                raise ValueError(
                    "OSCAR-INT2 C4 cache write requires a validated calibration"
                )
            from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
                store_oscar_int2_c4_indexer_cache,
            )

            return store_oscar_int2_c4_indexer_cache(
                cache_k.bfloat16(),
                self.index_k_with_scale_buffer[layer_id - self.start_layer],
                loc,
                calibration=oscar_calibration,
                page_size=self.page_size,
                write_mask=write_mask,
            )
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
            int4_store=False,
        )

    def set_index_k_bf16(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k_bf16: torch.Tensor,
    ) -> None:
        assert self.use_bf16_cache
        buf = self.index_k_bf16_buffer[layer_id - self.start_layer]
        _scatter_index_k_bf16_kernel[(loc.shape[0],)](
            buf,
            loc,
            index_k_bf16,
            PAGE_SIZE=self.page_size,
            HEAD_DIM=self.index_head_dim,
            DATA_STRIDE=index_k_bf16.stride(0),
            BLOCK_D=triton.next_power_of_2(self.index_head_dim),
        )

    def set_index_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        return store_fp4_index_k_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
        )


@triton.jit
def _scatter_index_k_bf16_kernel(
    buf_ptr,
    loc_ptr,
    data_ptr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DATA_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(loc_ptr + token_id)
    page = loc // PAGE_SIZE
    offset = loc % PAGE_SIZE
    dims = tl.arange(0, BLOCK_D)
    mask = dims < HEAD_DIM
    data = tl.load(
        data_ptr + token_id * DATA_STRIDE + dims,
        mask=mask,
        other=0.0,
    )
    output = (page * PAGE_SIZE + offset) * HEAD_DIM + dims
    tl.store(buf_ptr + output, data, mask=mask)


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


# The following kv pool follows ATOM's unified_kv kernel layout.
class DeepSeekV4UnifiedKVPool:
    """
    Layout:
    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16
    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)
    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)
    """

    K_PER_BLOCK = {0: 0, 4: 32, 128: 1}

    def __init__(
        self,
        *,
        stage_ratios: List[int],
        num_slots: int,
        num_blocks: int,
        page_size: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        device: str,
        memory_saver_adapter,
        custom_mem_pool,
        swa_ring_size: int,
    ):
        self.swa_ring_size = swa_ring_size
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool
                else nullcontext()
            ):
                for ratio in stage_ratios:
                    # Pad by one extra page. The KV pool reserves a null slot
                    # (token indices run 1..size).
                    compress_rows = self.num_blocks * self.k_per_block[ratio]
                    rows_per_page = self.page_size // ratio if ratio else 0
                    padded_compress_rows = compress_rows + rows_per_page
                    bufs.append(
                        torch.zeros(
                            self.swa_pages + padded_compress_rows,
                            self.head_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
        self.kv_buffer = bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):
    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        c4_state_dtype: torch.dtype,
        c128_state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        sliding_window: int = 128,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
        num_req_slots: Optional[int] = None,
        is_draft_worker: bool = False,
    ):
        requested_dtype = dtype
        device_capability = get_dsv4_device_capability(device)
        dtype = resolve_dsv4_kv_cache_dtype(
            requested_dtype,
            device_capability=device_capability,
        )
        self.use_int4_storage = envs.SGLANG_DSV4_INT4_KV_STORAGE.get()
        self.use_int4_indexer_storage = envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.get()
        self.use_oscar_int2_storage = envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.get()
        self.is_draft_worker = is_draft_worker
        self.oscar_consumer_role = "disabled"
        oscar_contract: _DSV4OscarKVPoolContract | None = None
        self.use_selective_c128_bf16_storage = (
            envs.SGLANG_DSV4_SM86_C128_BF16_STORAGE.get()
        )
        if self.use_oscar_int2_storage:
            oscar_contract = _resolve_dsv4_oscar_kv_pool_contract(
                is_draft_worker=is_draft_worker,
                compression_ratios=compression_ratios,
                swa_size=swa_size,
                c4_size=c4_size,
                c128_size=c128_size,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
            )
            self.oscar_consumer_role = oscar_contract.consumer_role
            if not dsv4_supports_oscar_int2_kv_storage(device_capability):
                raise ValueError(
                    "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE is implemented only "
                    "for exact SM86, got "
                    f"{format_dsv4_device_capability(device_capability)}"
                )
            if dtype == torch.bfloat16:
                raise ValueError(
                    "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE requires --kv-cache-dtype "
                    "fp8_e4m3 as its raw-byte configuration carrier"
                )
            if self.use_int4_storage or self.use_int4_indexer_storage:
                raise ValueError(
                    "OSCAR-INT2 is incompatible with the non-OSCAR DSV4 "
                    "INT4 cache prototypes"
                )
            if self.use_selective_c128_bf16_storage:
                raise ValueError(
                    "OSCAR-INT2 owns the C128 layout and is incompatible with "
                    "SGLANG_DSV4_SM86_C128_BF16_STORAGE"
                )
            if enable_hisparse:
                raise ValueError("DSV4 OSCAR-INT2 is incompatible with HiSparse")
        if self.use_int4_storage or self.use_int4_indexer_storage:
            if not dsv4_supports_int4_kv_storage(device_capability):
                raise ValueError(
                    "SGLANG_DSV4_INT4_KV_STORAGE is implemented only for exact "
                    f"SM86, got {format_dsv4_device_capability(device_capability)}"
                )
            if dtype == torch.bfloat16:
                raise ValueError(
                    "SGLANG_DSV4_INT4_KV_STORAGE requires --kv-cache-dtype "
                    "fp8_e4m3 as its raw-byte configuration carrier"
                )
            if enable_hisparse and self.use_int4_storage:
                raise ValueError(
                    "SGLANG_DSV4_INT4_KV_STORAGE is incompatible with HiSparse"
                )
        if self.use_selective_c128_bf16_storage:
            if not dsv4_supports_selective_c128_bf16_storage(device_capability):
                raise ValueError(
                    "SGLANG_DSV4_SM86_C128_BF16_STORAGE is implemented only "
                    "for exact SM86, got "
                    f"{format_dsv4_device_capability(device_capability)}"
                )
            if dtype == torch.bfloat16:
                raise ValueError(
                    "SGLANG_DSV4_SM86_C128_BF16_STORAGE requires --kv-cache-dtype "
                    "fp8_e4m3 as the SWA/C4 byte-storage carrier"
                )
            if self.use_int4_storage:
                raise ValueError(
                    "SGLANG_DSV4_SM86_C128_BF16_STORAGE is incompatible with "
                    "SGLANG_DSV4_INT4_KV_STORAGE"
                )
        self.oscar_calibrations: dict[int, Any] = {}
        self.oscar_c4_calibrations: dict[int, Any] = {}
        self.oscar_artifact_path = ""
        self.oscar_artifact_sha256 = ""
        self.oscar_model_config_sha256 = ""
        self.oscar_admission_receipt_path = ""
        self.oscar_admission_receipt_sha256 = ""
        self.oscar_admission_sha256 = ""
        self.oscar_artifact_provenance_sha256 = ""
        self.oscar_checkpoint_sha256 = ""
        self.oscar_checkpoint_fingerprint_sha256 = ""
        self.oscar_model_id = ""
        if self.use_oscar_int2_storage:
            assert oscar_contract is not None
            artifact_text = envs.SGLANG_DSV4_OSCAR_CALIBRATION_PATH.get().strip()
            if not artifact_text:
                raise ValueError(
                    "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE requires an explicit "
                    "SGLANG_DSV4_OSCAR_CALIBRATION_PATH"
                )
            artifact_path = Path(artifact_text)
            self.oscar_artifact_sha256 = _sha256_regular_file(
                artifact_path, label="OSCAR calibration artifact"
            )
            server_args = get_server_args()
            model_path = Path(server_args.model_path)
            if not model_path.is_absolute() or not model_path.is_dir():
                raise ValueError(
                    "DSV4 OSCAR-INT2 requires a local absolute model directory "
                    f"for provenance validation, got {model_path}"
                )
            config_path = model_path / "config.json"
            self.oscar_model_config_sha256 = _sha256_regular_file(
                config_path, label="DeepSeek V4 model config"
            )
            admission_text = envs.SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH.get().strip()
            if not admission_text:
                raise ValueError(
                    "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE requires a model-bound "
                    "SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH"
                )
            admission_path = Path(admission_text)
            admission, self.oscar_admission_receipt_sha256 = (
                _load_oscar_admission_receipt(
                    admission_path,
                    artifact_path=artifact_path,
                    artifact_sha256=self.oscar_artifact_sha256,
                    checkpoint_path=model_path,
                    config_sha256=self.oscar_model_config_sha256,
                )
            )
            self.oscar_admission_receipt_path = str(admission_path)
            self.oscar_admission_sha256 = str(admission["admission_sha256"])
            self.oscar_artifact_provenance_sha256 = str(
                admission["artifact_provenance_sha256"]
            )
            self.oscar_checkpoint_sha256 = str(admission["checkpoint_sha256"])
            self.oscar_checkpoint_fingerprint_sha256 = str(
                admission["checkpoint_fingerprint_sha256"]
            )
            self.oscar_model_id = str(admission["model_id"])
            (
                self.oscar_calibrations,
                self.oscar_c4_calibrations,
            ) = _load_dsv4_oscar_runtime_calibrations(
                artifact_path=artifact_path,
                device=device,
                config_sha256=self.oscar_model_config_sha256,
                contract=oscar_contract,
            )
            self.oscar_artifact_path = str(artifact_path)
            logger.info(
                "Loaded DSV4 OSCAR-INT2 artifact sha256=%s config_sha256=%s "
                "checkpoint_sha256=%s admission_sha256=%s role=%s layers=%d "
                "c4_layers=%d",
                self.oscar_artifact_sha256,
                self.oscar_model_config_sha256,
                self.oscar_checkpoint_sha256,
                self.oscar_admission_sha256,
                self.oscar_consumer_role,
                len(self.oscar_calibrations),
                len(self.oscar_c4_calibrations),
            )
        logger.info(
            "DeepSeek V4 KV cache storage: requested=%s, effective=%s, device=%s",
            dsv4_kv_cache_dtype_name(requested_dtype),
            dsv4_kv_cache_dtype_name(dtype),
            format_dsv4_device_capability(device_capability),
        )
        if dtype != requested_dtype:
            logger.warning(
                "DeepSeek V4 FP8 KV cache is unavailable on %s; using the "
                "BF16 layout (1024 bytes/token instead of 584 bytes/token).",
                format_dsv4_device_capability(device_capability),
            )
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.use_ampere_fp8_storage = (
            dtype != torch.bfloat16
            and not self.use_int4_storage
            and not self.use_oscar_int2_storage
            and dsv4_uses_ampere_fp8_kv_storage(device_capability)
        )
        self.kv_storage_mode = (
            "oscar_int2_asymmetric+protected_swa_bfloat16"
            if self.use_oscar_int2_storage
            else "int4_symmetric"
            if self.use_int4_storage
            else "fp8_e4m3+sparse_c128_bfloat16"
            if self.use_selective_c128_bf16_storage
            else "bfloat16"
            if dtype == torch.bfloat16
            else "fp8_e4m3"
        )
        if self.use_oscar_int2_storage:
            logger.info(
                "DeepSeek V4 KV cache on %s admitted the OSCAR-INT2 "
                "physical layout and does not initialize the generic E4M3 "
                "software-decode path.",
                format_dsv4_device_capability(device_capability),
            )
        elif (
            self.use_ampere_fp8_storage
            or self.use_int4_storage
            or self.use_int4_indexer_storage
        ):
            # The sparse-attention and indexer consumers use a stable E4M3FN
            # decode table.  Materialize it during pool construction, never on
            # the first invocation inside CUDA-graph capture.
            from sglang.kernels.ops.attention.dsv4.fp8_storage import (
                prime_e4m3fn_decode_lut,
            )

            prime_e4m3fn_decode_lut(device)
            if self.use_int4_storage or self.use_int4_indexer_storage:
                logger.warning(
                    "DeepSeek V4 KV cache on %s uses EXPERIMENTAL signed-INT4 "
                    "storage (latent=%s, c4_indexer=%s). "
                    "Coherency and acceptance quality gates are required.",
                    format_dsv4_device_capability(device_capability),
                    self.use_int4_storage,
                    self.use_int4_indexer_storage,
                )
            else:
                logger.info(
                    "DeepSeek V4 KV cache on %s uses raw E4M3FN bytes with "
                    "software decode into BF16 tensor-core consumers.",
                    format_dsv4_device_capability(device_capability),
                )
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        # SWA ring needs one slot per addressable req_pool_idx. PD decode inflates
        # req_to_token past max_num_reqs (pre-alloc), so the caller passes the real
        # capacity; sizing as max_num_reqs+1 overflows ("length out of range").
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        c128_ring_size = self.get_ring_size(128)
        if ONLINE_C128:
            # Request-scoped online C128 state is indexed by req_pool_idx.
            # PD decode can allocate pre-transfer slots beyond
            # max_num_reqs, so size to the actual req_to_token row count.
            c128_state_pool_size = max(c128_state_pool_size, self.num_req_slots)
        else:
            # Offline C128 keeps a per-request raw state ring.
            c128_state_pool_size = max(
                c128_state_pool_size, self.num_req_slots * c128_ring_size
            )
        self.c128_state_pool_size = c128_state_pool_size
        self.c4_state_dtype = c4_state_dtype
        self.c128_state_dtype = c128_state_dtype
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_state_num_req_slots = c128_state_pool_size
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                self.online_c128_state_num_req_slots, dtype=torch.int64, device=device
            )

        # Determine this PP stage's absolute layer range
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        assert page_size % swa_page_size == 0
        self.sliding_window = sliding_window

        # The shared resolver above keeps this physical layout decision in sync
        # with DSV4PoolConfigurator's capacity calculation.
        _bf16 = dtype == torch.bfloat16

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = (
            0 if (_bf16 or self.use_int4_storage or self.use_oscar_int2_storage) else 1
        )

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        stage_layer_num = len(stage_ratios)
        c4_layer_num = sum(1 for r in stage_ratios if r == 4)
        c128_layer_num = sum(1 for r in stage_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        self._unified_kv = is_unified_kv_triton()
        if (
            self.use_oscar_int2_storage
            or self.use_int4_storage
            or self.use_selective_c128_bf16_storage
        ) and self._unified_kv:
            raise ValueError(
                "experimental DSV4 mixed storage does not support unified-KV storage"
            )

        if self._unified_kv:
            self.swa_kv_pool = None
            self.c4_kv_pool = None
            self.c128_kv_pool = None
            server_args = get_server_args()
            spec_extra = (
                (get_spec().speculative_num_draft_tokens - 1)
                if get_spec().speculative_algorithm is not None
                else 0
            )
            self.unified_kv_pool = DeepSeekV4UnifiedKVPool(
                stage_ratios=stage_ratios,
                num_slots=self.num_req_slots,
                num_blocks=self.c128_size,
                page_size=page_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                device=device,
                memory_saver_adapter=self.memory_saver_adapter,
                custom_mem_pool=self.custom_mem_pool,
                swa_ring_size=self.sliding_window + spec_extra,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = self.sliding_window + spec_extra
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
        else:
            self.unified_kv_pool = None
            self.swa_kv_pool = self._make_kv_pool(
                size=swa_size,
                page_size=swa_page_size,
                dtype=dtype,
                layer_num=stage_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=swa_page_size,
                use_bf16_cache=_bf16 or self.use_oscar_int2_storage,
            )

            c4_kv_pool_type = DeepSeekV4SingleKVPool
            if enable_hisparse:
                c4_kv_pool_type = HiSparseC4DevicePool
            self.c4_kv_pool = self._make_kv_pool(
                size=c4_size,
                page_size=c4_page_size,
                dtype=dtype,
                layer_num=c4_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=page_size,
                cls=c4_kv_pool_type,
                use_bf16_cache=_bf16,
                use_oscar_int2_cache=self.use_oscar_int2_storage,
            )

            self.c128_kv_pool = self._make_kv_pool(
                size=c128_size,
                page_size=c128_page_size,
                dtype=dtype,
                layer_num=c128_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=page_size,
                use_bf16_cache=_bf16 or self.use_selective_c128_bf16_storage,
                use_oscar_int2_cache=self.use_oscar_int2_storage,
            )

        indexer_size = self.c4_logical_size
        self.c4_indexer_kv_pool = self._make_indexer_pool(
            indexer_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
            use_bf16_cache=_bf16,
            use_oscar_int2_cache=self.use_oscar_int2_storage,
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        # Under HiCache the compressed region is loaded H->D per layer; wait for this
        # layer's transfer before attention reads it. No-op when HiCache is off.
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        server_args = get_server_args()
        is_speculative = get_spec().speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv:
            # Unified buffer per layer: [swa_pages + padded_compress_rows, head_dim].
            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +
            # loc//ratio), so reuse the page-block PD transfer by offsetting the ptr
            # past the SWA ring and setting item_len = one page of rows. The SWA ring
            # ships separately as StateType.SWA_RING. Order [c4, c4_indexer, c128]
            # mirrors the non-unified kv_data layout (keeps PP ptr-slicing valid).
            stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
            swa_pages = self.unified_kv_pool.swa_pages

            def _append_compressed_entry(local_layer_id: int, ratio: int) -> None:
                buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                row_bytes = buf[0].nbytes
                rows_per_page = self.page_size // ratio
                compress_rows = buf.shape[0] - swa_pages
                data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                data_lens.append(compress_rows * row_bytes)
                item_lens.append(rows_per_page * row_bytes)

            c4_locals = [i for i, r in enumerate(stage_ratios) if r == 4]
            c128_locals = [i for i, r in enumerate(stage_ratios) if r == 128]

            for i in c4_locals:
                _append_compressed_entry(i, 4)
            for buf in self.c4_indexer_kv_pool.index_k_with_scale_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            for i in c128_locals:
                _append_compressed_entry(i, 128)

            return data_ptrs, data_lens, item_lens

        buf_groups = [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_bf16_buffer
            if self.c4_indexer_kv_pool.use_bf16_cache
            else self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]

        for bufs in buf_groups:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """SWA-ring region [0, swa_pages) of every unified_kv layer, addressed
        per-row by ring slot. Shipped as the StateType.SWA_RING PD component."""
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:
        """
        In unified_kv, swa/c4/c128 share one buffer with one slot per row. But the
        HiCache host pool transfers a whole page per indexed row, so we reshape the
        compressed region into the layout it expects: skip the SWA segment, reshape to
        one row per page, then cast to uint8.
        """
        assert self._unified_kv, "unified_region_buffers requires unified_kv layout"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"

        swa_pages = self.unified_kv_pool.swa_pages
        head_dim = self.unified_kv_pool.head_dim
        rows_per_page = self.page_size // ratio
        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == ratio]

        views: List[torch.Tensor] = []
        for local_layer_id in local_layer_ids:
            buf = self.unified_kv_pool.kv_buffer[local_layer_id]
            compress_rows = buf.shape[0] - swa_pages
            assert compress_rows % rows_per_page == 0, (
                f"compressed rows {compress_rows} not a multiple of "
                f"rows_per_page {rows_per_page} for ratio {ratio}"
            )
            num_pages = compress_rows // rows_per_page
            page_view = (
                buf.narrow(0, swa_pages, compress_rows)
                .reshape(num_pages, rows_per_page * head_dim)
                .view(torch.uint8)
            )
            views.append(page_view)

        item_bytes = (
            rows_per_page * head_dim * self.unified_kv_pool.kv_buffer[0].element_size()
        )
        return views, item_bytes

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if not self._unified_kv:
            for buf in self.swa_kv_pool.kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                if pool.ratio == 128:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_c128_state_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            item_lens.append(t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128)
        return data_ptrs, data_lens, item_lens

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
        use_bf16_cache: bool = False,
        use_oscar_int2_cache: bool = False,
    ) -> DeepSeekV4SingleKVPool:
        """Build a full / SWA / c4 / c128 single-KV pool. ``global_page_size``
        is the model-wide page_size (== ``page_size`` for the SWA pool, larger
        for the per-ratio c4/c128 pools); the default CUDA pool ignores it.
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the NPU bf16
        PA_ND variant, which needs ``global_page_size`` for its kernel view."""
        del global_page_size  # CUDA pools key only off their own page_size
        return cls(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            use_bf16_cache=use_bf16_cache,
            use_int4_cache=self.use_int4_storage,
            use_oscar_int2_cache=use_oscar_int2_cache,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        use_bf16_cache: bool = False,
        use_oscar_int2_cache: bool = False,
    ) -> DeepSeekV4IndexerPool:
        """Build the c4 lightning-indexer K pool (packed CUDA layout).
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the
        dedicated-buffer NPU variant (int8 K + fp16 scale)."""
        return DeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            use_bf16_cache=use_bf16_cache,
            use_int4_cache=self.use_int4_indexer_storage,
            use_oscar_int2_cache=use_oscar_int2_cache,
        )

    def _state_pool_size(self, ratio: int) -> int:
        return self.c4_state_pool_size if ratio == 4 else self.c128_state_pool_size

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer attention compress-state pool for ``ratio``
        (4 or 128). Overridden by :class:`DSV4NPUTokenToKVPool` to swap the
        ring-buffered pool for the NPU paged one."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=(ratio == 128 and ONLINE_C128),
            swa_page_size=self.swa_page_size,
            online_mtp_max_draft_tokens=(
                self.online_mtp_max_draft_tokens if ratio == 128 else 0
            ),
        )

    def _make_indexer_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer indexer compress-state pool (c4 only)."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.indexer_head_dim,
            device=self.device,
            dtype=self.c4_state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            self.compress_state_pools[idx] = self._make_attn_state_pool(
                ratio, enable_memory_saver
            )

            if ratio == 4:
                self.indexer_compress_state_pools[idx] = self._make_indexer_state_pool(
                    ratio, enable_memory_saver
                )

    def _init_compressed_layer_mapping(self):
        c1_cnt = c4_cnt = c128_cnt = 0
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=c4_cnt,
                    compress_kv_pool=self.c4_kv_pool,
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=c128_cnt,
                    compress_kv_pool=self.c128_kv_pool,
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_oscar_calibration(self, layer_id: int):
        """Return the exact model-bound shared-latent calibration for a layer."""

        if not self.use_oscar_int2_storage:
            raise RuntimeError("the DSV4 OSCAR-INT2 cache is not enabled")
        if self.oscar_consumer_role == "draft_swa_only":
            raise RuntimeError(
                "the OSCAR-admitted DSV4 draft is SWA-only and owns no "
                "compressed-layer calibration"
            )
        calibration = self.oscar_calibrations.get(layer_id)
        if calibration is None:
            raise ValueError(
                f"OSCAR calibration artifact does not cover model layer {layer_id}"
            )
        return calibration

    def get_oscar_c4_calibration(self, layer_id: int):
        """Return the exact model-bound C4 scorer calibration for a layer."""

        if not self.use_oscar_int2_storage:
            raise RuntimeError("the DSV4 OSCAR-INT2 cache is not enabled")
        if self.oscar_consumer_role == "draft_swa_only":
            raise RuntimeError(
                "the OSCAR-admitted DSV4 draft is SWA-only and owns no C4 scorer"
            )
        calibration = self.oscar_c4_calibrations.get(layer_id)
        if calibration is None:
            raise ValueError(
                f"OSCAR C4 calibration artifact does not cover model layer {layer_id}"
            )
        return calibration

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert compress_state_pool is not None, (
            "Only c4/c128 layers have attention states."
        )
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_max_draft_tokens)
        return 0

    def get_online_c128_state_num_req_slots(self) -> int:
        return self.online_c128_state_num_req_slots

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        assert self.online_c128_mtp_pending_seq_lens is not None
        return self.online_c128_mtp_pending_seq_lens

    def clear_c128_req_state(self, req_pool_idx: int) -> None:
        """Reset request-scoped C128 state for one req slot."""
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            state = pool.kv_score_buffer.kv_score
            if ONLINE_C128:
                row = state[req_pool_idx]
                head_dim = row.shape[-1] // 3
                row[:head_dim].fill_(float("-inf"))
                row[head_dim:].zero_()
            else:
                start = req_pool_idx * pool.ring_size
                rows = state[start : start + pool.ring_size]
                half = rows.shape[-1] // 2
                rows[:, :half].zero_()
                rows[:, half:].fill_(float("-inf"))

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        """Clear offline C128 ring slots written for rejected speculative tokens."""
        if ONLINE_C128 or num_draft_tokens <= 1 or req_pool_indices.numel() == 0:
            return

        bs = req_pool_indices.numel()
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            clear_unaccepted_c128_draft_states(
                pool.kv_score_buffer.kv_score,
                req_pool_indices,
                seq_lens,
                accept_lens,
                ring_size=pool.ring_size,
                num_draft_tokens=num_draft_tokens,
            )

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert indexer_compress_state_pool is not None, (
            "Only c4 layers have indexer states."
        )
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index."""
        return layer_id - self._stage_start

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack = None,
        cache_bf16_pack: Any = None,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id),
            loc,
            cache_nope_fp8_rope_bf16_pack,
            cache_bf16_pack=cache_bf16_pack,
        )

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack = None,
        cache_bf16_pack: Any = None,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id,
            loc,
            cache_nope_fp8_rope_bf16_pack,
            cache_bf16_pack=cache_bf16_pack,
        )

    def get_index_k_page_size(self) -> int:
        return self.c4_indexer_kv_pool.page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_index_k_bf16_buffer(self, layer_id: int) -> torch.Tensor:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_bf16_buffer(compress_layer_id)

    def set_index_k_bf16(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k_bf16: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_bf16(compress_layer_id, loc, index_k_bf16)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack = None,
        cache_bf16_pack: Any = None,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id),
            swa_loc,
            cache_nope_fp8_rope_bf16_pack,
            cache_bf16_pack=cache_bf16_pack,
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
            page_size=self.swa_kv_pool.page_size,
            bf16_store=self.swa_kv_pool.use_bf16_cache,
            int4_store=self.swa_kv_pool.use_int4_cache,
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        *,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(
            compress_layer_id,
            loc,
            cache_k,
            oscar_calibration=(
                self.get_oscar_calibration(layer_id)
                if self.use_oscar_int2_storage
                else None
            ),
            write_mask=write_mask,
        )

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        *,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(
            compress_layer_id,
            loc,
            cache_k,
            oscar_calibration=(
                self.get_oscar_c4_calibration(layer_id)
                if self.use_oscar_int2_storage
                else None
            ),
            write_mask=write_mask,
        )

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fp4(compress_layer_id, loc, cache_k)
