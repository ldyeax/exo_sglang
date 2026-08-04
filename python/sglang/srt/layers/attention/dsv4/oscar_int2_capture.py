"""Bounded, provenance-aware runtime capture for DSV4 OSCAR calibration.

The collector is intentionally off unless ``SGLANG_DSV4_OSCAR_CAPTURE_CONFIG``
names an absolute configuration file.  It captures eager target-model prefill
only; CUDA-graph capture/replay, decode, speculative verification, draft
models, and an OSCAR-enabled cache are rejected or ignored.

TP2 attention queries are written as deterministic rank-local 32-head shards.
The offline finalizer validates identical sampled token rows and concatenates
the shards into the model-global 64-head tensor.  The shared latent and the C4
scorer projections are replicated by the model, so only rank zero records
them.  C4 Q is nevertheless required to contain all 64 heads at runtime -- a
local-head tensor can never be mislabeled as a replicated scorer observation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import torch

CAPTURE_CONFIG_ENV: Final = "SGLANG_DSV4_OSCAR_CAPTURE_CONFIG"
_CAPTURE_CONFIG_PATH: Final = os.environ.get(CAPTURE_CONFIG_ENV, "")
CONFIG_FORMAT: Final = "dsv4-oscar-int2-runtime-capture-config"
CONTROL_FORMAT: Final = "dsv4-oscar-int2-runtime-capture-control"
RAW_FORMAT: Final = "dsv4-oscar-int2-runtime-capture-raw"
FORMAT_VERSION: Final = 1

NUM_LAYERS: Final = 43
NUM_ATTENTION_HEADS: Final = 64
LATENT_DIM: Final = 448
HEAD_DIM: Final = 512
INDEX_HEADS: Final = 64
INDEX_HEAD_DIM: Final = 128
EXPECTED_TP_SIZE: Final = 2

Split = Literal["train", "heldout"]
CaptureKind = Literal[
    "attention_query_nope",
    "swa_latent",
    "compressed_latent",
    "c4_scorer_query",
    "c4_scorer_key",
]

_KINDS: Final[tuple[CaptureKind, ...]] = (
    "attention_query_nope",
    "swa_latent",
    "compressed_latent",
    "c4_scorer_query",
    "c4_scorer_key",
)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y"}


def capture_configured() -> bool:
    """Cheap compile-safe launch-static gate used before capture work."""

    return bool(_CAPTURE_CONFIG_PATH)


def capture_should_materialize(forward_batch: object, *, target_model: bool) -> bool:
    """Return whether this eager forward has an actively armed prompt."""

    if not capture_configured() or not _eligible_forward(
        forward_batch, target_model=target_model
    ):
        return False
    capturer = _get_capturer()
    return capturer is not None and _parse_control(capturer.config) is not None


def _load_json_object(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _require_absolute_regular(path_value: object, *, label: str) -> Path:
    if (
        not isinstance(path_value, str)
        or not path_value
        or not Path(path_value).is_absolute()
    ):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular, non-symlink file")
    return path.resolve()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class _RuntimeConfig:
    config_sha256: str
    session_dir: Path
    control_path: Path
    session_id: str
    prompt_splits: dict[str, Split]
    maximum_rows_per_prompt: dict[CaptureKind, dict[Split, int]]
    expected_tp_size: int


@dataclass(frozen=True)
class _Arm:
    generation: int
    prompt_id: str
    split: Split


def _parse_config(path: Path) -> _RuntimeConfig:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(
            f"{CAPTURE_CONFIG_ENV} must name an absolute, regular, non-symlink file"
        )
    document = _load_json_object(path)
    if (
        document.get("format") != CONFIG_FORMAT
        or document.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("incompatible DSV4 OSCAR runtime capture config")
    session_dir_value = document.get("session_dir")
    if (
        not isinstance(session_dir_value, str)
        or not Path(session_dir_value).is_absolute()
    ):
        raise ValueError("capture session_dir must be absolute")
    session_dir = Path(session_dir_value).resolve()
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise ValueError("capture session_dir must be a regular directory")
    control_path = _require_absolute_regular(
        document.get("control_path"), label="capture control_path"
    )
    if control_path.parent != session_dir:
        raise ValueError("capture control_path must be directly inside session_dir")
    session_id = document.get("session_id")
    if not isinstance(session_id, str) or len(session_id) < 16:
        raise ValueError("capture session_id must be a non-empty opaque identifier")
    expected_tp_size = document.get("expected_tp_size")
    if expected_tp_size != EXPECTED_TP_SIZE:
        raise ValueError("DSV4 OSCAR capture requires exact TP2")
    raw_prompt_splits = document.get("prompt_splits")
    if not isinstance(raw_prompt_splits, dict) or not raw_prompt_splits:
        raise ValueError("capture prompt_splits must be a non-empty object")
    prompt_splits: dict[str, Split] = {}
    for prompt_id, split in raw_prompt_splits.items():
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or split
            not in (
                "train",
                "heldout",
            )
        ):
            raise ValueError("capture prompt_splits contains an invalid entry")
        prompt_splits[prompt_id] = cast(Split, split)
    if set(prompt_splits.values()) != {"train", "heldout"}:
        raise ValueError("capture prompt set must contain train and heldout")
    raw_limits = document.get("maximum_rows_per_prompt")
    if not isinstance(raw_limits, dict) or set(raw_limits) != set(_KINDS):
        raise ValueError("capture maximum_rows_per_prompt has the wrong tensor kinds")
    limits: dict[CaptureKind, dict[Split, int]] = {}
    for kind in _KINDS:
        raw_split_limits = raw_limits.get(kind)
        if not isinstance(raw_split_limits, dict) or set(raw_split_limits) != {
            "train",
            "heldout",
        }:
            raise ValueError(f"capture row limit for {kind} must cover both splits")
        parsed: dict[Split, int] = {}
        for split in cast(tuple[Split, Split], ("train", "heldout")):
            limit = raw_split_limits.get(split)
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or not 1 <= limit <= 256
            ):
                raise ValueError(
                    f"capture row limit for {kind}/{split} must be in [1, 256]"
                )
            parsed[split] = limit
        limits[kind] = parsed
    return _RuntimeConfig(
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        session_dir=session_dir,
        control_path=control_path,
        session_id=session_id,
        prompt_splits=prompt_splits,
        maximum_rows_per_prompt=limits,
        expected_tp_size=expected_tp_size,
    )


def _parse_control(config: _RuntimeConfig) -> _Arm | None:
    document = _load_json_object(config.control_path)
    if (
        document.get("format") != CONTROL_FORMAT
        or document.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("incompatible DSV4 OSCAR capture control document")
    generation = document.get("generation")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise ValueError("capture control generation must be a non-negative integer")
    state = document.get("state")
    if state == "idle":
        return None
    if state != "armed":
        raise ValueError("capture control state must be idle or armed")
    prompt_id = document.get("prompt_id")
    split = document.get("split")
    if not isinstance(prompt_id, str) or split not in ("train", "heldout"):
        raise ValueError("armed capture control has invalid prompt provenance")
    typed_split = cast(Split, split)
    if config.prompt_splits.get(prompt_id) != typed_split:
        raise ValueError("armed capture prompt is absent or has the wrong split")
    return _Arm(generation=generation, prompt_id=prompt_id, split=typed_split)


def _eligible_forward(forward_batch: object, *, target_model: bool) -> bool:
    if not target_model:
        return False
    forward_mode = getattr(forward_batch, "forward_mode", None)
    # ForwardMode.EXTEND has a stable public enum name.  Reject MIXED and all
    # speculative modes so one armed prompt can never absorb another request.
    if getattr(forward_mode, "name", None) != "EXTEND":
        return False
    if (
        getattr(forward_batch, "batch_size", None) != 1
        or getattr(forward_batch, "_original_forward_mode", None) is not None
        or getattr(forward_batch, "tbo_parent_token_range", None) is not None
    ):
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    return not get_is_capture_mode()


class Dsv4OscarRuntimeCapturer:
    def __init__(self, config: _RuntimeConfig) -> None:
        if _is_truthy(os.environ.get("SGLANG_DSV4_OSCAR_INT2_KV_STORAGE")):
            raise ValueError(
                "OSCAR calibration capture must observe the unrotated baseline; "
                "disable SGLANG_DSV4_OSCAR_INT2_KV_STORAGE for the capture run"
            )
        self.config = config
        self._lock = threading.Lock()

    def _raw_path(
        self, *, tp_rank: int, layer_id: int, split: Split, kind: CaptureKind
    ) -> Path:
        return (
            self.config.session_dir
            / "raw"
            / f"rank_{tp_rank:02d}"
            / f"layer_{layer_id:02d}"
            / split
            / f"{kind}.pt"
        )

    def _load_state(
        self,
        path: Path,
        *,
        kind: CaptureKind,
        layer_id: int,
        split: Split,
        tp_rank: int,
        tp_size: int,
        tail: tuple[int, ...],
        head_start: int | None,
    ) -> dict[str, object]:
        if not path.exists():
            return {
                "format": RAW_FORMAT,
                "format_version": FORMAT_VERSION,
                "config_sha256": self.config.config_sha256,
                "session_id": self.config.session_id,
                "kind": kind,
                "layer_id": layer_id,
                "split": split,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "head_start": head_start,
                "tail": list(tail),
                "tensor": torch.empty((0, *tail), dtype=torch.bfloat16),
                "priorities": torch.empty((0,), dtype=torch.float64),
                "row_prompt_ids": [],
                "seen_rows_by_prompt": {},
                "generations": [],
            }
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"capture raw state is not a regular file: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise TypeError(f"capture raw state is not an object: {path}")
        state = cast(dict[str, object], loaded)
        expected = {
            "format": RAW_FORMAT,
            "format_version": FORMAT_VERSION,
            "config_sha256": self.config.config_sha256,
            "session_id": self.config.session_id,
            "kind": kind,
            "layer_id": layer_id,
            "split": split,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "head_start": head_start,
            "tail": list(tail),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"capture raw state metadata mismatch for {path}: {key}"
                )
        return state

    def record(
        self,
        *,
        kind: CaptureKind,
        layer_id: int,
        tensor: torch.Tensor,
        forward_batch: object,
        target_model: bool,
        tp_rank: int,
        tp_size: int,
        head_start: int | None = None,
    ) -> None:
        if not _eligible_forward(forward_batch, target_model=target_model):
            return
        if tp_size != self.config.expected_tp_size or not 0 <= tp_rank < tp_size:
            raise ValueError("DSV4 OSCAR capture observed a non-TP2 runtime")
        if not 0 <= layer_id < NUM_LAYERS:
            raise ValueError("DSV4 OSCAR capture layer_id must be in [0, 42]")
        arm = _parse_control(self.config)
        if arm is None:
            return
        # Shared latent and both scorer domains are replicated.  Rank zero is
        # the only writer, avoiding duplicate calibration rows.
        if kind != "attention_query_nope" and tp_rank != 0:
            return
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{kind} capture input must be floating point")
        expected_tail: tuple[int, ...]
        if kind == "attention_query_nope":
            local_heads = NUM_ATTENTION_HEADS // tp_size
            expected_tail = (local_heads, LATENT_DIM)
            expected_head_start = tp_rank * local_heads
            if head_start != expected_head_start:
                raise ValueError("attention capture head shard is not rank ordered")
        elif kind in ("swa_latent", "compressed_latent"):
            expected_tail = (LATENT_DIM,)
            if head_start is not None:
                raise ValueError("shared latent capture cannot have a head offset")
        elif kind == "c4_scorer_query":
            expected_tail = (INDEX_HEADS, INDEX_HEAD_DIM)
            if head_start != 0:
                raise ValueError("C4 scorer Q must be a replicated full-head tensor")
        else:
            expected_tail = (INDEX_HEAD_DIM,)
            if head_start is not None:
                raise ValueError("C4 scorer K cannot have a head offset")
        if (
            tensor.ndim != len(expected_tail) + 1
            or tuple(tensor.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"{kind} capture must have shape [rows,{','.join(map(str, expected_tail))}], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] == 0:
            return
        self._record_rows(
            arm=arm,
            kind=kind,
            layer_id=layer_id,
            tensor=tensor,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tail=expected_tail,
            head_start=head_start,
        )

    def _record_rows(
        self,
        *,
        arm: _Arm,
        kind: CaptureKind,
        layer_id: int,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        tail: tuple[int, ...],
        head_start: int | None,
    ) -> None:
        path = self._raw_path(
            tp_rank=tp_rank, layer_id=layer_id, split=arm.split, kind=kind
        )
        with self._lock:
            state = self._load_state(
                path,
                kind=kind,
                layer_id=layer_id,
                split=arm.split,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tail=tail,
                head_start=head_start,
            )
            retained = state.get("tensor")
            priorities = state.get("priorities")
            row_prompt_ids = state.get("row_prompt_ids")
            seen_rows = state.get("seen_rows_by_prompt")
            generations = state.get("generations")
            if (
                not isinstance(retained, torch.Tensor)
                or retained.dtype != torch.bfloat16
                or tuple(retained.shape[1:]) != tail
                or not isinstance(priorities, torch.Tensor)
                or priorities.dtype != torch.float64
                or priorities.shape != (retained.shape[0],)
                or not isinstance(row_prompt_ids, list)
                or len(row_prompt_ids) != retained.shape[0]
                or not isinstance(seen_rows, dict)
                or not isinstance(generations, list)
            ):
                raise ValueError(f"capture raw state payload is malformed: {path}")
            old_prompt_positions = [
                index
                for index, prompt_id in enumerate(row_prompt_ids)
                if prompt_id == arm.prompt_id
            ]
            other_positions = [
                index
                for index, prompt_id in enumerate(row_prompt_ids)
                if prompt_id != arm.prompt_id
            ]
            old_prompt_index = torch.tensor(old_prompt_positions, dtype=torch.int64)
            other_index = torch.tensor(other_positions, dtype=torch.int64)
            old_prompt_tensor = retained.index_select(0, old_prompt_index)
            old_prompt_priorities = priorities.index_select(0, old_prompt_index)
            prior_seen = seen_rows.get(arm.prompt_id, 0)
            if not isinstance(prior_seen, int) or prior_seen < 0:
                raise ValueError(f"capture seen-row counter is malformed: {path}")
            incoming_rows = tensor.shape[0]
            seed_material = (
                f"{self.config.session_id}:{kind}:{layer_id}:{arm.split}:"
                f"{arm.prompt_id}:{prior_seen}:{incoming_rows}"
            )
            seed = int.from_bytes(
                hashlib.sha256(seed_material.encode()).digest()[:8], "big"
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            incoming_priorities = torch.rand(
                incoming_rows, generator=generator, dtype=torch.float64
            )
            combined_priorities = torch.cat(
                (old_prompt_priorities, incoming_priorities), dim=0
            )
            limit = self.config.maximum_rows_per_prompt[kind][arm.split]
            keep_count = min(limit, combined_priorities.shape[0])
            keep = torch.topk(
                combined_priorities, keep_count, largest=False, sorted=True
            ).indices
            old_count = old_prompt_tensor.shape[0]
            selected_old = keep[keep < old_count]
            selected_new = keep[keep >= old_count] - old_count
            new_cpu = (
                tensor.index_select(0, selected_new.to(device=tensor.device))
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
            prompt_tensor = torch.cat(
                (old_prompt_tensor.index_select(0, selected_old), new_cpu), dim=0
            )
            prompt_priorities = torch.cat(
                (
                    old_prompt_priorities.index_select(0, selected_old),
                    incoming_priorities.index_select(0, selected_new),
                ),
                dim=0,
            )
            other_tensor = retained.index_select(0, other_index)
            other_priorities = priorities.index_select(0, other_index)
            state["tensor"] = torch.cat(
                (other_tensor, prompt_tensor), dim=0
            ).contiguous()
            state["priorities"] = torch.cat(
                (other_priorities, prompt_priorities), dim=0
            ).contiguous()
            state["row_prompt_ids"] = [
                cast(str, row_prompt_ids[index]) for index in other_positions
            ] + [arm.prompt_id] * prompt_tensor.shape[0]
            seen_rows[arm.prompt_id] = prior_seen + incoming_rows
            state["seen_rows_by_prompt"] = seen_rows
            if arm.generation not in generations:
                generations.append(arm.generation)
            state["generations"] = generations
            _atomic_torch_save(state, path)


_CAPTURER_LOCK = threading.Lock()
_CAPTURER: Dsv4OscarRuntimeCapturer | None = None
_CAPTURER_PATH: str | None = None


def _get_capturer() -> Dsv4OscarRuntimeCapturer | None:
    global _CAPTURER, _CAPTURER_PATH
    configured_path = _CAPTURE_CONFIG_PATH
    if not configured_path:
        return None
    if _CAPTURER is not None:
        if configured_path != _CAPTURER_PATH:
            raise RuntimeError("DSV4 OSCAR capture config changed after initialization")
        return _CAPTURER
    with _CAPTURER_LOCK:
        if _CAPTURER is None:
            path = Path(configured_path)
            _CAPTURER = Dsv4OscarRuntimeCapturer(_parse_config(path))
            _CAPTURER_PATH = configured_path
    return _CAPTURER


def maybe_capture_attention_query_nope(
    *,
    layer_id: int,
    query: torch.Tensor,
    forward_batch: object,
    target_model: bool,
    tp_rank: int,
    tp_size: int,
) -> None:
    if not _eligible_forward(forward_batch, target_model=target_model):
        return
    capturer = _get_capturer()
    if capturer is None:
        return
    capturer.record(
        kind="attention_query_nope",
        layer_id=layer_id,
        tensor=query[..., :LATENT_DIM],
        forward_batch=forward_batch,
        target_model=target_model,
        tp_rank=tp_rank,
        tp_size=tp_size,
        head_start=tp_rank * (NUM_ATTENTION_HEADS // tp_size),
    )


def maybe_capture_swa_latent(
    *,
    layer_id: int,
    shared_kv: torch.Tensor,
    forward_batch: object,
    target_model: bool,
    tp_rank: int,
    tp_size: int,
) -> None:
    if not _eligible_forward(forward_batch, target_model=target_model):
        return
    capturer = _get_capturer()
    if capturer is None:
        return
    if shared_kv.ndim == 3 and shared_kv.shape[1] == 1:
        shared_kv = shared_kv[:, 0]
    capturer.record(
        kind="swa_latent",
        layer_id=layer_id,
        tensor=shared_kv[..., :LATENT_DIM],
        forward_batch=forward_batch,
        target_model=target_model,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def maybe_capture_compressed_domain(
    *,
    layer_id: int,
    compressed: torch.Tensor,
    is_indexer: bool,
    forward_batch: object,
    target_model: bool,
    tp_rank: int,
    tp_size: int,
) -> None:
    if not _eligible_forward(forward_batch, target_model=target_model):
        return
    capturer = _get_capturer()
    if capturer is None:
        return
    capturer.record(
        kind="c4_scorer_key" if is_indexer else "compressed_latent",
        layer_id=layer_id,
        tensor=compressed if is_indexer else compressed[..., :LATENT_DIM],
        forward_batch=forward_batch,
        target_model=target_model,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def _apply_c4_scorer_weight(
    query: torch.Tensor, head_weight: torch.Tensor, weight_scale: float
) -> torch.Tensor:
    if head_weight.shape != query.shape[:2] or not head_weight.is_floating_point():
        raise ValueError("C4 scorer weight must have shape [rows,64]")
    query.mul_(head_weight.to(query.dtype).unsqueeze(-1))
    query.mul_(weight_scale)
    return query


def maybe_capture_c4_scorer_query(
    *,
    layer_id: int,
    query_before_rope: torch.Tensor,
    head_weight: torch.Tensor,
    weight_scale: float,
    positions: torch.Tensor,
    freqs_cis: torch.Tensor,
    forward_batch: object,
    target_model: bool,
    tp_rank: int,
    tp_size: int,
) -> None:
    if not _eligible_forward(forward_batch, target_model=target_model):
        return
    capturer = _get_capturer()
    if capturer is None or tp_rank != 0:
        return
    if query_before_rope.ndim != 3 or tuple(query_before_rope.shape[1:]) != (
        INDEX_HEADS,
        INDEX_HEAD_DIM,
    ):
        raise ValueError(
            "C4 scorer projection must be replicated [rows,64,128] before capture"
        )
    query = query_before_rope.clone()
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

    fused_rope_inplace(query[..., -64:], None, freqs_cis, positions=positions)
    # compute_weights(skip_scale=True) supplies the learned per-token/head
    # factor.  The production scorer folds that factor and weight_scale into
    # weights_out alongside the transient FP8 q_scale.  At the pre-Hadamard
    # BF16 calibration boundary q_scale does not exist, so the mathematically
    # equivalent effective query is Q_rope * learned_weight * weight_scale.
    _apply_c4_scorer_weight(query, head_weight, weight_scale)
    capturer.record(
        kind="c4_scorer_query",
        layer_id=layer_id,
        tensor=query,
        forward_batch=forward_batch,
        target_model=target_model,
        tp_rank=tp_rank,
        tp_size=tp_size,
        head_start=0,
    )
