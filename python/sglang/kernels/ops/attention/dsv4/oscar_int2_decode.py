"""Fused mixed-BF16/OSCAR-INT2 sparse MLA decode for DeepSeek V4 SM86.

The recent SWA tier is a calibrated, rotated BF16 protection window.  C4 and
C128 history use the model-specific asymmetric OSCAR INT2 byte layout.  This
kernel reads both tiers in the same rotated basis, performs one online softmax,
and returns the weighted value in that basis.  The caller must apply the
calibration's inverse shared-latent rotation before the normal output path.

No BF16 cache-sized workspace is materialized.  INT2 codes are unpacked and
affinely dequantized in registers before Ampere BF16 HMMA.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    GROUP_SIZE,
    HEAD_DIM,
    NOPE_DIM,
    ROPE_OFFSET_BYTES,
    SCALE_ZERO_OFFSET_BYTES,
    STORAGE_BYTES_PER_TOKEN,
)

LOG2E = math.log2(math.e)
BF16_BYTES_PER_TOKEN = HEAD_DIM * torch.bfloat16.itemsize

SPLIT_HISTORY_EXECUTION = "sm86-oscar-int2-split-history-fp32-online-v1"
SPLIT_HISTORY_SPLIT_MAP = {
    1: 16,
    2: 16,
    3: 8,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
    8: 2,
}
SPLIT_HISTORY_NUM_HEADS = 64
SPLIT_HISTORY_MAX_PARTIAL_ROWS = 32
SPLIT_HISTORY_ACCUMULATOR_FLOATS = (
    SPLIT_HISTORY_MAX_PARTIAL_ROWS * SPLIT_HISTORY_NUM_HEADS * HEAD_DIM
)
SPLIT_HISTORY_STAT_FLOATS = (
    SPLIT_HISTORY_MAX_PARTIAL_ROWS * SPLIT_HISTORY_NUM_HEADS
)
SPLIT_HISTORY_WORKSPACE_FLOATS = (
    SPLIT_HISTORY_ACCUMULATOR_FLOATS + 2 * SPLIT_HISTORY_STAT_FLOATS
)
SPLIT_HISTORY_WORKSPACE_BYTES = (
    SPLIT_HISTORY_WORKSPACE_FLOATS * torch.float32.itemsize
)
assert SPLIT_HISTORY_WORKSPACE_BYTES == 4_210_688


def oscar_int2_split_history_count(num_tokens: int) -> int | None:
    """Return the fixed, capture-stable split count for a decode token shape."""

    if not isinstance(num_tokens, int):
        raise TypeError("OSCAR split-history num_tokens must be an integer")
    split_count = SPLIT_HISTORY_SPLIT_MAP.get(num_tokens)
    if (
        split_count is not None
        and num_tokens * split_count > SPLIT_HISTORY_MAX_PARTIAL_ROWS
    ):
        raise RuntimeError("OSCAR split-history map exceeds its fixed workspace")
    return split_count


@dataclass(frozen=True)
class OscarInt2SplitHistoryWorkspace:
    """Persistent views over the backend-owned 4,210,688-byte FP32 arena."""

    storage: torch.Tensor
    accumulator: torch.Tensor
    maxima: torch.Tensor
    sums: torch.Tensor
    fixed_data_ptr: int

    @classmethod
    def from_backend_storage(
        cls, storage: torch.Tensor
    ) -> OscarInt2SplitHistoryWorkspace:
        if (
            storage.dtype != torch.float32
            or storage.ndim != 1
            or not storage.is_contiguous()
            or storage.numel() != SPLIT_HISTORY_WORKSPACE_FLOATS
        ):
            raise ValueError(
                "OSCAR split-history storage must be a contiguous FP32 arena of "
                f"exactly {SPLIT_HISTORY_WORKSPACE_BYTES} bytes"
            )
        accumulator_end = SPLIT_HISTORY_ACCUMULATOR_FLOATS
        maxima_end = accumulator_end + SPLIT_HISTORY_STAT_FLOATS
        accumulator = storage[:accumulator_end].view(
            SPLIT_HISTORY_MAX_PARTIAL_ROWS,
            SPLIT_HISTORY_NUM_HEADS,
            HEAD_DIM,
        )
        maxima = storage[accumulator_end:maxima_end].view(
            SPLIT_HISTORY_MAX_PARTIAL_ROWS,
            SPLIT_HISTORY_NUM_HEADS,
        )
        sums = storage[maxima_end:].view(
            SPLIT_HISTORY_MAX_PARTIAL_ROWS,
            SPLIT_HISTORY_NUM_HEADS,
        )
        workspace = cls(
            storage=storage,
            accumulator=accumulator,
            maxima=maxima,
            sums=sums,
            fixed_data_ptr=storage.data_ptr(),
        )
        workspace.validate(device=storage.device)
        return workspace

    def validate(self, *, device: torch.device) -> None:
        expected_device = torch.device(device)
        # ModelRunner commonly publishes the rank-local device as the
        # unindexed ``cuda`` alias.  Tensor.device is always concrete
        # (``cuda:0``, ``cuda:1``), so direct equality rejects a correctly
        # allocated workspace during the scheduler's post-graph telemetry
        # handshake.  Resolve only that alias through the process-local
        # current device; never accept a different concrete GPU.
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if self.storage.device != expected_device:
            raise ValueError("OSCAR split-history workspace is on the wrong device")
        if self.storage.data_ptr() != self.fixed_data_ptr:
            raise RuntimeError("OSCAR split-history workspace address changed")
        if self.storage.numel() * self.storage.element_size() != (
            SPLIT_HISTORY_WORKSPACE_BYTES
        ):
            raise RuntimeError("OSCAR split-history workspace byte size changed")
        if (
            self.accumulator.shape
            != (
                SPLIT_HISTORY_MAX_PARTIAL_ROWS,
                SPLIT_HISTORY_NUM_HEADS,
                HEAD_DIM,
            )
            or self.maxima.shape
            != (SPLIT_HISTORY_MAX_PARTIAL_ROWS, SPLIT_HISTORY_NUM_HEADS)
            or self.sums.shape
            != (SPLIT_HISTORY_MAX_PARTIAL_ROWS, SPLIT_HISTORY_NUM_HEADS)
        ):
            raise RuntimeError("OSCAR split-history workspace views changed shape")
        if any(
            tensor.dtype != torch.float32 or not tensor.is_contiguous()
            for tensor in (self.accumulator, self.maxima, self.sums)
        ):
            raise RuntimeError("OSCAR split-history workspace views changed layout")
        float_bytes = torch.float32.itemsize
        if self.accumulator.data_ptr() != self.fixed_data_ptr:
            raise RuntimeError("OSCAR split-history accumulator address changed")
        if self.maxima.data_ptr() != (
            self.fixed_data_ptr + SPLIT_HISTORY_ACCUMULATOR_FLOATS * float_bytes
        ):
            raise RuntimeError("OSCAR split-history maxima address changed")
        if self.sums.data_ptr() != (
            self.fixed_data_ptr
            + (SPLIT_HISTORY_ACCUMULATOR_FLOATS + SPLIT_HISTORY_STAT_FLOATS)
            * float_bytes
        ):
            raise RuntimeError("OSCAR split-history sums address changed")


@functools.cache
def _decode_launch_config(device: torch.device) -> tuple[int, int, int]:
    """Return a setup-time cached SM86 launch geometry."""

    if device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 6):
        return 8, 16, 4
    return 8, 16, 8


@triton.jit
def _decode_sparse_attention_oscar_int2_kernel(
    q_ptr,
    swa_cache_bf16_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_u8_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    sink_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    swa_index_topk: tl.constexpr,
    extra_index_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_stride_block_bf16: tl.constexpr,
    extra_stride_block_bytes: tl.constexpr,
    extra_stride_block_bf16: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_swa_idx_t: tl.constexpr,
    stride_swa_idx_k: tl.constexpr,
    stride_extra_idx_t: tl.constexpr,
    stride_extra_idx_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    nope_d: tl.constexpr,
    group_d: tl.constexpr,
    token_bytes: tl.constexpr,
    rope_offset: tl.constexpr,
    scale_zero_offset: tl.constexpr,
    has_sink: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * block_h + tl.arange(0, block_h)
    dims = tl.arange(0, block_d)
    head_mask = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + dims[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )
    if has_sink:
        sink = tl.load(sink_ptr + heads, mask=head_mask, other=-float("inf"))
        running_max = sink * 1.4426950408889634
        running_sum = tl.where(head_mask, 1.0, 0.0)
    else:
        running_max = tl.full((block_h,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_h,), tl.float32)
    accumulator = tl.zeros((block_h, block_d), tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id)
    total_len = swa_len + extra_len
    loop_end = tl.cdiv(total_len, block_n) * block_n
    for start in range(0, loop_end, block_n):
        offsets = start + tl.arange(0, block_n)
        use_extra = offsets < extra_len
        use_swa = (offsets >= extra_len) & (offsets < total_len)
        extra_cols = offsets
        swa_cols = offsets - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_idx_t
            + extra_cols * stride_extra_idx_k,
            mask=extra_cols < extra_index_topk,
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr + token_id * stride_swa_idx_t + swa_cols * stride_swa_idx_k,
            mask=(swa_cols >= 0) & (swa_cols < swa_index_topk),
            other=-1,
        )

        extra_block = extra_idx // extra_block_size
        extra_position = extra_idx - extra_block * extra_block_size
        swa_block = swa_idx // swa_block_size
        swa_position = swa_idx - swa_block * swa_block_size
        valid_extra = (
            use_extra
            & (extra_idx >= 0)
            & (extra_block >= 0)
            & (extra_block < extra_num_blocks)
        )
        valid_swa = (
            use_swa & (swa_idx >= 0) & (swa_block >= 0) & (swa_block < swa_num_blocks)
        )
        valid = valid_extra | valid_swa

        safe_extra_block = tl.where(valid_extra, extra_block, 0)
        safe_extra_position = tl.where(valid_extra, extra_position, 0)
        extra_token_base_bytes = (
            safe_extra_block * extra_stride_block_bytes
            + safe_extra_position * token_bytes
        )
        extra_token_base_bf16 = (
            safe_extra_block * extra_stride_block_bf16
            + safe_extra_position * (token_bytes // 2)
        )

        is_nope = dims < nope_d
        packed = tl.load(
            extra_cache_u8_ptr + extra_token_base_bytes[:, None] + dims[None, :] // 4,
            mask=valid_extra[:, None] & is_nope[None, :],
            other=0,
        ).to(tl.uint32)
        shifts = (dims & 3) * 2
        codes = ((packed >> shifts[None, :]) & 0x03).to(tl.float32)

        metadata_base = extra_token_base_bf16 + scale_zero_offset // 2
        scale_0 = tl.load(
            extra_cache_bf16_ptr + metadata_base,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_0 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 1,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)
        scale_1 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 2,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_1 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 3,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)
        scale_2 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 4,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_2 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 5,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)
        scale_3 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 6,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_3 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 7,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)
        scale_4 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 8,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_4 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 9,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)
        scale_5 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 10,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_5 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 11,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)
        scale_6 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 12,
            mask=valid_extra,
            other=1.0,
        ).to(tl.float32)
        zero_6 = tl.load(
            extra_cache_bf16_ptr + metadata_base + 13,
            mask=valid_extra,
            other=0.0,
        ).to(tl.float32)

        group = dims // group_d
        scales = tl.where(group[None, :] == 0, scale_0[:, None], 1.0)
        zeros = tl.where(group[None, :] == 0, zero_0[:, None], 0.0)
        scales = tl.where(group[None, :] == 1, scale_1[:, None], scales)
        zeros = tl.where(group[None, :] == 1, zero_1[:, None], zeros)
        scales = tl.where(group[None, :] == 2, scale_2[:, None], scales)
        zeros = tl.where(group[None, :] == 2, zero_2[:, None], zeros)
        scales = tl.where(group[None, :] == 3, scale_3[:, None], scales)
        zeros = tl.where(group[None, :] == 3, zero_3[:, None], zeros)
        scales = tl.where(group[None, :] == 4, scale_4[:, None], scales)
        zeros = tl.where(group[None, :] == 4, zero_4[:, None], zeros)
        scales = tl.where(group[None, :] == 5, scale_5[:, None], scales)
        zeros = tl.where(group[None, :] == 5, zero_5[:, None], zeros)
        scales = tl.where(group[None, :] == 6, scale_6[:, None], scales)
        zeros = tl.where(group[None, :] == 6, zero_6[:, None], zeros)
        extra_nope = (codes - zeros) * scales
        extra_rope = tl.load(
            extra_cache_bf16_ptr
            + extra_token_base_bf16[:, None]
            + rope_offset // 2
            + dims[None, :]
            - nope_d,
            mask=valid_extra[:, None] & (~is_nope[None, :]),
            other=0.0,
        ).to(tl.float32)
        extra_k = tl.where(is_nope[None, :], extra_nope, extra_rope)

        safe_swa_block = tl.where(valid_swa, swa_block, 0)
        safe_swa_position = tl.where(valid_swa, swa_position, 0)
        swa_token_base = (
            safe_swa_block * swa_stride_block_bf16 + safe_swa_position * block_d
        )
        swa_k = tl.load(
            swa_cache_bf16_ptr + swa_token_base[:, None] + dims[None, :],
            mask=valid_swa[:, None],
            other=0.0,
        ).to(tl.float32)
        k = tl.where(use_extra[:, None], extra_k, swa_k).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
        qk = tl.where(
            head_mask[:, None] & valid[None, :],
            qk,
            -3.4028234663852886e38,
        )
        next_max = tl.maximum(tl.max(qk, axis=1), running_max)
        rescale = tl.exp2(running_max - next_max)
        probability = tl.exp2(qk - next_max[:, None])
        probability = tl.where(head_mask[:, None] & valid[None, :], probability, 0.0)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(tl.bfloat16), k
        )
        running_sum = running_sum * rescale + tl.sum(probability, axis=1)
        running_max = next_max

    accumulator /= tl.maximum(running_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + dims[None, :] * stride_out_d,
        accumulator.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


@triton.jit
def _decode_sparse_attention_oscar_int2_split_stage1_kernel(
    q_ptr,
    swa_cache_bf16_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_u8_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    partial_accumulator_ptr,
    partial_maxima_ptr,
    partial_sums_ptr,
    num_heads: tl.constexpr,
    swa_index_topk: tl.constexpr,
    extra_index_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_stride_block_bf16: tl.constexpr,
    extra_stride_block_bytes: tl.constexpr,
    extra_stride_block_bf16: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_swa_idx_t: tl.constexpr,
    stride_swa_idx_k: tl.constexpr,
    stride_extra_idx_t: tl.constexpr,
    stride_extra_idx_k: tl.constexpr,
    num_splits: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    nope_d: tl.constexpr,
    group_d: tl.constexpr,
    token_bytes: tl.constexpr,
    rope_offset: tl.constexpr,
    scale_zero_offset: tl.constexpr,
):
    """Produce sink-free, unnormalised FP32 online-softmax partials."""

    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    split_id = tl.program_id(2)
    heads = head_block * block_h + tl.arange(0, block_h)
    dims = tl.arange(0, block_d)
    head_mask = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + dims[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )
    running_max = tl.full((block_h,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_h,), tl.float32)
    accumulator = tl.zeros((block_h, block_d), tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id)
    total_len = swa_len + extra_len
    split_span = tl.cdiv(total_len, num_splits)
    split_start = split_span * split_id
    split_end = tl.minimum(split_start + split_span, total_len)

    if split_end > split_start:
        for start in tl.range(split_start, split_end, block_n, num_stages=1):
            offsets = start + tl.arange(0, block_n)
            in_split = offsets < split_end
            use_extra = in_split & (offsets < extra_len)
            use_swa = in_split & (offsets >= extra_len) & (offsets < total_len)
            extra_cols = offsets
            swa_cols = offsets - extra_len
            extra_idx = tl.load(
                extra_indices_ptr
                + token_id * stride_extra_idx_t
                + extra_cols * stride_extra_idx_k,
                mask=in_split & (extra_cols < extra_index_topk),
                other=-1,
            )
            swa_idx = tl.load(
                swa_indices_ptr
                + token_id * stride_swa_idx_t
                + swa_cols * stride_swa_idx_k,
                mask=in_split & (swa_cols >= 0) & (swa_cols < swa_index_topk),
                other=-1,
            )

            extra_block = extra_idx // extra_block_size
            extra_position = extra_idx - extra_block * extra_block_size
            swa_block = swa_idx // swa_block_size
            swa_position = swa_idx - swa_block * swa_block_size
            valid_extra = (
                use_extra
                & (extra_idx >= 0)
                & (extra_block >= 0)
                & (extra_block < extra_num_blocks)
            )
            valid_swa = (
                use_swa
                & (swa_idx >= 0)
                & (swa_block >= 0)
                & (swa_block < swa_num_blocks)
            )
            valid = valid_extra | valid_swa

            safe_extra_block = tl.where(valid_extra, extra_block, 0)
            safe_extra_position = tl.where(valid_extra, extra_position, 0)
            extra_token_base_bytes = (
                safe_extra_block * extra_stride_block_bytes
                + safe_extra_position * token_bytes
            )
            extra_token_base_bf16 = (
                safe_extra_block * extra_stride_block_bf16
                + safe_extra_position * (token_bytes // 2)
            )

            is_nope = dims < nope_d
            packed = tl.load(
                extra_cache_u8_ptr
                + extra_token_base_bytes[:, None]
                + dims[None, :] // 4,
                mask=valid_extra[:, None] & is_nope[None, :],
                other=0,
            ).to(tl.uint32)
            shifts = (dims & 3) * 2
            codes = ((packed >> shifts[None, :]) & 0x03).to(tl.float32)

            metadata_base = extra_token_base_bf16 + scale_zero_offset // 2
            scale_0 = tl.load(
                extra_cache_bf16_ptr + metadata_base,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_0 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 1,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)
            scale_1 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 2,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_1 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 3,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)
            scale_2 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 4,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_2 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 5,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)
            scale_3 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 6,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_3 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 7,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)
            scale_4 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 8,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_4 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 9,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)
            scale_5 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 10,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_5 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 11,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)
            scale_6 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 12,
                mask=valid_extra,
                other=1.0,
            ).to(tl.float32)
            zero_6 = tl.load(
                extra_cache_bf16_ptr + metadata_base + 13,
                mask=valid_extra,
                other=0.0,
            ).to(tl.float32)

            group = dims // group_d
            scales = tl.where(group[None, :] == 0, scale_0[:, None], 1.0)
            zeros = tl.where(group[None, :] == 0, zero_0[:, None], 0.0)
            scales = tl.where(group[None, :] == 1, scale_1[:, None], scales)
            zeros = tl.where(group[None, :] == 1, zero_1[:, None], zeros)
            scales = tl.where(group[None, :] == 2, scale_2[:, None], scales)
            zeros = tl.where(group[None, :] == 2, zero_2[:, None], zeros)
            scales = tl.where(group[None, :] == 3, scale_3[:, None], scales)
            zeros = tl.where(group[None, :] == 3, zero_3[:, None], zeros)
            scales = tl.where(group[None, :] == 4, scale_4[:, None], scales)
            zeros = tl.where(group[None, :] == 4, zero_4[:, None], zeros)
            scales = tl.where(group[None, :] == 5, scale_5[:, None], scales)
            zeros = tl.where(group[None, :] == 5, zero_5[:, None], zeros)
            scales = tl.where(group[None, :] == 6, scale_6[:, None], scales)
            zeros = tl.where(group[None, :] == 6, zero_6[:, None], zeros)
            extra_nope = (codes - zeros) * scales
            extra_rope = tl.load(
                extra_cache_bf16_ptr
                + extra_token_base_bf16[:, None]
                + rope_offset // 2
                + dims[None, :]
                - nope_d,
                mask=valid_extra[:, None] & (~is_nope[None, :]),
                other=0.0,
            ).to(tl.float32)
            extra_k = tl.where(is_nope[None, :], extra_nope, extra_rope)

            safe_swa_block = tl.where(valid_swa, swa_block, 0)
            safe_swa_position = tl.where(valid_swa, swa_position, 0)
            swa_token_base = (
                safe_swa_block * swa_stride_block_bf16
                + safe_swa_position * block_d
            )
            swa_k = tl.load(
                swa_cache_bf16_ptr + swa_token_base[:, None] + dims[None, :],
                mask=valid_swa[:, None],
                other=0.0,
            ).to(tl.float32)
            k = tl.where(use_extra[:, None], extra_k, swa_k).to(tl.bfloat16)

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
            qk = tl.where(
                head_mask[:, None] & valid[None, :],
                qk,
                -3.4028234663852886e38,
            )
            next_max = tl.maximum(tl.max(qk, axis=1), running_max)
            rescale = tl.exp2(running_max - next_max)
            probability = tl.exp2(qk - next_max[:, None])
            probability = tl.where(
                head_mask[:, None] & valid[None, :], probability, 0.0
            )
            accumulator = accumulator * rescale[:, None] + tl.dot(
                probability.to(tl.bfloat16), k
            )
            running_sum = running_sum * rescale + tl.sum(probability, axis=1)
            running_max = next_max

    partial_row = token_id * num_splits + split_id
    tl.store(
        partial_accumulator_ptr
        + partial_row * num_heads * block_d
        + heads[:, None] * block_d
        + dims[None, :],
        accumulator,
        mask=head_mask[:, None],
    )
    tl.store(
        partial_maxima_ptr + partial_row * num_heads + heads,
        running_max,
        mask=head_mask,
    )
    tl.store(
        partial_sums_ptr + partial_row * num_heads + heads,
        running_sum,
        mask=head_mask,
    )


@triton.jit
def _decode_sparse_attention_oscar_int2_split_stage2_kernel(
    partial_accumulator_ptr,
    partial_maxima_ptr,
    partial_sums_ptr,
    sink_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    num_splits: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    block_h: tl.constexpr,
    block_d: tl.constexpr,
):
    """Merge split IDs in order and introduce the attention sink exactly once."""

    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * block_h + tl.arange(0, block_h)
    dims = tl.arange(0, block_d)
    head_mask = heads < num_heads

    sink = tl.load(sink_ptr + heads, mask=head_mask, other=-float("inf"))
    running_max = sink * 1.4426950408889634
    running_sum = tl.where(head_mask, 1.0, 0.0)
    accumulator = tl.zeros((block_h, block_d), tl.float32)

    for split_id in tl.range(0, num_splits, num_stages=1):
        partial_row = token_id * num_splits + split_id
        partial_max = tl.load(
            partial_maxima_ptr + partial_row * num_heads + heads,
            mask=head_mask,
            other=-float("inf"),
        )
        partial_sum = tl.load(
            partial_sums_ptr + partial_row * num_heads + heads,
            mask=head_mask,
            other=0.0,
        )
        partial_accumulator = tl.load(
            partial_accumulator_ptr
            + partial_row * num_heads * block_d
            + heads[:, None] * block_d
            + dims[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        next_max = tl.maximum(running_max, partial_max)
        running_rescale = tl.exp2(running_max - next_max)
        partial_rescale = tl.where(
            partial_sum > 0.0,
            tl.exp2(partial_max - next_max),
            0.0,
        )
        accumulator = (
            accumulator * running_rescale[:, None]
            + partial_accumulator * partial_rescale[:, None]
        )
        running_sum = (
            running_sum * running_rescale + partial_sum * partial_rescale
        )
        running_max = next_max

    accumulator /= tl.maximum(running_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + dims[None, :] * stride_out_d,
        accumulator.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


def _canonical_bf16_cache(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    cache_u8 = cache.view(torch.uint8).reshape(cache.shape[0], -1)
    expected = page_size * BF16_BYTES_PER_TOKEN
    if cache_u8.dtype != torch.uint8 or cache_u8.shape[1] != expected:
        raise ValueError(
            f"protected BF16 SWA page must contain exactly {expected} bytes, "
            f"got {tuple(cache_u8.shape)}"
        )
    if cache_u8.stride(1) != 1 or cache_u8.stride(0) % 2:
        raise ValueError("protected BF16 SWA pages must be contiguous and BF16 aligned")
    return cache_u8.view(torch.bfloat16)


def _canonical_oscar_cache(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    cache_u8 = cache.view(torch.uint8).reshape(cache.shape[0], -1)
    expected = page_size * STORAGE_BYTES_PER_TOKEN
    if cache_u8.dtype != torch.uint8 or cache_u8.shape[1] != expected:
        raise ValueError(
            f"OSCAR INT2 history page must contain exactly {expected} bytes, "
            f"got {tuple(cache_u8.shape)}"
        )
    if cache_u8.stride(1) != 1 or cache_u8.stride(0) % 2:
        raise ValueError("OSCAR INT2 history pages must be contiguous and BF16 aligned")
    return cache_u8


def decode_sparse_attention_oscar_int2(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    swa_block_size: int,
    *,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lens: torch.Tensor,
    extra_block_size: int,
    split_workspace: OscarInt2SplitHistoryWorkspace | None = None,
) -> None:
    """Run allocation-free mixed-tier OSCAR sparse attention into ``out``."""

    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if torch.cuda.get_device_capability(q.device) != (8, 6):
        raise RuntimeError("DSV4 OSCAR-INT2 decode is implemented only for exact SM86")
    if q.dtype != torch.bfloat16 or head_dim != HEAD_DIM or not q.is_contiguous():
        raise ValueError(f"q must be contiguous BF16 [T, H, {HEAD_DIM}]")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 with the same shape as q")
    if not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("OSCAR sparse-attention scale must be a positive float")
    if not isinstance(swa_block_size, int) or swa_block_size <= 0:
        raise ValueError("swa_block_size must be a positive integer")
    if not isinstance(extra_block_size, int) or extra_block_size <= 0:
        raise ValueError("extra_block_size must be a positive integer")
    if (
        swa_indices.ndim != 2
        or extra_indices.ndim != 2
        or swa_indices.shape[0] != num_tokens
        or extra_indices.shape[0] != num_tokens
        or swa_lens.shape != (num_tokens,)
        or extra_lens.shape != (num_tokens,)
    ):
        raise ValueError("OSCAR sparse metadata shapes do not match q tokens")
    metadata = (swa_indices, swa_lens, extra_indices, extra_lens)
    if any(tensor.dtype != torch.int32 for tensor in metadata):
        raise ValueError("OSCAR sparse metadata must use int32")
    if attn_sink is not None and (
        attn_sink.shape != (num_heads,)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError("attn_sink must be contiguous FP32 with one value per head")

    swa_bf16 = _canonical_bf16_cache(swa_cache, swa_block_size)
    extra_u8 = _canonical_oscar_cache(extra_cache, extra_block_size)
    extra_bf16 = extra_u8.view(torch.bfloat16)
    tensors = (
        *metadata,
        swa_bf16,
        extra_u8,
        out,
        *((attn_sink,) if attn_sink is not None else ()),
        *(
            (
                split_workspace.storage,
                split_workspace.accumulator,
                split_workspace.maxima,
                split_workspace.sums,
            )
            if split_workspace is not None
            else ()
        ),
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all OSCAR sparse-attention tensors must share one device")

    block_h, block_n, num_warps = _decode_launch_config(q.device)
    if split_workspace is not None:
        num_splits = oscar_int2_split_history_count(num_tokens)
        if num_splits is None:
            raise ValueError(
                "OSCAR split-history workspace was supplied for an unsupported "
                f"token shape T={num_tokens}"
            )
        if num_heads != SPLIT_HISTORY_NUM_HEADS:
            raise ValueError(
                "OSCAR split-history decode requires exactly "
                f"{SPLIT_HISTORY_NUM_HEADS} heads"
            )
        if attn_sink is None:
            raise ValueError("OSCAR split-history stage 2 requires the attention sink")
        split_workspace.validate(device=q.device)
        _decode_sparse_attention_oscar_int2_split_stage1_kernel[
            (
                num_tokens,
                triton.cdiv(num_heads, block_h),
                num_splits,
            )
        ](
            q,
            swa_bf16,
            swa_indices,
            swa_lens,
            extra_u8,
            extra_bf16,
            extra_indices,
            extra_lens,
            split_workspace.accumulator,
            split_workspace.maxima,
            split_workspace.sums,
            num_heads=num_heads,
            swa_index_topk=swa_indices.shape[-1],
            extra_index_topk=extra_indices.shape[-1],
            swa_num_blocks=swa_bf16.shape[0],
            extra_num_blocks=extra_u8.shape[0],
            swa_block_size=swa_block_size,
            extra_block_size=extra_block_size,
            swa_stride_block_bf16=swa_bf16.stride(0),
            extra_stride_block_bytes=extra_u8.stride(0),
            extra_stride_block_bf16=extra_bf16.stride(0),
            sm_scale_log2=scale * LOG2E,
            stride_qt=q.stride(0),
            stride_qh=q.stride(1),
            stride_qd=q.stride(2),
            stride_swa_idx_t=swa_indices.stride(0),
            stride_swa_idx_k=swa_indices.stride(1),
            stride_extra_idx_t=extra_indices.stride(0),
            stride_extra_idx_k=extra_indices.stride(1),
            num_splits=num_splits,
            block_h=block_h,
            block_n=block_n,
            block_d=HEAD_DIM,
            nope_d=NOPE_DIM,
            group_d=GROUP_SIZE,
            token_bytes=STORAGE_BYTES_PER_TOKEN,
            rope_offset=ROPE_OFFSET_BYTES,
            scale_zero_offset=SCALE_ZERO_OFFSET_BYTES,
            num_stages=1,
            num_warps=num_warps,
        )
        _decode_sparse_attention_oscar_int2_split_stage2_kernel[
            (num_tokens, triton.cdiv(num_heads, block_h))
        ](
            split_workspace.accumulator,
            split_workspace.maxima,
            split_workspace.sums,
            attn_sink,
            out,
            num_heads=num_heads,
            num_splits=num_splits,
            stride_out_t=out.stride(0),
            stride_out_h=out.stride(1),
            stride_out_d=out.stride(2),
            block_h=block_h,
            block_d=HEAD_DIM,
            num_stages=1,
            num_warps=num_warps,
        )
        return

    _decode_sparse_attention_oscar_int2_kernel[
        (num_tokens, triton.cdiv(num_heads, block_h))
    ](
        q,
        swa_bf16,
        swa_indices,
        swa_lens,
        extra_u8,
        extra_bf16,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_heads=num_heads,
        swa_index_topk=swa_indices.shape[-1],
        extra_index_topk=extra_indices.shape[-1],
        swa_num_blocks=swa_bf16.shape[0],
        extra_num_blocks=extra_u8.shape[0],
        swa_block_size=swa_block_size,
        extra_block_size=extra_block_size,
        swa_stride_block_bf16=swa_bf16.stride(0),
        extra_stride_block_bytes=extra_u8.stride(0),
        extra_stride_block_bf16=extra_bf16.stride(0),
        sm_scale_log2=scale * LOG2E,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_swa_idx_t=swa_indices.stride(0),
        stride_swa_idx_k=swa_indices.stride(1),
        stride_extra_idx_t=extra_indices.stride(0),
        stride_extra_idx_k=extra_indices.stride(1),
        stride_out_t=out.stride(0),
        stride_out_h=out.stride(1),
        stride_out_d=out.stride(2),
        block_h=block_h,
        block_n=block_n,
        block_d=HEAD_DIM,
        nope_d=NOPE_DIM,
        group_d=GROUP_SIZE,
        token_bytes=STORAGE_BYTES_PER_TOKEN,
        rope_offset=ROPE_OFFSET_BYTES,
        scale_zero_offset=SCALE_ZERO_OFFSET_BYTES,
        has_sink=attn_sink is not None,
        num_stages=1,
        num_warps=num_warps,
    )


__all__ = [
    "SPLIT_HISTORY_EXECUTION",
    "SPLIT_HISTORY_MAX_PARTIAL_ROWS",
    "SPLIT_HISTORY_SPLIT_MAP",
    "SPLIT_HISTORY_WORKSPACE_BYTES",
    "OscarInt2SplitHistoryWorkspace",
    "decode_sparse_attention_oscar_int2",
    "oscar_int2_split_history_count",
]
