"""Compact Triton W8A16 batched matmul for absorbed MLA ``kv_b_proj``.

The preferred entry point consumes the same GPTQ row-packed checkpoint bytes
that Marlin can repack at load time, without materializing a floating-point
weight:

* ``qweight`` is ``torch.int32`` with shape ``[H, K / 4, N]``.  Word
  ``qweight[h, k // 4, n]`` stores four biased INT8 values along ``K`` in
  little-endian bit fields: bits ``[8*i, 8*i+7]`` hold the byte for
  ``k = 4 * (k // 4) + i``.
* ``scales`` is BF16 with shape ``[H, 1, N]``.  Quantization is symmetric and
  channelwise along ``K``.
* ``x`` is BF16 with shape ``[M, H, K]``.
* the result is BF16 with shape ``[H, M, N]``.

Offline conversion should compute each stored scale first (rounded to BF16),
then quantize with that stored scale into ``[-128, 127]`` and add 128.  The
four biased bytes are packed into an INT32 along ``K``.  This is the standard
GPTQ ``pack_rows`` W8 layout accepted by ``gptq_marlin_moe_repack``.  Thus an
immutable checkpoint can serve both Marlin and Triton.  The serialized
footprint is one byte per weight plus one BF16 scale per output channel.  The
launch bit-extracts and dequantizes only a tile in registers, then feeds it
directly to a BF16 tensor-core dot with FP32 accumulation; it never persists an
unpacked or BF16 weight.

Unlike the generic Triton MoE path, attention heads have a fixed one-to-one
mapping and need no token sorting or padding.  The launch grid is exactly
``(ceil(M / BLOCK_M), ceil(N / BLOCK_N), H)``.

``mla_kv_b_w8a16_bmm`` remains available for raw ``torch.uint8 [H,K,N]``
experiments.  Production checkpoints should use
``mla_kv_b_gptq_w8a16_bmm``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


class _LaunchConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


def _select_launch_config(token_count: int) -> _LaunchConfig:
    """Select an Ampere-oriented configuration without runtime autotuning."""

    if token_count <= 4:
        return _LaunchConfig(16, 64, 32, 4, 3)
    if token_count <= 32:
        return _LaunchConfig(32, 64, 32, 4, 3)
    return _LaunchConfig(64, 64, 32, 8, 3)


@triton.jit
def _mla_kv_b_w8a16_bmm_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    output_ptr,
    token_count,
    output_features: tl.constexpr,
    input_features: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_xk,
    stride_wh,
    stride_wk,
    stride_wn,
    stride_sh,
    stride_sn,
    stride_oh,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GPTQ_PACKED: tl.constexpr,
):
    token_block = tl.program_id(axis=0)
    output_block = tl.program_id(axis=1)
    head = tl.program_id(axis=2)

    token_offsets = token_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    output_offsets = output_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    input_offsets = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = (
        x_ptr
        + token_offsets[:, None] * stride_xm
        + head * stride_xh
        + input_offsets[None, :] * stride_xk
    )
    if GPTQ_PACKED:
        packed_input_offsets = tl.arange(0, BLOCK_SIZE_K // 4)
        qweight_ptrs = (
            qweight_ptr
            + head * stride_wh
            + packed_input_offsets[:, None] * stride_wk
            + output_offsets[None, :] * stride_wn
        )
    else:
        qweight_ptrs = (
            qweight_ptr
            + head * stride_wh
            + input_offsets[:, None] * stride_wk
            + output_offsets[None, :] * stride_wn
        )
    scale = tl.load(
        scales_ptr + head * stride_sh + output_offsets * stride_sn,
        mask=output_offsets < output_features,
        other=0.0,
    ).to(tl.float32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for input_start in range(0, input_features, BLOCK_SIZE_K):
        input_mask = input_offsets < input_features - input_start
        activations = tl.load(
            x_ptrs,
            mask=(token_offsets[:, None] < token_count) & input_mask[None, :],
            other=0.0,
        )
        if GPTQ_PACKED:
            packed_input_mask = (
                packed_input_offsets < (input_features - input_start) // 4
            )
            packed_weight = tl.load(
                qweight_ptrs,
                mask=packed_input_mask[:, None]
                & (output_offsets[None, :] < output_features),
                other=0,
            )
            # Repeat each packed word four times in K order, then select its
            # little-endian byte.  The broadcast/reshape is register-only.
            expanded_weight = tl.broadcast_to(
                packed_weight[:, None, :],
                (BLOCK_SIZE_K // 4, 4, BLOCK_SIZE_N),
            )
            expanded_weight = tl.reshape(
                expanded_weight,
                (BLOCK_SIZE_K, BLOCK_SIZE_N),
            )
            shifts = (input_offsets % 4) * 8
            biased_weight = (expanded_weight >> shifts[:, None]) & 0xFF
        else:
            biased_weight = tl.load(
                qweight_ptrs,
                mask=input_mask[:, None] & (output_offsets[None, :] < output_features),
                other=128,
            )
        signed_weight = biased_weight.to(tl.int32) - 128
        dequantized_weight = (signed_weight.to(tl.float32) * scale[None, :]).to(
            tl.bfloat16
        )
        accumulator = tl.dot(activations, dequantized_weight, acc=accumulator)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        if GPTQ_PACKED:
            qweight_ptrs += (BLOCK_SIZE_K // 4) * stride_wk
        else:
            qweight_ptrs += BLOCK_SIZE_K * stride_wk

    output_offsets_2d = (
        head * stride_oh
        + token_offsets[:, None] * stride_om
        + output_offsets[None, :] * stride_on
    )
    output_mask = (token_offsets[:, None] < token_count) & (
        output_offsets[None, :] < output_features
    )
    tl.store(output_ptr + output_offsets_2d, accumulator, mask=output_mask)


def _validate_inputs(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    gptq_packed: bool,
) -> tuple[int, int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [M, H, K], got {tuple(x.shape)}")
    if qweight.ndim != 3:
        layout = "[H, K / 4, N]" if gptq_packed else "[H, K, N]"
        raise ValueError(
            f"qweight must have shape {layout}, got {tuple(qweight.shape)}"
        )
    if scales.ndim != 3 or scales.shape[1] != 1:
        raise ValueError(
            f"scales must have channelwise shape [H, 1, N], got {tuple(scales.shape)}"
        )
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be BF16, got {x.dtype}")
    expected_qweight_dtype = torch.int32 if gptq_packed else torch.uint8
    if qweight.dtype != expected_qweight_dtype:
        raise TypeError(
            f"qweight must be {expected_qweight_dtype}, got {qweight.dtype}"
        )
    if scales.dtype != torch.bfloat16:
        raise TypeError(f"scales must be BF16, got {scales.dtype}")
    if not x.is_cuda or not qweight.is_cuda or not scales.is_cuda:
        raise ValueError("x, qweight, and scales must all be CUDA tensors")
    if x.device != qweight.device or x.device != scales.device:
        raise ValueError(
            "x, qweight, and scales must be on the same CUDA device, "
            f"got x={x.device}, qweight={qweight.device}, scales={scales.device}"
        )

    token_count, head_count, input_features = x.shape
    weight_heads, stored_input_features, output_features = qweight.shape
    weight_input_features = (
        stored_input_features * 4 if gptq_packed else stored_input_features
    )
    if gptq_packed and input_features % 4 != 0:
        raise ValueError(
            f"GPTQ W8 row packing requires K divisible by 4, got K={input_features}"
        )
    if head_count != weight_heads or input_features != weight_input_features:
        raise ValueError(
            "x and qweight dimensions disagree: "
            f"x={tuple(x.shape)}, qweight={tuple(qweight.shape)}"
        )
    if scales.shape != (head_count, 1, output_features):
        raise ValueError(
            "qweight and scales dimensions disagree: "
            f"qweight={tuple(qweight.shape)}, scales={tuple(scales.shape)}"
        )
    if head_count == 0 or input_features == 0 or output_features == 0:
        raise ValueError(
            "H, K, and N must be nonzero, "
            f"got H={head_count}, K={input_features}, N={output_features}"
        )
    return token_count, head_count, input_features, output_features


def _launch_mla_kv_b_w8a16_bmm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    gptq_packed: bool,
) -> torch.Tensor:
    token_count, head_count, input_features, output_features = _validate_inputs(
        x,
        qweight,
        scales,
        gptq_packed=gptq_packed,
    )
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        raise RuntimeError("MLA W8A16 Triton requires an Ampere-or-newer GPU")

    output = torch.empty(
        (head_count, token_count, output_features),
        dtype=torch.bfloat16,
        device=x.device,
    )
    if token_count == 0:
        return output

    config = _select_launch_config(token_count)
    grid = (
        triton.cdiv(token_count, config.block_m),
        triton.cdiv(output_features, config.block_n),
        head_count,
    )
    _mla_kv_b_w8a16_bmm_kernel[grid](
        x,
        qweight,
        scales,
        output,
        token_count=token_count,
        output_features=output_features,
        input_features=input_features,
        stride_xm=x.stride(0),
        stride_xh=x.stride(1),
        stride_xk=x.stride(2),
        stride_wh=qweight.stride(0),
        stride_wk=qweight.stride(1),
        stride_wn=qweight.stride(2),
        stride_sh=scales.stride(0),
        stride_sn=scales.stride(2),
        stride_oh=output.stride(0),
        stride_om=output.stride(1),
        stride_on=output.stride(2),
        BLOCK_SIZE_M=config.block_m,
        BLOCK_SIZE_N=config.block_n,
        BLOCK_SIZE_K=config.block_k,
        GPTQ_PACKED=gptq_packed,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


def _fake_mla_kv_b_gptq_w8a16_bmm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    del scales
    return x.new_empty((qweight.shape[0], x.shape[0], qweight.shape[2]))


@register_custom_op(fake_impl=_fake_mla_kv_b_gptq_w8a16_bmm)
def mla_kv_b_gptq_w8a16_bmm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run MLA BMM directly on GPTQ row-packed W8 checkpoint weights.

    ``qweight`` must be INT32 ``[H, K / 4, N]`` with four biased uint8b128
    values packed little-endian along ``K``.  This is the production,
    backend-neutral layout: Marlin can repack it with
    ``gptq_marlin_moe_repack``, while Triton bit-extracts it in registers.
    """

    return _launch_mla_kv_b_w8a16_bmm(
        x,
        qweight,
        scales,
        gptq_packed=True,
    )


def mla_kv_b_w8a16_bmm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run the experimental raw-uint8 ``[H,K,N]`` form of MLA W8A16 BMM.

    Both forms support arbitrary positive ``M`` and the zero-token case on
    Ampere (SM80+) with FP32 accumulation.  Prefer
    :func:`mla_kv_b_gptq_w8a16_bmm` for persistent checkpoints.
    """

    return _launch_mla_kv_b_w8a16_bmm(
        x,
        qweight,
        scales,
        gptq_packed=False,
    )
