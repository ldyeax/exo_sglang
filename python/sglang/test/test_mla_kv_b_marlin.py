from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.mla_kv_b_w8 import GPTQMLAKVW8Method


def _format_config() -> dict[str, object]:
    return {
        "bits": 8,
        "block_size_m": 8,
        "format": "gptq_packed_rows_per_head_v1",
        "implicit_bias": 128,
        "kv_lora_rank": 512,
        "num_attention_heads": 64,
        "pack_axis": "K",
        "pack_order": "little_endian_k_lanes_0_1_2_3",
        "qk_nope_head_dim": 192,
        "scale_compute_dtype": "float32",
        "scale_dtype": "bfloat16",
        "v_head_dim": 256,
    }


def _method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    local_heads: int,
) -> GPTQMLAKVW8Method:
    monkeypatch.setenv("SGLANG_MLA_KV_B_W8_BACKEND", backend)
    config = SimpleNamespace(
        weight_bits=8,
        group_size=-1,
        desc_act=False,
        is_sym=True,
        pack_factor=4,
    )
    method = GPTQMLAKVW8Method(config, _format_config())
    method.num_local_heads = local_heads
    return method


def _quantize_gptq_rows(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize logical ``[H, K, N]`` weights like the offline converter."""

    maximum = weight.float().amax(dim=1, keepdim=True)
    minimum = weight.float().amin(dim=1, keepdim=True)
    scales = torch.maximum(maximum.abs() / 127.0, minimum.abs() / 128.0)
    scales = scales.clamp_min(torch.finfo(torch.bfloat16).tiny).to(torch.bfloat16)
    signed = torch.round(weight.float() / scales.float()).clamp(-128, 127)
    raw_qweight = (signed.to(torch.int16) + 128).to(torch.uint8)
    reference = ((raw_qweight.to(torch.int16) - 128).float() * scales.float()).to(
        torch.bfloat16
    )

    heads, input_features, output_features = raw_qweight.shape
    packed = torch.zeros(
        heads,
        input_features // 4,
        output_features,
        dtype=torch.int64,
        device=weight.device,
    )
    for byte_index in range(4):
        packed |= raw_qweight[:, byte_index::4, :].to(torch.int64) << (8 * byte_index)
    return packed.to(torch.int32), scales, reference


def _canonical_layer(
    *,
    local_heads: int,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    layer = torch.nn.Module()
    references = []
    for stem, input_features, output_features in (
        ("kc", 192, 512),
        ("vc", 512, 256),
    ):
        source = torch.randn(
            local_heads,
            input_features,
            output_features,
            dtype=torch.bfloat16,
            device="cuda",
        )
        qweight, scales, reference = _quantize_gptq_rows(source)
        layer.register_parameter(
            f"{stem}_qweight",
            torch.nn.Parameter(qweight, requires_grad=False),
        )
        layer.register_parameter(
            f"{stem}_scales",
            torch.nn.Parameter(scales, requires_grad=False),
        )
        references.append(reference)
    return layer, references[0], references[1]


@pytest.mark.parametrize(
    ("tokens", "stem", "input_features", "output_features"),
    [
        (0, "kc", 192, 512),
        (1, "kc", 192, 512),
        (3, "vc", 512, 256),
        (128, "kc", 192, 512),
    ],
)
def test_canonical_checkpoint_repack_matches_quantized_bmm(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    stem: str,
    input_features: int,
    output_features: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("grouped W8 Marlin requires Ampere or newer")

    torch.manual_seed(7)
    local_heads = 4
    layer, kc_reference, vc_reference = _canonical_layer(local_heads=local_heads)
    reference_weight = kc_reference if stem == "kc" else vc_reference
    canonical_size_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in layer.parameters()
    )
    method = _method(
        monkeypatch,
        backend="marlin",
        local_heads=local_heads,
    )

    method.process_weights_after_loading(layer)

    # Repacking changes only the compact INT32/scale layouts. It must not
    # materialize the logical BF16 KC or VC matrices.
    assert not hasattr(layer, "w_kc")
    assert not hasattr(layer, "w_vc")
    repacked_size_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in layer.parameters()
    )
    assert repacked_size_bytes == canonical_size_bytes

    x = torch.randn(
        tokens,
        local_heads,
        input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = torch.bmm(x.transpose(0, 1), reference_weight)
    if stem == "kc":
        actual = method.apply_mla_k(layer, x)
    else:
        actual = method.apply_mla_v_bmm(layer, x)

    assert actual.shape == (local_heads, tokens, output_features)
    error = actual.float() - expected.float()
    if tokens:
        relative_rms_error = (
            error.square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-12)
        )
        cosine_similarity = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        )
        assert error.abs().max().item() <= 0.6
        assert relative_rms_error.item() <= 0.004
        assert cosine_similarity.item() >= 0.99999


def test_rejects_ordinary_linear_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    method = _method(monkeypatch, backend="triton", local_heads=32)

    with pytest.raises(RuntimeError, match="must use absorbed MLA"):
        method.apply(
            torch.nn.Module(),
            torch.empty(1, 512, dtype=torch.bfloat16),
        )


def test_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_MLA_KV_B_W8_BACKEND", "expanded-bf16")
    config = SimpleNamespace(
        weight_bits=8,
        group_size=-1,
        desc_act=False,
        is_sym=True,
        pack_factor=4,
    )

    with pytest.raises(ValueError, match="SGLANG_MLA_KV_B_W8_BACKEND"):
        GPTQMLAKVW8Method(config, _format_config())


def test_tp2_parameters_shard_complete_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_heads = 32
    method = _method(
        monkeypatch,
        backend="triton",
        local_heads=local_heads,
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=512,
        output_partition_sizes=[local_heads * (192 + 256)],
        input_size=512,
        output_size=64 * (192 + 256),
        params_dtype=torch.bfloat16,
        weight_loader=lambda _parameter, _loaded_weight: None,
    )

    for name, input_features, output_features, dtype in (
        ("kc_qweight", 48, 512, torch.int32),
        ("kc_scales", 1, 512, torch.bfloat16),
        ("vc_qweight", 128, 256, torch.int32),
        ("vc_scales", 1, 256, torch.bfloat16),
    ):
        global_weight = (
            torch.arange(64, dtype=torch.int32)
            .view(64, 1, 1)
            .expand(64, input_features, output_features)
            .to(dtype)
        )
        parameter = getattr(layer, name)
        parameter.load_column_parallel_weight(global_weight, tp_rank=1)
        expected = global_weight[local_heads:]
        torch.testing.assert_close(parameter, expected)

    assert layer.kc_qweight.shape == (32, 48, 512)
    assert layer.kc_scales.shape == (32, 1, 512)
    assert layer.vc_qweight.shape == (32, 128, 256)
    assert layer.vc_scales.shape == (32, 1, 256)
