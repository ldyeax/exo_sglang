from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.environ import envs
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_kv_cache_configurator(kv_cache_dtype: torch.dtype, dtype_name: str):
    model_config = SimpleNamespace(
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        index_head_dim=128,
        context_len=524288,
        compress_ratios=[4, 128],
        window_size=128,
    )
    server_args = SimpleNamespace(
        swa_full_tokens_ratio=1024 / 524288,
        speculative_algorithm=None,
        max_speculative_num_draft_tokens=None,
        max_running_requests=1,
        disaggregation_mode="null",
        disaggregation_decode_extra_slots=0,
        enable_hisparse=False,
        enable_deepseek_v4_fp4_indexer=False,
    )
    return SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_dtype_str=dtype_name,
        device="cuda",
        gpu_id=0,
        page_size=256,
        model_config=model_config,
        layer_info=SimpleNamespace(start_layer=0, end_layer=2),
        ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
        server_args=server_args,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )


def test_sm86_plans_for_the_requested_packed_fp8_layout():
    fp8_kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    bf16_kvc = _make_kv_cache_configurator(torch.bfloat16, "bfloat16")
    with patch(
        "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
        return_value=(8, 6),
    ):
        fp8_configurator = DSV4PoolConfigurator(fp8_kvc)
        bf16_configurator = DSV4PoolConfigurator(bf16_kvc)

    assert not fp8_configurator.use_bf16_cache
    assert bf16_configurator.use_bf16_cache
    assert fp8_configurator.bytes_per_full_token < (
        bf16_configurator.bytes_per_full_token
    )


def test_sm80_and_sm87_still_plan_for_bfloat16_storage():
    for device_capability in ((8, 0), (8, 7)):
        kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
        with patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=device_capability,
        ):
            configurator = DSV4PoolConfigurator(kvc)

        assert configurator.use_bf16_cache


def test_sm120_preserves_requested_fp8_planning_and_bfloat16_override():
    fp8_kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    bf16_kvc = _make_kv_cache_configurator(torch.bfloat16, "bfloat16")
    with patch(
        "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
        return_value=(12, 0),
    ):
        fp8_configurator = DSV4PoolConfigurator(fp8_kvc)
        bf16_configurator = DSV4PoolConfigurator(bf16_kvc)

    assert not fp8_configurator.use_bf16_cache
    assert bf16_configurator.use_bf16_cache
    assert fp8_configurator.bytes_per_full_token < (
        bf16_configurator.bytes_per_full_token
    )


@pytest.mark.parametrize(
    ("latent_int4", "indexer_int4"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_sm86_plans_all_four_independent_int4_storage_modes(
    latent_int4: bool,
    indexer_int4: bool,
):
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with (
        envs.SGLANG_DSV4_INT4_KV_STORAGE.override(latent_int4),
        envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(indexer_int4),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
    ):
        configurator = DSV4PoolConfigurator(kvc)

    assert configurator.use_int4_storage is latent_int4
    assert configurator.use_int4_indexer_storage is indexer_int4


def test_sm86_int4_memory_savings_are_independent_and_additive():
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    planned_bytes = {}
    with patch(
        "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
        return_value=(8, 6),
    ):
        for latent_int4, indexer_int4 in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with (
                envs.SGLANG_DSV4_INT4_KV_STORAGE.override(latent_int4),
                envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(indexer_int4),
            ):
                planned_bytes[(latent_int4, indexer_int4)] = DSV4PoolConfigurator(
                    kvc
                ).bytes_per_full_token

    fp8 = planned_bytes[(False, False)]
    latent = planned_bytes[(True, False)]
    indexer = planned_bytes[(False, True)]
    both = planned_bytes[(True, True)]
    assert fp8 > latent > both
    assert fp8 > indexer > both
    # One C4 layer contributes one indexer token per four full tokens:
    # (132 FP8 bytes - 72 INT4 bytes) / 4 = 15 bytes/full-token.
    assert fp8 - indexer == pytest.approx(15.0)
    assert latent - both == pytest.approx(15.0)
    assert fp8 - latent == pytest.approx(indexer - both)


def test_sm86_oscar_int2_plans_bf16_recent_and_int2_history():
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with patch(
        "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
        return_value=(8, 6),
    ):
        baseline = DSV4PoolConfigurator(kvc)
        with envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.override(True):
            oscar = DSV4PoolConfigurator(kvc)

    assert oscar.use_oscar_int2_storage
    assert oscar.bytes_per_full_token < baseline.bytes_per_full_token
    # Two local layers have protected BF16 SWA; C4/C128 history is the
    # naturally aligned 272-byte OSCAR row.  The C4 scorer is one asymmetric
    # G128 group: 32 bytes of uint2 codes plus one FP32 scale/zero pair (8 B).
    expected_delta = (1_024 - 585) * 2 / 512
    expected_delta += (272 - 585) / 4
    expected_delta += (272 - 864) / 128
    expected_delta += (40 - 132) / 4
    assert oscar.bytes_per_full_token - baseline.bytes_per_full_token == pytest.approx(
        expected_delta
    )


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (None, None)])
def test_oscar_int2_planner_fails_closed_outside_exact_sm86(capability):
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with (
        envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.override(True),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=capability,
        ),
        pytest.raises(ValueError, match="only for exact SM86"),
    ):
        DSV4PoolConfigurator(kvc)


@pytest.mark.parametrize("conflict", ["latent", "indexer", "c128"])
def test_oscar_int2_rejects_non_oscar_cache_layouts(conflict):
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with (
        envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.override(True),
        envs.SGLANG_DSV4_INT4_KV_STORAGE.override(conflict == "latent"),
        envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(conflict == "indexer"),
        envs.SGLANG_DSV4_SM86_C128_BF16_STORAGE.override(conflict == "c128"),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
        pytest.raises(ValueError, match="incompatible"),
    ):
        DSV4PoolConfigurator(kvc)


def test_sm86_selective_c128_bf16_plans_only_the_c128_storage_delta():
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with patch(
        "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
        return_value=(8, 6),
    ):
        baseline = DSV4PoolConfigurator(kvc)
        with envs.SGLANG_DSV4_SM86_C128_BF16_STORAGE.override(True):
            selective = DSV4PoolConfigurator(kvc)

    assert not baseline.use_selective_c128_bf16_storage
    assert selective.use_selective_c128_bf16_storage
    # page_size=256 gives two-token C128 pages.  Packed FP8 allocates
    # ceil(2*584/576)*576 = 1728 bytes/page (864/token), while unpadded BF16
    # allocates 2*1024 = 2048 bytes/page (1024/token).  One C128 layer thus
    # adds (1024-864)/128 = 1.25 bytes per full-context token.
    assert (
        selective.bytes_per_full_token - baseline.bytes_per_full_token
        == pytest.approx(1.25)
    )
    assert (
        selective._get_bytes_per_full_token() - baseline._get_bytes_per_full_token()
        == pytest.approx(1.25)
    )


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (None, None)])
def test_selective_c128_bf16_planner_fails_closed_outside_exact_sm86(capability):
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with (
        envs.SGLANG_DSV4_SM86_C128_BF16_STORAGE.override(True),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=capability,
        ),
        pytest.raises(ValueError, match="only for exact SM86"),
    ):
        DSV4PoolConfigurator(kvc)


def test_selective_c128_bf16_planner_rejects_nonselective_storage_modes():
    bf16_kvc = _make_kv_cache_configurator(torch.bfloat16, "bfloat16")
    fp8_kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with (
        envs.SGLANG_DSV4_SM86_C128_BF16_STORAGE.override(True),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
    ):
        with pytest.raises(ValueError, match="requires fp8_e4m3"):
            DSV4PoolConfigurator(bf16_kvc)
        with (
            envs.SGLANG_DSV4_INT4_KV_STORAGE.override(True),
            pytest.raises(ValueError, match="incompatible with.*INT4_KV_STORAGE"),
        ):
            DSV4PoolConfigurator(fp8_kvc)


@pytest.mark.parametrize(
    ("flag", "capability"),
    [
        ("latent", (8, 0)),
        ("indexer", (8, 9)),
        ("latent", (9, 0)),
        ("indexer", (None, None)),
    ],
)
def test_int4_planner_fails_closed_outside_exact_sm86(flag, capability):
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    with (
        envs.SGLANG_DSV4_INT4_KV_STORAGE.override(flag == "latent"),
        envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(flag == "indexer"),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=capability,
        ),
        pytest.raises(ValueError, match="only for exact SM86"),
    ):
        DSV4PoolConfigurator(kvc)


@pytest.mark.parametrize("flag", ["latent", "indexer"])
def test_int4_planner_rejects_bfloat16_as_storage_carrier(flag):
    kvc = _make_kv_cache_configurator(torch.bfloat16, "bfloat16")
    with (
        envs.SGLANG_DSV4_INT4_KV_STORAGE.override(flag == "latent"),
        envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(flag == "indexer"),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
        pytest.raises(ValueError, match="requires fp8_e4m3"),
    ):
        DSV4PoolConfigurator(kvc)


def test_latent_int4_planner_rejects_hisparse():
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    kvc.server_args.enable_hisparse = True
    with (
        envs.SGLANG_DSV4_INT4_KV_STORAGE.override(True),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
        pytest.raises(ValueError, match="incompatible with HiSparse"),
    ):
        DSV4PoolConfigurator(kvc)


def test_indexer_int4_planner_rejects_fp4_indexer():
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    kvc.server_args.enable_deepseek_v4_fp4_indexer = True
    with (
        envs.SGLANG_DSV4_INT4_C4_INDEXER_STORAGE.override(True),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
        pytest.raises(ValueError, match="incompatible with.*FP4 indexer"),
    ):
        DSV4PoolConfigurator(kvc)


def test_oscar_int2_planner_rejects_fp4_indexer_before_pool_allocation():
    kvc = _make_kv_cache_configurator(torch.float8_e4m3fn, "fp8_e4m3")
    kvc.server_args.enable_deepseek_v4_fp4_indexer = True
    with (
        envs.SGLANG_DSV4_OSCAR_INT2_KV_STORAGE.override(True),
        patch(
            "sglang.srt.mem_cache.dsv4_kv_cache_dtype.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
        patch.object(
            DSV4PoolConfigurator,
            "_get_bytes_per_full_token",
            side_effect=AssertionError("OSCAR+FP4 reached pool sizing"),
        ),
        pytest.raises(ValueError, match="incompatible with.*FP4 indexer"),
    ):
        DSV4PoolConfigurator(kvc)
