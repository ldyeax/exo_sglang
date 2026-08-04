from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _require_dsv4_oscar_wo_a_output_restore_absorbed,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    _Dsv4OscarWoAOutputRotationBinding,
    _fold_dsv4_oscar_output_rotation_into_wo_a_weight_,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _project_per_group(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    groups, _output_rank, heads, head_dim = weight.shape
    assert values.shape[1:] == (groups, heads, head_dim)
    return torch.einsum("tghd,grhd->tgr", values, weight)


def test_fold_orientation_is_w_nope_times_r_and_rope_is_unchanged() -> None:
    generator = torch.Generator().manual_seed(7)
    nope_dim, rope_dim = 4, 2
    groups, output_rank, heads = 2, 3, 2
    head_dim = nope_dim + rope_dim
    rotation, _ = torch.linalg.qr(
        torch.randn((nope_dim, nope_dim), generator=generator, dtype=torch.float64)
    )
    rotation = rotation.contiguous()
    original_weight = torch.randn(
        (groups * output_rank, heads * head_dim),
        generator=generator,
        dtype=torch.float64,
    )
    rotated_attention_output = torch.randn(
        (5, groups, heads, head_dim), generator=generator, dtype=torch.float64
    )

    restored = rotated_attention_output.clone()
    restored[..., :nope_dim] = restored[..., :nope_dim] @ rotation.T
    expected = _project_per_group(
        restored,
        original_weight.view(groups, output_rank, heads, head_dim),
    )

    folded_weight = original_weight.clone()
    rope_before = folded_weight.view(groups, output_rank, heads, head_dim)[
        ..., nope_dim:
    ].clone()
    _fold_dsv4_oscar_output_rotation_into_wo_a_weight_(
        folded_weight,
        rotation,
        num_local_groups=groups,
        output_rank=output_rank,
        heads_per_group=heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    actual = _project_per_group(
        rotated_attention_output,
        folded_weight.view(groups, output_rank, heads, head_dim),
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    assert torch.equal(
        folded_weight.view(groups, output_rank, heads, head_dim)[..., nope_dim:],
        rope_before,
    )
    assert not torch.equal(folded_weight, original_weight)


def test_bf16_fold_matches_two_gemm_restore_with_expected_roundoff() -> None:
    generator = torch.Generator().manual_seed(11)
    nope_dim, rope_dim = 64, 8
    groups, output_rank, heads = 2, 16, 3
    head_dim = nope_dim + rope_dim
    rotation, _ = torch.linalg.qr(
        torch.randn((nope_dim, nope_dim), generator=generator, dtype=torch.float64)
    )
    rotation = rotation.to(torch.bfloat16).contiguous()
    original_weight = (
        torch.randn(
            (groups * output_rank, heads * head_dim), generator=generator
        )
        * 0.02
    ).to(torch.bfloat16)
    rotated_attention_output = torch.randn(
        (8, groups, heads, head_dim), generator=generator
    ).to(torch.bfloat16)

    restored = rotated_attention_output.clone()
    restored[..., :nope_dim] = restored[..., :nope_dim] @ rotation.T
    expected = _project_per_group(
        restored,
        original_weight.view(groups, output_rank, heads, head_dim),
    ).float()
    folded_weight = original_weight.clone()
    _fold_dsv4_oscar_output_rotation_into_wo_a_weight_(
        folded_weight,
        rotation,
        num_local_groups=groups,
        output_rank=output_rank,
        heads_per_group=heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    actual = _project_per_group(
        rotated_attention_output,
        folded_weight.view(groups, output_rank, heads, head_dim),
    ).float()

    relative_l2 = torch.linalg.vector_norm(
        actual - expected
    ) / torch.linalg.vector_norm(expected)
    assert relative_l2.item() < 0.007


def test_caller_owned_fold_workspaces_prevent_internal_allocation() -> None:
    weight = torch.arange(2 * 2 * 6, dtype=torch.float64).view(2, 12)
    rotation = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    input_workspace = torch.empty((4, 4), dtype=torch.float64)
    output_workspace = torch.empty_like(input_workspace)

    with (
        patch.object(deepseek_v4.torch, "empty", side_effect=AssertionError),
        patch.object(deepseek_v4.torch, "empty_like", side_effect=AssertionError),
    ):
        _fold_dsv4_oscar_output_rotation_into_wo_a_weight_(
            weight,
            rotation,
            num_local_groups=1,
            output_rank=2,
            heads_per_group=2,
            nope_dim=4,
            rope_dim=2,
            input_workspace=input_workspace,
            output_workspace=output_workspace,
        )


def _bound_layer_pool_and_calibration():
    weight = torch.nn.Parameter(torch.ones((2, 8), dtype=torch.bfloat16))
    rotation = torch.eye(4, dtype=torch.bfloat16)
    calibration = SimpleNamespace(layer_id=2, rotation=rotation)
    binding = _Dsv4OscarWoAOutputRotationBinding(
        layer_id=2,
        artifact_sha256="a" * 64,
        admission_sha256="b" * 64,
        rotation=rotation,
        weight=weight,
        weight_version=weight._version,
        consumer_role="target_compressed",
    )
    layer = SimpleNamespace(_dsv4_oscar_wo_a_output_rotation_binding=binding)
    pool = SimpleNamespace(
        oscar_consumer_role="target_compressed",
        oscar_artifact_sha256="a" * 64,
        oscar_admission_sha256="b" * 64,
    )
    return layer, pool, calibration, binding


def test_backend_skip_proof_is_allocation_free_and_does_not_mutate_weight() -> None:
    layer, pool, calibration, binding = _bound_layer_pool_and_calibration()
    pointer = binding.weight.data_ptr()
    version = binding.weight._version
    with patch.object(torch, "empty", side_effect=AssertionError):
        for _ in range(3):
            _require_dsv4_oscar_wo_a_output_restore_absorbed(
                layer_id=2,
                layer=layer,
                token_to_kv_pool=pool,
                calibration=calibration,
            )
    assert binding.weight.data_ptr() == pointer
    assert binding.weight._version == version


@pytest.mark.parametrize("failure", ["missing", "rotation", "artifact", "weight"])
def test_backend_skip_proof_fails_closed_on_stale_binding(failure: str) -> None:
    layer, pool, calibration, binding = _bound_layer_pool_and_calibration()
    if failure == "missing":
        del layer._dsv4_oscar_wo_a_output_rotation_binding
    elif failure == "rotation":
        calibration.rotation = calibration.rotation.clone()
    elif failure == "artifact":
        pool.oscar_artifact_sha256 = "c" * 64
    else:
        with torch.no_grad():
            binding.weight.add_(1)

    with pytest.raises(RuntimeError):
        _require_dsv4_oscar_wo_a_output_restore_absorbed(
            layer_id=2,
            layer=layer,
            token_to_kv_pool=pool,
            calibration=calibration,
        )


def test_model_absorption_is_idempotent_and_telemetry_is_exact() -> None:
    model = object.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    layer, pool, calibration, binding = _bound_layer_pool_and_calibration()
    attention = SimpleNamespace(
        wo_a=SimpleNamespace(weight=binding.weight), attn_mqa=layer
    )
    model.model = SimpleNamespace(layers={2: SimpleNamespace(self_attn=attention)})
    pool.use_oscar_int2_storage = True
    pool.get_oscar_calibration = lambda layer_id: calibration
    model._dsv4_oscar_wo_a_pool = pool
    model._dsv4_oscar_wo_a_bindings = {2: binding}
    model._dsv4_oscar_wo_a_expected_layer_ids = (2,)
    model._dsv4_oscar_wo_a_apply_count = 1
    version = binding.weight._version

    first = model.absorb_dsv4_oscar_output_rotation_into_wo_a(pool)
    second = model.absorb_dsv4_oscar_output_rotation_into_wo_a(pool)

    assert first == second
    assert second["apply_count"] == 1
    assert second["runtime_restore_skipped_layer_ids"] == [2]
    assert second["all_local_target_compressed_layers_skip_runtime_restore"] is True
    assert second["fold_orientation"] == "wo_a_nope@rotation"
    assert binding.weight._version == version


def test_model_runner_getter_revalidates_and_returns_a_copy(monkeypatch) -> None:
    calls: list[object] = []
    state = {
        "enabled": True,
        "consumer_role": "target_compressed",
        "applied": True,
        "apply_count": 1,
        "all_local_target_compressed_layers_absorbed": True,
        "all_local_target_compressed_layers_skip_runtime_restore": True,
        "runtime_restore_skipped_layer_ids": [2],
    }
    pool = SimpleNamespace(
        use_oscar_int2_storage=True,
        oscar_consumer_role="target_compressed",
    )
    runner = object.__new__(ModelRunner)
    runner.token_to_kv_pool = pool
    runner.is_draft_worker = False
    runner.model = SimpleNamespace(
        absorb_dsv4_oscar_output_rotation_into_wo_a=lambda received_pool: (
            calls.append(received_pool) or dict(state)
        )
    )
    runner._dsv4_oscar_wo_a_absorption_state = None
    monkeypatch.setattr(
        model_runner_module, "get_lora", lambda: SimpleNamespace(enable_lora=False)
    )

    runner._ensure_dsv4_oscar_wo_a_output_rotation_absorbed()
    exported = runner.get_dsv4_oscar_wo_a_absorption_state()
    exported["runtime_restore_skipped_layer_ids"].append(99)

    assert calls == [pool, pool]
    assert pool.oscar_wo_a_absorption_state == state
    assert runner._dsv4_oscar_wo_a_absorption_state == state


def test_post_pool_hook_precedes_every_other_post_pool_component(monkeypatch) -> None:
    events: list[str] = []
    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace()
    runner._token_oracle_manager = object()
    runner.token_to_kv_pool = object()
    runner._ensure_dsv4_oscar_wo_a_output_rotation_absorbed = lambda: events.append(
        "absorb"
    )
    runner.init_ngram_embedding_manager = lambda: events.append("ngram")
    runner.maybe_init_hisparse_coordinator = lambda: events.append("hisparse")
    runner.init_routed_experts_capturer = lambda: events.append("experts")
    runner.init_indexer_capturer = lambda: events.append("indexer")
    monkeypatch.setattr(
        model_runner_module,
        "install_canary",
        lambda **_kwargs: events.append("canary") or object(),
    )

    ModelRunner._init_post_memory_pool_components(runner)

    assert events == ["absorb", "canary", "ngram", "hisparse", "experts", "indexer"]


def test_every_graph_capture_entry_revalidates_before_warmup(monkeypatch) -> None:
    events: list[str] = []
    runner = object.__new__(ModelRunner)
    runner.eager_runner = object()
    runner._ensure_dsv4_oscar_wo_a_output_rotation_absorbed = lambda: events.append(
        "validate"
    )

    monkeypatch.setattr(
        model_runner_module,
        "capture_cuda_graphs",
        lambda **_kwargs: events.append("combined-capture")
        or SimpleNamespace(
            eager_runner=object(),
            prefill_runner=object(),
            decode=SimpleNamespace(runner=object(), graph_mem_usage=1),
        ),
    )
    ModelRunner.init_cuda_graphs(runner)
    assert events == ["validate", "combined-capture"]

    events.clear()
    monkeypatch.setattr(
        model_runner_module,
        "capture_decode_graph",
        lambda **_kwargs: events.append("decode-warmup-capture")
        or SimpleNamespace(runner=object(), graph_mem_usage=1),
    )
    ModelRunner.init_decode_cuda_graph(runner)
    assert events == ["validate", "decode-warmup-capture"]

    events.clear()
    monkeypatch.setattr(
        model_runner_module,
        "capture_prefill_graph",
        lambda **_kwargs: events.append("prefill-warmup-capture") or object(),
    )
    ModelRunner.init_prefill_cuda_graph(runner)
    assert events == ["validate", "prefill-warmup-capture"]


def _require_sm86() -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA is required")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("OSCAR wo_a absorption is fail-closed to SM86")


def test_sm86_production_absorption_replays_without_weight_mutation() -> None:
    _require_sm86()
    layer_id = 2
    generator = torch.Generator(device="cuda").manual_seed(23)
    permutation = torch.arange(447, -1, -1, device="cuda")
    rotation = torch.zeros((448, 448), dtype=torch.bfloat16, device="cuda")
    rotation[torch.arange(448, device="cuda"), permutation] = 1
    calibration = SimpleNamespace(layer_id=layer_id, rotation=rotation)
    weight = torch.nn.Parameter(
        torch.randn((4, 2 * 512), generator=generator, device="cuda").to(
            torch.bfloat16
        ),
        requires_grad=False,
    )
    rope_before = weight.view(1, 4, 2, 512)[..., 448:].clone()
    attention = SimpleNamespace(
        compress_ratio=4,
        wo_a=SimpleNamespace(
            weight=weight,
            quant_method=UnquantizedLinearMethod(),
        ),
        attn_mqa=SimpleNamespace(),
        n_heads=2,
        n_groups=1,
        n_local_groups=1,
        o_lora_rank=4,
    )
    model = object.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(
        start_layer=layer_id,
        end_layer=layer_id + 1,
        layers={layer_id: SimpleNamespace(self_attn=attention)},
    )
    model._dsv4_oscar_wo_a_pool = None
    model._dsv4_oscar_wo_a_bindings = {}
    model._dsv4_oscar_wo_a_expected_layer_ids = ()
    model._dsv4_oscar_wo_a_apply_count = 0
    pool = SimpleNamespace(
        use_oscar_int2_storage=True,
        is_draft_worker=False,
        oscar_consumer_role="target_compressed",
        oscar_artifact_sha256="a" * 64,
        oscar_admission_sha256="b" * 64,
        compression_ratios=[0, 0, 4],
        get_oscar_calibration=lambda requested: calibration,
    )

    # ServerArgs disables the default FP8 wo_a gate on SM86 before model load;
    # this focused constructor bypasses that normal startup phase.
    with patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False):
        first = model.absorb_dsv4_oscar_output_rotation_into_wo_a(pool)
        folded = weight.detach().clone()
        second = model.absorb_dsv4_oscar_output_rotation_into_wo_a(pool)
    assert first == second
    assert torch.equal(weight, folded)
    assert torch.equal(weight.view(1, 4, 2, 512)[..., 448:], rope_before)

    values = torch.randn((3, weight.shape[1]), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    output = torch.empty((3, weight.shape[0]), dtype=torch.bfloat16, device="cuda")
    torch.mm(values, weight.T, out=output)
    graph = torch.cuda.CUDAGraph()
    version = weight._version
    with torch.cuda.graph(graph):
        torch.mm(values, weight.T, out=output)
    values.add_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, values @ weight.T, rtol=0, atol=0)
    assert weight._version == version
