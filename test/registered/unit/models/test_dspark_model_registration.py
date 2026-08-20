"""Registration coverage for released DSpark checkpoint architecture names."""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_released_dspark_draft_architecture_resolves_to_native_model() -> None:
    from sglang.srt.models.dspark import DSparkDraftModel
    from sglang.srt.models.registry import ModelRegistry

    model_cls, resolved_arch = ModelRegistry.resolve_model_cls("DSparkDraftModel")

    assert resolved_arch == "DSparkDraftModel"
    assert model_cls is DSparkDraftModel


def test_released_qwen38_dspark_config_contract_is_native() -> None:
    from sglang.srt.speculative.dspark_components.dspark_config import (
        parse_dspark_draft_config,
        resolve_runtime_config,
    )

    config = SimpleNamespace(
        block_size=7,
        dflash_config={
            "mask_token_id": 248077,
            "target_layer_ids": [4, 16, 28, 40, 52],
        },
        hidden_size=5120,
        markov_head_type="vanilla",
        markov_rank=256,
        num_hidden_layers=5,
        num_target_layers=64,
        vocab_size=248320,
    )

    draft_config = parse_dspark_draft_config(draft_hf_config=config)
    runtime_config = resolve_runtime_config(
        draft_hf_config=config,
        speculative_num_draft_tokens=8,
        target_vocab_size=248320,
    )

    assert draft_config.num_hidden_layers == 5
    assert draft_config.num_target_layers == 64
    assert draft_config.target_layer_ids == [4, 16, 28, 40, 52]
    assert draft_config.markov_rank == 256
    assert draft_config.markov_head_type == "vanilla"
    assert runtime_config.gamma == 7
    assert runtime_config.verify_num_draft_tokens == 8
    assert runtime_config.mask_token_id == 248077
