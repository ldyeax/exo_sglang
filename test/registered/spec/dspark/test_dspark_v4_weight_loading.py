from types import SimpleNamespace

import pytest
import torch
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark


class _RecordingParameter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.weight_loader = self._load

    def _load(self, _param, _weight, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _WeightLoaderHarness:
    load_weights = DeepseekV4ForCausalLMDSpark.load_weights
    _required_shared_expert_source_names = (
        DeepseekV4ForCausalLMDSpark._required_shared_expert_source_names
    )
    _assert_confidence_head_loaded = (
        DeepseekV4ForCausalLMDSpark._assert_confidence_head_loaded
    )
    _remap_dspark_weight_name = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name

    def __init__(self, *, with_destinations: bool = True) -> None:
        self.config = SimpleNamespace(n_routed_experts=256, n_shared_experts=1)
        self.num_stages = 3
        self.confidence_head = None
        self.params: dict[str, _RecordingParameter] = {}
        if with_destinations:
            for stage_id in range(self.num_stages):
                prefix = f"stages.{stage_id}.mlp.shared_experts"
                for projection in ("gate_up_proj", "down_proj"):
                    for suffix in ("weight", "weight_scale_inv"):
                        self.params[f"{prefix}.{projection}.{suffix}"] = (
                            _RecordingParameter()
                        )

    def named_parameters(self):
        return self.params.items()


def _shared_expert_weights(*, omit: str | None = None):
    for stage_id in range(3):
        for projection in (1, 2, 3):
            for suffix in ("weight", "scale"):
                name = f"mtp.{stage_id}.ffn.shared_experts.w{projection}.{suffix}"
                if name != omit:
                    yield name, torch.ones(1)


def test_dspark_v4_requires_every_shared_expert_source_tensor():
    harness = _WeightLoaderHarness()
    missing = "mtp.1.ffn.shared_experts.w3.scale"

    with pytest.raises(ValueError, match=r"missing required shared-expert") as exc:
        harness.load_weights(_shared_expert_weights(omit=missing))

    assert missing in str(exc.value)


def test_dspark_v4_shared_expert_coverage_accepts_all_18_sources(caplog):
    harness = _WeightLoaderHarness()

    caplog.set_level("INFO")
    harness.load_weights(_shared_expert_weights())

    assert sum(len(param.calls) for param in harness.params.values()) == 18
    for stage_id in range(3):
        prefix = f"stages.{stage_id}.mlp.shared_experts"
        for suffix in ("weight", "weight_scale_inv"):
            gate_up_calls = harness.params[f"{prefix}.gate_up_proj.{suffix}"].calls
            assert [args for args, _kwargs in gate_up_calls] == [(0,), (1,)]
            down_calls = harness.params[f"{prefix}.down_proj.{suffix}"].calls
            assert [args for args, _kwargs in down_calls] == [()]
    assert "shared-expert checkpoint coverage verified: 18/18" in caplog.text


def test_dspark_v4_mapped_destination_miss_is_fatal():
    harness = _WeightLoaderHarness(with_destinations=False)

    with pytest.raises(ValueError, match=r"no model destination") as exc:
        harness.load_weights([("mtp.0.ffn.shared_experts.w1.weight", torch.ones(1))])

    assert "stages.0.mlp.shared_experts.gate_proj.weight" in str(exc.value)
