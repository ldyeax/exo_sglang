from types import SimpleNamespace

import torch
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_first_pipeline_stage_honors_supplied_input_embeddings(monkeypatch) -> None:
    model = object.__new__(deepseek_v4.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.embed_tokens = lambda _input_ids: (_ for _ in ()).throw(
        AssertionError("token embedding lookup must be skipped")
    )
    model.hc_mult = 2
    model.dspark_layers_to_capture = None
    model.use_fused_mhc_post_pre = False
    model.start_layer = 0
    model.end_layer = 0
    model.layers = torch.nn.ModuleList()
    model.hc_head = lambda hidden_states, *_args: hidden_states
    model.hc_head_fn = None
    model.hc_head_scale = 1.0
    model.hc_head_base = 1.0
    model.norm = lambda hidden_states: hidden_states
    model._can_run_tbo = lambda _forward_batch: False

    monkeypatch.setattr(
        deepseek_v4, "get_parallel", lambda: SimpleNamespace(attn_dp_size=1)
    )
    monkeypatch.setattr(deepseek_v4, "dsa_use_prefill_cp", lambda _batch: False)

    input_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    hidden_states, pre_hc_head = model.forward(
        input_ids=torch.tensor([9, 8, 7]),
        positions=torch.arange(3),
        forward_batch=SimpleNamespace(),
        input_embeds=input_embeds,
    )

    expected = input_embeds.unsqueeze(1).repeat(1, 2, 1)
    torch.testing.assert_close(hidden_states, expected)
    torch.testing.assert_close(pre_hc_head, expected.flatten(1))
