from types import SimpleNamespace

import torch
from sglang.srt.layers.attention import triton_backend
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner(*, is_draft_worker: bool):
    algorithm = SpeculativeAlgorithm.DSPARK
    server_args = SimpleNamespace(speculative_num_draft_tokens=8)
    runner = SimpleNamespace(
        is_draft_worker=is_draft_worker,
        server_args=server_args,
        spec_algorithm=algorithm,
    )
    runner.decode_num_tokens_per_req = lambda *, num_draft_tokens=None: (
        algorithm.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens or server_args.speculative_num_draft_tokens,
            is_draft_worker,
        )
    )
    return runner


def test_triton_uses_distinct_dspark_target_and_draft_verify_widths(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        triton_backend,
        "get_spec",
        lambda: SimpleNamespace(speculative_num_draft_tokens=8),
    )

    assert (
        triton_backend._resolve_verify_num_tokens_per_req(
            _runner(is_draft_worker=False)
        )
        == 8
    )
    assert (
        triton_backend._resolve_verify_num_tokens_per_req(_runner(is_draft_worker=True))
        == 7
    )


def test_triton_decode_only_runner_preserves_disabled_verify_width(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        triton_backend,
        "get_spec",
        lambda: SimpleNamespace(speculative_num_draft_tokens=None),
    )
    runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.NONE,
        decode_num_tokens_per_req=lambda **kwargs: 1,
    )

    assert triton_backend._resolve_verify_num_tokens_per_req(runner) is None


def test_dspark_dummy_verify_metadata_uses_runner_width() -> None:
    server_args = SimpleNamespace(
        speculative_num_draft_tokens=8,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
    )

    draft_info = create_dummy_verify_input(
        SpeculativeAlgorithm.DSPARK,
        server_args,
        torch.empty(0, dtype=torch.bool),
        num_tokens_per_req=7,
        is_draft_worker=True,
    )
    target_info = create_dummy_verify_input(
        SpeculativeAlgorithm.DSPARK,
        server_args,
        torch.empty(0, dtype=torch.bool),
        num_tokens_per_req=8,
        is_draft_worker=False,
    )

    assert draft_info is not None
    assert target_info is not None
    assert draft_info.draft_token_num == draft_info.num_tokens_per_req == 7
    assert target_info.draft_token_num == target_info.num_tokens_per_req == 8
