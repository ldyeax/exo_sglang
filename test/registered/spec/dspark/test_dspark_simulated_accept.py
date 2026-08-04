import torch
from sglang.srt.speculative.dspark_components.dspark_verify import (
    greedy_bonus_at_accept_len,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_simulated_greedy_accept_recomputes_bonus_at_selected_row() -> None:
    target_logits = torch.tensor(
        [
            [[0.0, 4.0, 1.0], [5.0, 0.0, 1.0], [0.0, 1.0, 6.0]],
            [[7.0, 1.0, 0.0], [0.0, 8.0, 1.0], [0.0, 1.0, 9.0]],
        ]
    ).view(6, 3)
    correct_len = torch.tensor([0, 2], dtype=torch.int32)

    bonus = greedy_bonus_at_accept_len(
        target_logits=target_logits,
        correct_len=correct_len,
        verify_num_draft_tokens=3,
    )

    assert torch.equal(bonus, torch.tensor([1, 2]))
