from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.expert_distribution import _StatAccumulator


def _make_accumulator(*, pp_size: int, global_rank: int) -> _StatAccumulator:
    accumulator = object.__new__(_StatAccumulator)
    accumulator._server_args = SimpleNamespace(pp_size=pp_size)
    accumulator._expert_location_metadata = SimpleNamespace(
        num_layers=61,
        num_logical_experts=384,
        physical_to_logical_map=Mock(),
    )
    accumulator._global_physical_count_of_buffered_step = Mock()
    accumulator._global_physical_count_of_buffered_step.get_all.return_value = (
        torch.empty(0)
    )
    accumulator._first_dump = False
    accumulator._rank = 0
    accumulator._global_rank = global_rank
    accumulator._get_global_average_utilization_rate = Mock(return_value=None)
    return accumulator


def test_stat_dump_writes_rank_local_profile_without_collective_under_pp():
    accumulator = _make_accumulator(pp_size=3, global_rank=2)
    logical_count = torch.full((2, 61, 384), 7, dtype=torch.int32)

    with (
        patch(
            "sglang.srt.eplb.expert_distribution."
            "_convert_global_physical_count_to_logical_count",
            return_value=logical_count,
        ),
        patch("torch.distributed.all_reduce") as all_reduce,
        patch("sglang.srt.eplb.expert_distribution._dump_to_file") as dump,
        patch("sglang.srt.eplb.expert_distribution.time.time", return_value=12.5),
    ):
        accumulator.dump(output_mode="file")

    all_reduce.assert_not_called()
    dump.assert_called_once()
    filename, output = dump.call_args.args
    assert filename == "expert_distribution_recorder_12.5_2.pt"
    assert output["rank"] == 2
    assert output["logical_count"] is logical_count


def test_stat_dump_preserves_world_reduction_without_pp():
    accumulator = _make_accumulator(pp_size=1, global_rank=0)
    logical_count = torch.full((2, 61, 384), 11, dtype=torch.int32)

    with (
        patch(
            "sglang.srt.eplb.expert_distribution."
            "_convert_global_physical_count_to_logical_count",
            return_value=logical_count,
        ),
        patch("torch.distributed.all_reduce") as all_reduce,
        patch("sglang.srt.eplb.expert_distribution._dump_to_file") as dump,
        patch("sglang.srt.eplb.expert_distribution.time.time", return_value=15.0),
    ):
        accumulator.dump(output_mode="file")

    all_reduce.assert_called_once_with(
        logical_count,
        op=torch.distributed.ReduceOp.SUM,
    )
    dump.assert_called_once()
    assert dump.call_args.args[0] == "expert_distribution_recorder_15.0.pt"
