import argparse
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.eplb.expert_distribution import (
    _Accumulator,
    _ExpertDistributionRecorderReal,
    _GpuExpertMaskSinglePassGatherer,
)
from sglang.srt.server_args import ServerArgs


def _metadata(*, layers: int = 2, experts: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        num_layers=layers,
        num_logical_experts=experts,
    )


def test_kt_gpu_mask_flag_enables_paired_stat_recording() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model-path", "dummy", "--record-kt-gpu-expert-distribution"]
    )
    assert parsed.record_kt_gpu_expert_distribution is True

    server_args = ServerArgs(
        model_path="dummy",
        record_kt_gpu_expert_distribution=True,
    )
    server_args._handle_expert_distribution_metrics()

    assert server_args.expert_distribution_recorder_mode == "stat"
    assert server_args.expert_distribution_recorder_buffer_size == 1000


def test_gpu_mask_gatherer_snapshots_and_validates_shape() -> None:
    gatherer = _GpuExpertMaskSinglePassGatherer(_metadata(), device="cpu")
    mask = torch.tensor([True, False, True, False])

    gatherer.on_layer(1, mask)
    snapshot = gatherer.collect()
    mask.zero_()

    assert snapshot.tolist() == [
        [False, False, False, False],
        [True, False, True, False],
    ]
    with pytest.raises(IndexError, match="outside"):
        gatherer.on_layer(2, torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="wrong shape"):
        gatherer.on_layer(0, torch.zeros(3, dtype=torch.bool))


def test_real_recorder_dumps_gpu_masks_with_logical_counts() -> None:
    main_output = {"logical_count": torch.ones(1, 2, 4, dtype=torch.int32)}
    main_accumulator = Mock()
    main_accumulator.get_single_pass_gatherer_keys.return_value = []
    main_accumulator.dump.return_value = main_output
    server_args = SimpleNamespace(
        record_kt_gpu_expert_distribution=True,
        expert_distribution_recorder_buffer_size=-1,
        enable_expert_distribution_metrics=False,
        device="cpu",
    )

    with patch.object(_Accumulator, "init_new", return_value=main_accumulator):
        recorder = _ExpertDistributionRecorderReal(
            server_args,
            _metadata(),
            rank=0,
        )

    recorder.start_record()
    recorder.on_gpu_expert_mask(
        0,
        torch.tensor([False, True, False, True]),
    )
    recorder._on_forward_pass_end(7, {})
    output = recorder.dump_record(output_mode="object")

    assert output is main_output
    assert output["gpu_expert_masks"].shape == (1, 2, 4)
    assert output["gpu_expert_masks"].tolist() == [
        [
            [False, True, False, True],
            [False, False, False, False],
        ]
    ]
    assert recorder._gpu_expert_mask_accumulator.dump().shape == (0, 2, 4)
    main_accumulator.append.assert_not_called()


def test_nonzero_rank_does_not_allocate_kt_gpu_mask_buffers() -> None:
    main_accumulator = Mock()
    main_accumulator.get_single_pass_gatherer_keys.return_value = []
    server_args = SimpleNamespace(
        record_kt_gpu_expert_distribution=True,
        expert_distribution_recorder_buffer_size=-1,
        enable_expert_distribution_metrics=False,
        device="cpu",
    )

    with patch.object(_Accumulator, "init_new", return_value=main_accumulator):
        recorder = _ExpertDistributionRecorderReal(
            server_args,
            _metadata(),
            rank=1,
        )

    assert not recorder._record_kt_gpu_expert_distribution
    recorder.on_gpu_expert_mask(0, torch.ones(4, dtype=torch.bool))
