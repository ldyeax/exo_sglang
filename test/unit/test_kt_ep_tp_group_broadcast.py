import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

with (
    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    patch.object(torch.cuda, "current_device", return_value=0),
):
    from sglang.srt.layers.moe import kt_ep_wrapper as kt


class TestKtEpTpGroupBroadcast(unittest.TestCase):
    def _group(self):
        return SimpleNamespace(
            first_rank=2,
            world_size=2,
            cpu_group=object(),
            device_group=object(),
        )

    def test_object_list_uses_global_first_rank_of_tp_group(self):
        group = self._group()
        broadcast = Mock()
        values = ["stage-local"]

        with (
            patch.object(kt.dist, "is_initialized", return_value=True),
            patch.object(kt.dist, "broadcast_object_list", broadcast),
            patch.object(kt, "get_tp_group", return_value=group),
        ):
            kt._broadcast_tp_object_list(values)

        broadcast.assert_called_once_with(
            values,
            src=2,
            group=group.cpu_group,
        )

    def test_cpu_tensor_uses_global_first_rank_and_cpu_group(self):
        group = self._group()
        broadcast = Mock()
        tensor = torch.zeros((2,), dtype=torch.bool)

        with (
            patch.object(kt.dist, "is_initialized", return_value=True),
            patch.object(kt.dist, "broadcast", broadcast),
            patch.object(kt, "get_tp_group", return_value=group),
        ):
            kt._broadcast_tp_tensor(tensor, on_device=False)

        broadcast.assert_called_once_with(
            tensor,
            src=2,
            group=group.cpu_group,
        )

    def test_device_tensor_uses_global_first_rank_and_device_group(self):
        group = self._group()
        broadcast = Mock()
        tensor = torch.zeros((2,), dtype=torch.int64)

        with (
            patch.object(kt.dist, "is_initialized", return_value=True),
            patch.object(kt.dist, "broadcast", broadcast),
            patch.object(kt, "get_tp_group", return_value=group),
        ):
            kt._broadcast_tp_tensor(tensor, on_device=True)

        broadcast.assert_called_once_with(
            tensor,
            src=2,
            group=group.device_group,
        )

    def test_uninitialized_distributed_runtime_is_a_noop(self):
        with (
            patch.object(kt.dist, "is_initialized", return_value=False),
            patch.object(kt, "get_tp_group") as get_tp_group,
        ):
            kt._broadcast_tp_object_list(["local"])
            kt._broadcast_tp_tensor(torch.zeros((1,)), on_device=False)

        get_tp_group.assert_not_called()

    def test_singleton_tp_group_skips_all_broadcasts(self):
        group = SimpleNamespace(
            first_rank=2,
            world_size=1,
            cpu_group=object(),
            device_group=object(),
        )
        with (
            patch.object(kt.dist, "is_initialized", return_value=True),
            patch.object(kt.dist, "broadcast") as broadcast,
            patch.object(kt.dist, "broadcast_object_list") as broadcast_object_list,
            patch.object(kt, "get_tp_group", return_value=group),
        ):
            kt._broadcast_tp_object_list(["local"])
            kt._broadcast_tp_tensor(torch.zeros((1,)), on_device=False)
            kt._broadcast_tp_tensor(torch.zeros((1,)), on_device=True)

        broadcast.assert_not_called()
        broadcast_object_list.assert_not_called()


if __name__ == "__main__":
    unittest.main()
