from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("rank", "expected_calls"),
    [
        (0, [("send", 23)]),
        (1, [("recv", 17), ("send", 42)]),
        (2, [("recv", 23)]),
    ],
)
def test_pipeline_prewarm_uses_forward_edges(
    monkeypatch, rank: int, expected_calls: list[tuple[str, int]]
) -> None:
    calls = []
    group = object()
    tensor = torch.empty(1, dtype=torch.uint8)
    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: tensor)
    monkeypatch.setattr(
        torch.distributed,
        "recv",
        lambda _tensor, src, group: calls.append(("recv", src)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "send",
        lambda _tensor, dst, group: calls.append(("send", dst)),
    )
    monkeypatch.setattr(
        kv_cache_configurator.current_platform,
        "synchronize",
        lambda: calls.append(("sync", -1)),
    )
    pp_group = SimpleNamespace(
        world_size=3,
        rank_in_group=rank,
        ranks=[17, 23, 42],
        device_group=group,
    )

    kv_cache_configurator._preinitialize_pipeline_communicators(
        pp_group=pp_group, device="cpu"
    )

    assert calls == [*expected_calls, ("sync", -1)]


@pytest.mark.parametrize(
    ("pp_size", "expected_distributed"), [(1, True), (2, False)]
)
def test_pipeline_profiles_local_bytes_before_token_reduction(
    monkeypatch, pp_size: int, expected_distributed: bool
) -> None:
    events = []
    cpu_group = object()
    monkeypatch.setattr(
        kv_cache_configurator,
        "_preinitialize_pipeline_communicators",
        lambda **_kwargs: events.append("prewarm"),
    )

    def available_memory(_device, _gpu_id, *, distributed, cpu_group):
        events.append(("profile", distributed, cpu_group))
        return 6.0

    monkeypatch.setattr(
        kv_cache_configurator, "get_available_gpu_memory", available_memory
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_world_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=cpu_group),
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_schedule",
        lambda: SimpleNamespace(mem_fraction_static=1.0),
    )
    fake = SimpleNamespace(
        pp_group=SimpleNamespace(world_size=pp_size),
        device="cpu",
        gpu_id=0,
        mambaish_config=None,
    )

    available_bytes = KVCacheConfigurator._profile_available_bytes(fake, 8)

    assert available_bytes == 6 * (1 << 30)
    assert events == ["prewarm", ("profile", expected_distributed, cpu_group)]


def test_pipeline_reduces_derived_token_capacity(monkeypatch) -> None:
    cpu_group = object()
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_schedule",
        lambda: SimpleNamespace(max_total_tokens=None),
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_world_group",
        lambda: SimpleNamespace(cpu_group=cpu_group),
    )

    def reduce_to_common_capacity(tensor, *, op, group) -> None:
        assert op is torch.distributed.ReduceOp.MIN
        assert group is cpu_group
        tensor.fill_(5)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_to_common_capacity)
    fake = SimpleNamespace(server_args=SimpleNamespace(pp_size=2))

    assert KVCacheConfigurator._apply_token_constraints(fake, 8) == 5
