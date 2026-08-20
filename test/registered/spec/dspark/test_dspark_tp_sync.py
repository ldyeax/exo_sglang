from contextlib import nullcontext
from types import SimpleNamespace

import torch
from sglang.srt.speculative.dspark_components import dspark_worker_v2
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    CommitInjectCtx,
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingGroup:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.calls: list[tuple[torch.Tensor, int]] = []

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.calls.append((tensor, src))


class _ParallelContext:
    def __init__(
        self,
        group: _RecordingGroup,
        *,
        attention_group: _RecordingGroup | None = None,
    ) -> None:
        self.tp_group = group
        self.attn_tp_group = attention_group or group


def test_dspark_broadcasts_each_tensor_from_tp_rank_zero(monkeypatch) -> None:
    group = _RecordingGroup(world_size=2)
    monkeypatch.setattr(dspark_worker_v2, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(
        dspark_worker_v2, "get_parallel", lambda: _ParallelContext(group)
    )
    tensors = (torch.tensor([1]), torch.tensor([2]))

    dspark_worker_v2.broadcast_dspark_tensors(tensors)

    assert group.calls == [(tensors[0], 0), (tensors[1], 0)]


def test_dspark_skips_broadcast_for_single_rank(monkeypatch) -> None:
    group = _RecordingGroup(world_size=1)
    monkeypatch.setattr(dspark_worker_v2, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(
        dspark_worker_v2, "get_parallel", lambda: _ParallelContext(group)
    )

    dspark_worker_v2.broadcast_dspark_tensors((torch.tensor([1]),))

    assert group.calls == []


def test_dspark_dp_attention_broadcasts_only_within_attention_tp_group(
    monkeypatch,
) -> None:
    enclosing_group = _RecordingGroup(world_size=4)
    attention_group = _RecordingGroup(world_size=1)
    monkeypatch.setattr(dspark_worker_v2, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(
        dspark_worker_v2,
        "get_parallel",
        lambda: _ParallelContext(
            enclosing_group,
            attention_group=attention_group,
        ),
    )

    dspark_worker_v2.broadcast_dspark_tensors((torch.tensor([1]),))

    assert enclosing_group.calls == []
    assert attention_group.calls == []


def test_dspark_prefill_synchronizes_first_decode_anchor(monkeypatch) -> None:
    events: list[object] = []
    next_token_ids = torch.tensor([7], dtype=torch.int64)
    hidden_states = torch.tensor([[1.0]])
    batch_output = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden_states),
        next_token_ids=next_token_ids,
        new_seq_lens=None,
        next_draft_input=None,
    )
    target_worker = SimpleNamespace(
        forward_batch_generation=lambda *args, **kwargs: batch_output
    )
    kv_injector = SimpleNamespace(
        inject_target_hidden=lambda **kwargs: events.append("inject")
    )
    worker = object.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker._target_worker = target_worker
    worker.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend="dsv4")
    )
    worker._kv_injector = kv_injector

    def synchronize(tensors) -> None:
        events.append("broadcast")
        tensors[0].fill_(11)

    def make_draft_input(*, bonus_tokens, new_seq_lens):
        events.append(("draft", bonus_tokens.clone()))
        return SimpleNamespace(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
        )

    monkeypatch.setattr(dspark_worker_v2, "broadcast_dspark_tensors", synchronize)
    monkeypatch.setattr(dspark_worker_v2, "make_next_draft_input", make_draft_input)
    monkeypatch.setattr(
        dspark_worker_v2,
        "compute_position",
        lambda *args, **kwargs: (torch.tensor([0]), None),
    )

    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        extend_lens=[1],
        prefix_lens=[0],
        out_cache_loc=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
    )
    result = worker._forward_prefill(batch, on_publish=None)

    assert events[0] == "broadcast"
    assert events[1] == "inject"
    assert events[2][0] == "draft"
    assert torch.equal(result.next_token_ids, torch.tensor([11]))
    assert torch.equal(result.next_draft_input.bonus_tokens, torch.tensor([11]))


def _commit_ctx(*, tp_size: int, resolve_pool) -> CommitInjectCtx:
    return CommitInjectCtx(
        draft_model=object(),
        block_pos_offsets=torch.arange(3),
        resolve_pool=resolve_pool,
        resolve_req_to_token=lambda: torch.zeros((1, 3), dtype=torch.int64),
        tp_size=tp_size,
    )


def test_dspark_tp2_defers_folded_commit_until_after_accept_sync() -> None:
    def fail_if_resolved():
        raise AssertionError("TP2 must not inspect or write the KV pool in-graph")

    epilogue = DsparkVerifyEpilogue(
        max_bs=1,
        verify_num_draft_tokens=3,
        device="cpu",
        commit_ctx=_commit_ctx(tp_size=2, resolve_pool=fail_if_resolved),
    )

    assert not epilogue.folds_commit


def test_dspark_tp1_keeps_folded_commit_fast_path() -> None:
    pool = SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=object())
    epilogue = DsparkVerifyEpilogue(
        max_bs=1,
        verify_num_draft_tokens=3,
        device="cpu",
        commit_ctx=_commit_ctx(tp_size=1, resolve_pool=lambda: pool),
    )

    assert epilogue.folds_commit


def test_dspark_tp2_broadcasts_accept_before_eager_commit(monkeypatch) -> None:
    events: list[object] = []

    class FakeDraftInput:
        pass

    class FakeSeqLens:
        def record_stream(self, stream) -> None:
            events.append("record_stream")

        def __len__(self) -> int:
            return 1

    class FakeObservers:
        def begin_step(self) -> None:
            pass

        def segment(self, segment):
            return nullcontext()

        def observe_verify_step(self, **kwargs) -> None:
            pass

    authoritative_commit_len = 2
    local_accept = AcceptOuts(
        correct_len=torch.tensor([0], dtype=torch.int64),
        bonus=torch.tensor([17], dtype=torch.int64),
        cap_trim_lens=torch.tensor([0], dtype=torch.int32),
        commit_lens=torch.tensor([1], dtype=torch.int32),
        new_seq_lens=torch.tensor([11], dtype=torch.int64),
        out_tokens=torch.tensor([[17, 0, 0]], dtype=torch.int64),
    )
    epilogue = SimpleNamespace(folds_commit=False)

    def synchronize(tensors) -> None:
        if len(tensors) == 1:
            events.append("proposal_broadcast")
            return
        events.append("accept_broadcast")
        tensors[3].fill_(authoritative_commit_len)

    def commit_hidden(**kwargs) -> None:
        events.append(("commit", kwargs["commit_lens"].clone()))

    verify_executor = SimpleNamespace(
        verify_epilogue=epilogue,
        run_compact=lambda **kwargs: (
            SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.zeros((3, 8)),
                    hidden_states=torch.zeros((3, 4)),
                ),
                can_run_cuda_graph=True,
            ),
            torch.zeros((3, 4)),
        ),
        accept_and_finalize=lambda **kwargs: local_accept,
        commit_hidden=commit_hidden,
    )
    draft_tokens = torch.tensor([[6, 7]], dtype=torch.int64)
    proposal = SimpleNamespace(
        draft_block_ids=torch.tensor([[5]], dtype=torch.int64),
        draft_block=SimpleNamespace(draft_tokens=draft_tokens),
        draft_tokens=draft_tokens,
        draft_hidden=None,
        confidence=torch.tensor([1.0]),
        confidence_tap=None,
        folded=True,
    )
    layout = SimpleNamespace(verify_lens=torch.tensor([3], dtype=torch.int64))
    planner = SimpleNamespace(
        resolve_verify_token_budget=lambda **kwargs: None,
        schedule_layout=lambda **kwargs: layout,
        should_run_compact=lambda **kwargs: True,
    )

    worker = object.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker.device = "cpu"
    worker.gamma = 2
    worker.verify_num_draft_tokens = 3
    worker._block_pos_offsets = torch.arange(3)
    worker._simulate_acc_len = 0.0
    worker._draft_dp_context_enabled = False
    worker._draft_is_moe = False
    worker._observers = FakeObservers()
    worker._proposer = SimpleNamespace(propose=lambda **kwargs: proposal)
    worker._verify_planner = planner
    worker._verify_executor = verify_executor
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=object())
    )
    worker.model_runner = worker._target_worker.model_runner
    worker.server_args = SimpleNamespace(enable_dp_attention=False)
    worker._commit_target_mamba_states_after_verify = lambda **kwargs: events.append(
        ("mamba_commit", kwargs["commit_lens"].clone())
    )

    monkeypatch.setattr(dspark_worker_v2, "DFlashDraftInputV2", FakeDraftInput)
    monkeypatch.setattr(
        dspark_worker_v2, "alloc_verify_window", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        dspark_worker_v2, "prepare_mamba_track_for_verify", lambda batch: None
    )
    monkeypatch.setattr(dspark_worker_v2, "broadcast_dspark_tensors", synchronize)
    monkeypatch.setattr(
        dspark_worker_v2,
        "make_next_draft_input",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    batch = SimpleNamespace(
        spec_info=FakeDraftInput(),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=FakeSeqLens(),
        sampling_info=None,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        has_grammar=False,
        forward_iter=0,
        reqs=[],
        spec_verify_tier_num_tokens=3,
    )
    worker._forward_decode(batch, on_publish=None)

    accept_broadcast_index = events.index("accept_broadcast")
    mamba_commit_index = next(
        index for index, event in enumerate(events) if event[0] == "mamba_commit"
    )
    kv_commit_index = next(
        index for index, event in enumerate(events) if event[0] == "commit"
    )
    assert accept_broadcast_index < mamba_commit_index < kv_commit_index
    assert torch.equal(
        events[mamba_commit_index][1], torch.tensor([2], dtype=torch.int32)
    )
    assert torch.equal(events[kv_commit_index][1], torch.tensor([2], dtype=torch.int32))


def test_dspark_replayssm_fold_commits_the_authoritative_chain_lens(
    monkeypatch,
) -> None:
    recorded: dict[str, object] = {}
    spec_state = object()

    def record_fold(**kwargs) -> None:
        recorded.update(kwargs)

    class LegacyBackend:
        def update_mamba_state_after_mtp_verify(self, **kwargs) -> None:
            raise AssertionError("ReplaySSM fold must not use the legacy state scatter")

    req_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(
            replayssm_spec_fold=True,
            replayssm_is_kda=False,
        ),
        get_speculative_mamba2_params_all_layers=lambda: spec_state,
        get_mamba_indices=lambda indices: indices + 10,
    )
    worker = object.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker._need_mamba_verify_commit = True
    worker.server_args = SimpleNamespace(
        speculative_eagle_topk=1,
        mamba_track_interval=4,
    )
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=LegacyBackend(),
            req_to_token_pool=req_pool,
        )
    )
    monkeypatch.setattr(
        dspark_worker_v2,
        "_commit_gdn_replayssm_fold_after_verify",
        record_fold,
    )

    commit_lens = torch.tensor([2], dtype=torch.int32)
    mamba_track_indices = torch.tensor([7], dtype=torch.int64)
    worker._commit_target_mamba_states_after_verify(
        batch=SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            mamba_track_indices=mamba_track_indices,
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
        ),
        seq_lens_pre_verify=torch.tensor([3], dtype=torch.int64),
        seq_lens_post_verify=torch.tensor([5], dtype=torch.int64),
        commit_lens=commit_lens,
    )

    assert recorded["spec_state"] is spec_state
    assert torch.equal(recorded["state_batch_indices"], torch.tensor([13]))
    assert recorded["accept_lens"] is commit_lens
    assert torch.equal(recorded["last_correct_step_indices"], torch.tensor([1]))
    assert recorded["mamba_track_indices"] is mamba_track_indices
    assert torch.equal(recorded["mamba_steps_to_track"], torch.tensor([0]))
    assert recorded["null_block_id"] == -1
