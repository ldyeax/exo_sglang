"""CPU regressions for selecting EAGLE draft-extend rows before lm_head."""

from types import SimpleNamespace

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_select_index_prunes_logits_input_and_last_hidden_together():
    processor = LogitsProcessor.__new__(LogitsProcessor)
    hidden_states = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    select_index = torch.tensor([2, 7], dtype=torch.int64)
    metadata = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        draft_extend_select_index=select_index,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )

    (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        _,
        _,
    ) = processor._get_pruned_states(hidden_states, None, None, metadata)
    stored_hidden_states = processor._get_hidden_states_to_store(
        hidden_states,
        None,
        None,
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        metadata,
    )

    expected = hidden_states[select_index]
    torch.testing.assert_close(pruned_states, expected)
    torch.testing.assert_close(stored_hidden_states, expected)


class _Event:
    def record(self):
        pass


class _DraftExtendBackend:
    def __init__(self):
        self.replay_select_index = None

    def init_forward_metadata_out_graph(self, forward_batch):
        self.replay_select_index = forward_batch.spec_info.select_index.clone()


def test_graph_replay_copies_static_select_index_and_returns_request_rows():
    raw_batch_size = 2
    capture_batch_size = 4
    request_width = 4
    raw_num_tokens = raw_batch_size * request_width
    max_num_tokens = capture_batch_size * request_width
    select_index = torch.tensor([1, 7], dtype=torch.int64)

    runner = EAGLEDraftExtendCudaGraphRunner.__new__(EAGLEDraftExtendCudaGraphRunner)
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    runner.prune_draft_extend_logits = True
    runner.require_mlp_tp_gather = False
    runner.require_gathered_buffer = False
    runner.captured_req_width = request_width
    runner.capture_bs = [1, capture_batch_size]
    runner.seq_len_fill_value = 1
    runner.extend_seq_lens_cpu = [request_width] * capture_batch_size
    runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    runner.device_module = SimpleNamespace(Event=_Event)
    runner.model_runner = SimpleNamespace(device_timer=None)
    runner.draft_extend_attn_backend = _DraftExtendBackend()
    runner.buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int64),
        seq_lens=torch.zeros(capture_batch_size, dtype=torch.int64),
        out_cache_loc=torch.zeros(max_num_tokens, dtype=torch.int64),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64),
        req_pool_indices=torch.zeros(capture_batch_size, dtype=torch.int64),
        extend_seq_lens=torch.zeros(capture_batch_size, dtype=torch.int32),
        num_correct_drafts=torch.zeros(capture_batch_size, dtype=torch.int32),
        num_accept_tokens=torch.zeros(capture_batch_size, dtype=torch.int32),
        select_index=(
            torch.arange(capture_batch_size, dtype=torch.int64) * request_width
            + request_width
            - 1
        ),
        hidden_states=torch.zeros(max_num_tokens, 3),
        seq_lens_cpu=torch.zeros(capture_batch_size, dtype=torch.int64),
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
    )
    graph_logits = torch.arange(capture_batch_size * 5, dtype=torch.float32).reshape(
        capture_batch_size, 5
    )
    graph_hidden = torch.arange(capture_batch_size * 3, dtype=torch.float32).reshape(
        capture_batch_size, 3
    )
    runner._replay_graph = lambda *_: LogitsProcessorOutput(
        next_token_logits=graph_logits,
        hidden_states=graph_hidden,
    )

    forward_batch = SimpleNamespace(
        batch_size=raw_batch_size,
        input_ids=torch.arange(raw_num_tokens, dtype=torch.int64),
        seq_lens=torch.tensor([9, 10], dtype=torch.int64),
        out_cache_loc=torch.arange(raw_num_tokens, dtype=torch.int64),
        positions=torch.arange(raw_num_tokens, dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        extend_seq_lens=torch.full((raw_batch_size,), request_width, dtype=torch.int32),
        seq_lens_cpu=torch.tensor([9, 10], dtype=torch.int64),
        extend_seq_lens_cpu=[request_width] * raw_batch_size,
        seq_lens_sum=19,
        spec_info=SimpleNamespace(
            hidden_states=torch.zeros(raw_num_tokens, 3),
            num_correct_drafts=torch.tensor([0, 3], dtype=torch.int32),
            num_accept_tokens=torch.tensor([1, 4], dtype=torch.int32),
            select_index=select_index,
        ),
    )

    output = runner.execute(forward_batch)

    torch.testing.assert_close(
        runner.buffers.select_index[:raw_batch_size], select_index
    )
    torch.testing.assert_close(
        runner.draft_extend_attn_backend.replay_select_index[:raw_batch_size],
        select_index,
    )
    assert output.next_token_logits.shape == (raw_batch_size, 5)
    assert output.hidden_states.shape == (raw_batch_size, 3)
    torch.testing.assert_close(output.next_token_logits, graph_logits[:raw_batch_size])
    torch.testing.assert_close(output.hidden_states, graph_hidden[:raw_batch_size])
