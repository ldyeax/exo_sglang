import json
import logging
import os
from contextlib import ExitStack, contextmanager, nullcontext
from copy import copy
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.srt.distributed.parallel_state import patch_pipeline_parallel_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftBlockProposer,
    make_next_draft_input,
    maybe_build_draft_sampler,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


class LazySafetensorTokenEmbedding(torch.nn.Module):
    """Page only requested embedding rows for a last-only PP draft."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        checkpoint = Path(model_path)
        index_path = checkpoint / "model.safetensors.index.json"
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        weight_name = next(
            (
                candidate
                for candidate in ("embed.weight", "model.embed_tokens.weight")
                if candidate in weight_map
            ),
            None,
        )
        if weight_name is None:
            raise RuntimeError(
                f"PP DSpark could not find an embedding tensor in {index_path}."
            )
        self.weight_name = weight_name
        self.weight_path = checkpoint / weight_map[weight_name]

    @eager_on_graph(True)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # The last-only PP draft intentionally has no full GPU embedding table.
        # Row paging contains a CUDA->CPU ID copy plus host safetensors access,
        # neither of which can be recorded in a CUDA graph.  Keep this small
        # operation as an eager break between graph segments; its GPU output is
        # bridged back into the captured draft graph.
        flat_ids = input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        with safe_open(self.weight_path, framework="pt", device="cpu") as weights:
            tensor_slice = weights.get_slice(self.weight_name)
            rows = torch.cat(
                [
                    tensor_slice[int(token_id) : int(token_id) + 1]
                    for token_id in unique_ids.tolist()
                ],
                dim=0,
            )
        embedded = rows[inverse].reshape(*input_ids.shape, rows.shape[-1])
        return embedded.to(device=input_ids.device, non_blocking=True)


class DSparkWorkerV2(BaseSpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device
        self._is_pipeline_parallel = server_args.pp_size > 1
        self._is_last_pipeline_rank = bool(
            target_worker.model_runner.pp_group.is_last_rank
        )
        self._has_local_draft = (
            not self._is_pipeline_parallel or self._is_last_pipeline_rank
        )
        self._draft_pp_group = None
        if self._is_pipeline_parallel and self._is_last_pipeline_rank:
            self._draft_pp_group = copy(target_worker.model_runner.pp_group)
            self._draft_pp_group.world_size = 1
            self._draft_pp_group.rank_in_group = 0
            self._draft_pp_group.ranks = [self._draft_pp_group.rank]

        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        attn_tp_size = server_args.tp_size // max(server_args.dp_size, 1)
        if server_args.enable_dp_attention and self._draft_is_moe and attn_tp_size > 1:
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        bundle = None
        if self._has_local_draft:
            with self._draft_context():
                bundle = build_draft_tp_worker(
                    server_args=server_args,
                    gpu_id=gpu_id,
                    tp_rank=tp_rank,
                    dp_rank=dp_rank,
                    moe_ep_rank=moe_ep_rank,
                    attn_cp_rank=attn_cp_rank,
                    moe_dp_rank=moe_dp_rank,
                    nccl_port=nccl_port,
                    target_model_config=target_worker.model_runner.model_config,
                    algo_label="DSPARK",
                    attention_backend_override=(
                        DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                    ),
                    standalone_pipeline=self._is_pipeline_parallel,
                )
        self._draft_worker = None if bundle is None else bundle.draft_worker
        self.draft_model_runner = (
            None if bundle is None else bundle.draft_model_runner
        )
        self.draft_model = None if bundle is None else bundle.draft_model
        self._draft_sampler = None

        runtime_config = resolve_runtime_config(
            draft_hf_config=(
                self.draft_model_runner.model_config.hf_config
                if self.draft_model_runner is not None
                else target_worker.model_runner.model_config.hf_config
            ),
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            target_vocab_size=int(
                self.target_worker.model_runner.model_config.vocab_size
            ),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id

        if self.tp_rank == 0 and self._has_local_draft:
            assert bundle is not None
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, mask_token_id=%s, "
                "markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )
        elif self.tp_rank == 0:
            logger.info(
                "Initialized PP DSpark target-only stage; the complete draft "
                "runner is placed on the last pipeline rank."
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.gamma), device=self.device
        )

        self._verify_planner = None
        self._kv_injector = None
        self._proposer = None
        if self._has_local_draft:
            assert self.draft_model is not None
            assert self.draft_model_runner is not None
            target_model = self.target_worker.model_runner.model
            lm_head = getattr(target_model, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError(
                    "DSpark requires the target model to expose `lm_head` with `weight`."
                )
            self.draft_model.attach_shared_modules(
                embed_tokens=self._resolve_target_embed_tokens(target_model),
                lm_head=lm_head,
            )

            self._verify_planner = DSparkVerifyPlanner(
                draft_model=self.draft_model,
                gamma=self.gamma,
                model_runner=self.model_runner,
                device=self.device,
                tp_rank=self.tp_rank,
                server_args=self.server_args,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
            )
            if (
                server_args.enable_dp_attention
                and not self._draft_is_moe
                and self._verify_planner.is_compact_mode
                and not server_args.disable_cuda_graph
            ):
                raise ValueError(
                    "DSpark dense-draft compact verify under --enable-dp-attention does not "
                    "yet support cuda graph (idle DP groups cannot join the token-keyed "
                    "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                    "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                    "cuda graph under DP."
                )
            self._kv_injector = TargetHiddenKvInjector(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                model_runner=self.model_runner,
                device=self.device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
            )
            self._proposer = DraftBlockProposer(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                gamma=self.gamma,
                mask_token_id=self._mask_token_id,
                draft_block_spec_info=self._draft_block_spec_info,
                dp_moe_sync=self._draft_is_moe and server_args.enable_dp_attention,
            )
        self._verify_epilogue = None
        if (
            self._verify_planner is not None
            and self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
            and is_cuda()
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(server_args.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and self._verify_planner is not None
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

        self._forced_budget_frac: Optional[float] = None

        self._observers = (
            DsparkStepObservers(
                planner=self._verify_planner,
                gamma=self.gamma,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                tp_rank=self.tp_rank,
                device=self.device,
                simulate_acc_len=self._simulate_acc_len,
            )
            if self._verify_planner is not None
            else None
        )

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            embed_tokens = target_model.get_input_embeddings()
        else:
            embed_tokens = target_model.model.get_input_embeddings()
        if hasattr(embed_tokens, "weight"):
            return embed_tokens
        if self._is_pipeline_parallel and self._is_last_pipeline_rank:
            logger.info(
                "PP DSpark last stage uses lazy safetensors embedding rows "
                "from %s.",
                self.server_args.model_path,
            )
            return LazySafetensorTokenEmbedding(self.server_args.model_path)
        return embed_tokens

    @property
    def carries_confidence(self) -> bool:
        return (
            self._verify_planner is not None
            and self._verify_planner.carries_confidence
        )

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        target_backend = self._target_worker.model_runner.attn_backend
        if self.draft_model_runner is None:
            return (target_backend,)
        return (target_backend, self.draft_model_runner.attn_backend)

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    @contextmanager
    def _draft_context(self):
        with ExitStack() as stack:
            if self._draft_pp_group is not None:
                stack.enter_context(
                    patch_pipeline_parallel_group(self._draft_pp_group)
                )
            if self._draft_dp_context_enabled:
                stack.enter_context(draft_tp_context(get_parallel().attn_tp_group))
            else:
                stack.enter_context(nullcontext())
            yield

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if self._draft_worker is None:
            return
        with self._draft_context():
            self._draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )

    def init_attention_backends(self):
        if self._draft_worker is None:
            return
        with self._draft_context():
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        if self._draft_worker is None:
            return
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if os.environ.get("SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH") == "1":
            capture_decode_cuda_graph = False
            if self.tp_rank == 0:
                logger.info(
                    "Disable DSpark draft CUDA graph by "
                    "SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH=1; "
                    "target verify CUDA graph remains enabled."
                )
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            # The SM86 V4 draft graph is about 70 MiB. Keep a conservative
            # margin without discarding speculation merely for being <1 GiB.
            if available_mem < 0.25:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                disable_folded_sampler = (
                    os.environ.get("SGLANG_DSV4_DRAFT_DISABLE_FOLDED_SAMPLER") == "1"
                )
                self._draft_sampler = (
                    None
                    if disable_folded_sampler
                    else self._maybe_build_draft_sampler()
                )
                if disable_folded_sampler and self.tp_rank == 0:
                    logger.info(
                        "DSpark draft model CUDA graph enabled with eager proposal "
                        "sampling by SGLANG_DSV4_DRAFT_DISABLE_FOLDED_SAMPLER=1."
                    )
                if self._draft_sampler is not None:
                    self.draft_model_runner.capture_tail_hooks.append(
                        make_draft_sampler_capture_hook(self._draft_sampler)
                    )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self):
        if self.draft_model is None or self._verify_planner is None:
            return None
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.tp_rank,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        self._forced_budget_frac = frac
        if self._verify_planner is not None:
            self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        return None if self._observers is None else self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        if self._observers is not None:
            self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        if self._observers is None:
            return None
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if self._observers is not None:
            self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if self._verify_planner is not None:
                self._verify_planner.note_non_decode_step()
            if self._observers is not None:
                self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish, pp_proxy_tensors)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if self.server_args.enable_dp_attention:
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
                self.target_worker.forward_batch_generation(
                    batch, pp_proxy_tensors=pp_proxy_tensors
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(
            batch, pp_proxy_tensors=pp_proxy_tensors
        )
        if not self._is_last_pipeline_rank:
            return batch_output
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")
        if self._kv_injector is None:
            raise RuntimeError("PP DSpark last stage has no target-hidden KV injector.")

        device = next_token_ids.device
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if self._is_pipeline_parallel:
            return self._forward_decode_pp(
                batch=batch,
                on_publish=on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            assert self._observers is not None
            self._observers.note_idle_decode_step()
            if self.server_args.enable_dp_attention:
                if self._draft_is_moe:
                    self._proposer.run_idle_participation(batch)
                self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        self._observers.begin_step()
        assert self._verify_planner is not None
        assert self._proposer is not None

        target_model = self.target_worker.model_runner.model

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        confidence = proposal.confidence
        if confidence is None:
            confidence = self._verify_planner.compute_confidence_tensor(
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
                confidence_tap=proposal.confidence_tap,
            )

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
        )
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    inject_gate=fold_eligible,
                )
            else:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = fold_eligible and run_compact and can_run_cuda_graph
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        folded_commit = folded_accept and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal.folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )

    def _forward_decode_pp(
        self,
        *,
        batch: ScheduleBatch,
        on_publish,
        pp_proxy_tensors,
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "PP DSpark expected DFlashDraftInputV2 state on every pipeline rank."
            )
        if batch.forward_mode.is_idle():
            raise RuntimeError("PP DSpark does not support DP idle batches.")

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        if draft_input.pp_draft_block_ids is None:
            return self._forward_decode_pp_bootstrap(
                batch=batch,
                draft_input=draft_input,
                on_publish=on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        return self._forward_decode_pp_verify(
            batch=batch,
            draft_input=draft_input,
            on_publish=on_publish,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def _forward_decode_pp_bootstrap(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        on_publish,
        pp_proxy_tensors,
    ) -> GenerationBatchResult:
        """Run one ordinary target token, then seed the first ring proposal."""
        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )
        bootstrap_cache_loc = verify_window.verify_cache_loc_2d[:, 0].contiguous()
        batch.input_ids = draft_input.bonus_tokens.reshape(-1)
        batch.out_cache_loc = bootstrap_cache_loc
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(
            batch, pp_proxy_tensors=pp_proxy_tensors
        )
        if not self._is_last_pipeline_rank:
            return batch_output

        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        if logits_output is None or logits_output.hidden_states is None:
            raise RuntimeError(
                "PP DSpark bootstrap requires target aux hidden states on the "
                "last pipeline rank."
            )
        if next_token_ids is None:
            raise RuntimeError("PP DSpark bootstrap target produced no sampled token.")
        if self._kv_injector is None:
            raise RuntimeError("PP DSpark bootstrap has no draft KV injector.")

        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=bootstrap_cache_loc,
            positions=prefix_lens,
        )
        logits_output.hidden_states = None

        new_seq_lens = prefix_lens + 1
        next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=new_seq_lens,
        )
        self._attach_next_pp_proposal(
            batch=batch,
            draft_input=next_draft_input,
            new_seq_lens=new_seq_lens,
        )

        padded_tokens = torch.zeros(
            (bs, self.verify_num_draft_tokens),
            dtype=torch.int64,
            device=self.device,
        )
        padded_tokens[:, 0].copy_(next_token_ids.reshape(-1))
        accept_lens = torch.ones((bs,), dtype=torch.int32, device=self.device)
        if on_publish is not None:
            on_publish(new_seq_lens)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=padded_tokens.reshape(-1),
            accept_lens=accept_lens,
            block_accept_lens=accept_lens,
            can_run_cuda_graph=batch_output.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=new_seq_lens,
        )

    def _forward_decode_pp_verify(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        on_publish,
        pp_proxy_tensors,
    ) -> GenerationBatchResult:
        draft_block_ids = draft_input.pp_draft_block_ids
        draft_tokens = draft_input.pp_draft_tokens
        greedy_mask = draft_input.pp_greedy_mask
        temperatures = draft_input.pp_temperatures
        if (
            draft_block_ids is None
            or draft_tokens is None
            or greedy_mask is None
            or temperatures is None
        ):
            raise RuntimeError("PP DSpark received an incomplete draft proposal.")

        corrected_logits = draft_input.pp_corrected_logits
        if corrected_logits is not None and corrected_logits.numel() == 0:
            corrected_logits = None
        draft_block = DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=corrected_logits,
            greedy_mask=greedy_mask,
            temperatures=temperatures,
        )
        confidence = draft_input.pp_confidence
        if confidence is not None and confidence.numel() == 0:
            confidence = None

        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )
        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()
        target_verify = self._verify_executor.run_non_compact(
            batch=batch,
            draft_input=draft_input,
            verify_ids_2d=verify_ids_2d,
            verify_window=verify_window,
            sampling_info=batch.sampling_info,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self._is_last_pipeline_rank:
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=(
                    target_verify.pp_hidden_states_proxy_tensors
                ),
                can_run_cuda_graph=target_verify.can_run_cuda_graph,
            )

        logits_output = target_verify.logits_output
        if logits_output is None:
            raise RuntimeError("PP DSpark last stage returned no target logits.")
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=False,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=batch.sampling_info,
            draft_input=draft_input,
            layout=None,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        if on_publish is not None:
            on_publish(accept.new_seq_lens, confidence=confidence)

        self._verify_executor.commit_hidden(
            batch=batch,
            layout=None,
            hidden_strided=None,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=accept.commit_lens,
            bs=bs,
            run_compact=False,
        )
        logits_output.hidden_states = None

        if self._observers is not None:
            self._observers.begin_step()
            self._observers.observe_verify_step(
                forward_ct=int(batch.forward_iter),
                reqs=batch.reqs,
                bs=bs,
                proposal_folded=False,
                verify_ids_2d=verify_ids_2d,
                target_logits=logits_output.next_token_logits,
                layout=None,
                confidence=confidence,
                prefix_lens=prefix_lens,
                draft_tokens=draft_tokens,
                draft_block=draft_block,
                sampling_info=batch.sampling_info,
                correct_len=accept.correct_len,
                cap_trim_lens=accept.cap_trim_lens,
                bonus=accept.bonus,
                commit_lens=accept.commit_lens,
                verify_token_budget=None,
                req_pool_indices=batch.req_pool_indices,
                verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
                dp_tier_num_tokens=None,
            )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        self._attach_next_pp_proposal(
            batch=batch,
            draft_input=next_draft_input,
            new_seq_lens=accept.new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            can_run_cuda_graph=target_verify.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )

    def _attach_next_pp_proposal(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        new_seq_lens: torch.Tensor,
    ) -> None:
        if (
            self._proposer is None
            or self._verify_planner is None
            or self.draft_model is None
        ):
            raise RuntimeError("PP DSpark proposal attempted without a local draft.")

        saved_seq_lens = batch.seq_lens
        saved_seq_lens_cpu = batch.seq_lens_cpu
        saved_seq_lens_sum = batch.seq_lens_sum
        saved_out_cache_loc = batch.out_cache_loc
        try:
            batch.seq_lens = new_seq_lens
            batch.seq_lens_cpu = new_seq_lens.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            verify_window = alloc_verify_window(
                batch=batch,
                bs=len(new_seq_lens),
                device=self.device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
                model_runner=self.model_runner,
            )
            with self._draft_context():
                proposal = self._proposer.propose(
                    batch=batch,
                    draft_input=draft_input,
                    verify_window=verify_window,
                    bs=len(new_seq_lens),
                    device=self.device,
                    target_model=self.target_worker.model_runner.model,
                    sampling_info=batch.sampling_info,
                )
            confidence = proposal.confidence
            if confidence is None and self._verify_planner.carries_confidence:
                confidence = self._verify_planner.compute_confidence_tensor(
                    draft_hidden=proposal.draft_hidden,
                    anchor_tokens=proposal.draft_block_ids[:, 0],
                    draft_tokens=proposal.draft_block.draft_tokens,
                    confidence_tap=proposal.confidence_tap,
                )
            draft_input.pp_draft_block_ids = proposal.draft_block_ids
            draft_input.pp_draft_tokens = proposal.draft_block.draft_tokens
            draft_input.pp_corrected_logits = proposal.draft_block.corrected_logits
            draft_input.pp_greedy_mask = proposal.draft_block.greedy_mask
            draft_input.pp_temperatures = proposal.draft_block.temperatures
            draft_input.pp_confidence = confidence
        finally:
            batch.seq_lens = saved_seq_lens
            batch.seq_lens_cpu = saved_seq_lens_cpu
            batch.seq_lens_sum = saved_seq_lens_sum
            batch.out_cache_loc = saved_out_cache_loc

    def get_confidence_budget_prepare(self):
        if self._verify_planner is None:
            return None
        return self._verify_planner.confidence_budget_prepare()
