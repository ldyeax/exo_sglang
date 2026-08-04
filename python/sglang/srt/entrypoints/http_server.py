# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import hashlib
import logging
import os
import ssl
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import aiohttp
import numpy as np
import requests
import torch
import uvicorn
import uvloop
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.configs.embedding_model_spec import resolved_embedding_plan
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.engine import (
    Engine,
    init_tokenizer_manager,
    run_detokenizer_process,
    run_scheduler_process,
)
from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ClassifyRequest,
    CompletionRequest,
    DetokenizeRequest,
    EmbeddingRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ResponsesRequest,
    ScoringRequest,
    TokenizeRequest,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,
    OpenAIServingTokenize,
)
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.entrypoints.request_headers import apply_header_overrides
from sglang.srt.entrypoints.warmup import execute_warmups
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import (
    AbortReq,
    AttachHiCacheStorageReqInput,
    CheckWeightsReqInput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DumperControlReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    KTExpertHotspotReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SendWeightsToRemoteInstanceReqInput,
    SeparateReasoningReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    TokenizerWorker,
    get_main_process_id,
    get_tokenizer_worker_class,
    read_from_shared_memory,
    write_data_for_multi_tokenizer,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.srt.observability.func_timer import enable_func_timer
from sglang.srt.observability.trace import (
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_prometheus_middleware,
    add_prometheus_track_response_middleware,
    delete_directory,
    get_bool_env_var,
    is_mps,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.srt.utils.auth import AuthLevel, app_has_admin_force_endpoints, auth_level
from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,
    dumps_json,
    orjson_response,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.utils import get_exception_traceback
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global constants
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))
WAIT_WEIGHTS_READY_TIMEOUT = int(os.getenv("SGLANG_WAIT_WEIGHTS_READY_TIMEOUT", 120))


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: Union[TokenizerManager, MultiTokenizerRouter, TokenizerWorker]
    template_manager: TemplateManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


def get_global_state() -> _GlobalState:
    return _global_state


async def init_multi_tokenizer() -> ServerArgs:
    """
    Initialization function for multi-process tokenizer mode.
    It read args information from shm and inits tokenizer manager for current process.
    """

    # Read configuration from shared memory
    main_pid = get_main_process_id()
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"
    )
    server_args: ServerArgs
    port_args: PortArgs

    # API key authentication is not supported in multi-tokenizer mode
    assert server_args.api_key is None, (
        "API key is not supported in multi-tokenizer mode"
    )

    # Create a new ipc name for the current process
    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    logger.info(
        f"Start multi-tokenizer worker process {os.getpid()}, "
        f"ipc_name={port_args.tokenizer_ipc_name}"
    )

    # Launch multi-tokenizer manager process
    tokenizer_worker_class = get_tokenizer_worker_class(server_args)
    tokenizer_manager = tokenizer_worker_class(server_args, port_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    return server_args


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    grpc_handle = None
    sidecar = None
    warmup_thread = None
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    else:
        # Initialize multi-tokenizer support for worker processes
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Init tracing
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill" + thread_label
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode" + thread_label
        trace_set_thread_info(thread_label)

    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_classify = OpenAIServingClassify(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_tokenize = OpenAIServingTokenize(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_detokenize = OpenAIServingDetokenize(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_transcription = OpenAIServingTranscription(
        _global_state.tokenizer_manager
    )

    # Initialize Ollama-compatible serving handler
    fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)

    # Initialize Anthropic-compatible serving handler
    fast_api_app.state.anthropic_serving = AnthropicServing(
        fast_api_app.state.openai_serving_chat
    )

    # Launch tool server
    tool_server = None
    if server_args.tool_server == "demo":
        from sglang.srt.entrypoints.openai.tool_server import DemoToolServer

        tool_server = DemoToolServer()
    elif server_args.tool_server:
        from sglang.srt.entrypoints.openai.tool_server import MCPToolServer

        tool_server = MCPToolServer()
        await tool_server.add_tool_server(server_args.tool_server)
    elif envs.EXA_API_KEY.get():
        from sglang.srt.entrypoints.openai.tool_server import NativeToolServer

        tool_server = NativeToolServer()

    try:
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        fast_api_app.state.openai_serving_responses = OpenAIServingResponses(
            _global_state.tokenizer_manager,
            _global_state.template_manager,
            enable_prompt_tokens_details=True,
            tool_server=tool_server,
        )
    except Exception as e:
        # Optional endpoint; a load failure (e.g. the gpt-oss harmony vocab
        # download) must not look like a fatal error. One-line WARNING, full
        # traceback at DEBUG.
        logger.warning(
            f"OpenAI Responses API (/v1/responses) disabled: "
            f"OpenAIServingResponses init failed ({type(e).__name__}: {e})"
        )
        logger.debug(
            f"OpenAIServingResponses init traceback:\n{get_exception_traceback()}"
        )

    # Execute custom warmups
    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,
            server_args.warmups.split(","),
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")

    # Start the native gRPC server and warmup inside the try so a failure in
    # either still runs the finally cleanup below. Native gRPC is enabled via
    # --grpc-port / SGLANG_GRPC_PORT; only the single-tokenizer process is
    # gRPC-capable (__post_init__ rejects --tokenizer-worker-num > 1).
    try:
        if (
            getattr(fast_api_app, "is_single_tokenizer_mode", False)
            and server_args.grpc_port is not None
            and not (server_args.smg_grpc_mode or server_args.grpc_mode)
        ):
            grpc_handle = _start_native_grpc_server_for_runtime(
                server_args=server_args,
                tokenizer_manager=_global_state.tokenizer_manager,
                template_manager=_global_state.template_manager,
                scheduler_info=_global_state.scheduler_info,
            )
            if server_args.sidecar is not None:
                from sglang.srt.entrypoints.sidecar import start_sidecar

                sidecar = start_sidecar(server_args)

        # Execute the general warmup
        warmup_thread = threading.Thread(
            target=_wait_and_warmup,
            kwargs=warmup_thread_kwargs,
        )
        warmup_thread.start()

        # Start the HTTP server
        yield
    finally:
        if sidecar is not None:
            try:
                sidecar.stop()
            except Exception:
                logger.exception("Failed to stop sidecar")
        _shutdown_native_grpc_server(grpc_handle)
        if tool_server is not None and hasattr(tool_server, "aclose"):
            await tool_server.aclose()
        if warmup_thread is not None:
            warmup_thread.join()


# Fast API
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if envs.SGLANG_ENABLE_REQUEST_DECOMPRESSION.get():
    from sglang.srt.entrypoints.http_request_decompression import (
        RequestDecompressionMiddleware,
    )

    app.add_middleware(RequestDecompressionMiddleware)

# Include routers
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router

app.include_router(v1_loads_router)

from sglang.srt.entrypoints.elastic_ep import router as elastic_ep_router

app.include_router(elastic_ep_router)


def _anthropic_validation_message(raw_errors) -> str:
    """Render Pydantic-style errors for an Anthropic /v1/messages route.

    Builds a short ``loc: msg`` digest that names the offending fields without
    leaking file paths or Python internals (the default ``str(exc)`` includes
    the dispatcher's ``File "/.../http_server.py"`` line).
    """
    parts: list[str] = []
    for err in raw_errors or []:
        loc = err.get("loc") or ()
        if loc:
            loc_str = ".".join(str(p) for p in loc if p not in ("body",))
        else:
            loc_str = ""
        msg = (err.get("msg") or "").strip()
        if loc_str and msg:
            parts.append(f"{loc_str}: {msg}")
        elif msg:
            parts.append(msg)
    text = "; ".join(parts) or "Invalid request"
    if len(text) > 500:
        text = text[:500] + "…"
    return text


def _anthropic_error_response(*, status_code: int, error_type: str, message: str):
    """Anthropic-format error envelope: {"type":"error","error":{"type":...,"message":...}}."""
    return ORJSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )


@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: HTTPException):
    """Enrich HTTP exception with status code and other details.

    For /v1/responses, emit OpenAI-style nested error envelope:
    {"error": {"message": "...", "type": "...", "param": null, "code": <status>}}
    For /v1/messages, emit Anthropic-style envelope so SDK clients can parse it.
    """
    if request.url.path.startswith("/v1/messages"):
        # Map HTTP status to Anthropic error.type; fall back to api_error.
        anthropic_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            413: "request_too_large",
            422: "invalid_request_error",
            429: "rate_limit_error",
            500: "api_error",
            502: "api_error",
            503: "overloaded_error",
            504: "api_error",
        }.get(exc.status_code, "api_error")
        # 5xx must never echo upstream detail (may contain stack/PII).
        message = (
            "Internal server error"
            if exc.status_code >= 500
            else (str(exc.detail) if exc.detail else "Request failed")
        )
        return _anthropic_error_response(
            status_code=exc.status_code,
            error_type=anthropic_type,
            message=message,
        )

    # adjust fmt for responses api
    if request.url.path.startswith("/v1/responses"):
        nested_error = {
            "message": exc.detail,
            "type": HTTPStatus(exc.status_code).phrase,
            "param": None,
            "code": exc.status_code,
        }
        return ORJSONResponse(
            content={"error": nested_error}, status_code=exc.status_code
        )

    error = ErrorResponse(
        object="error",
        message=exc.detail,
        type=str(exc.status_code),
        code=exc.status_code,
    )
    return ORJSONResponse(content=error.model_dump(), status_code=exc.status_code)


# Custom exception handlers to change validation error status codes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override FastAPI's default 422 validation error with 400.

    For /v1/messages, emit Anthropic-style envelope and scrub the message so
    file paths or Python internals from the default ``str(exc)`` representation
    never reach the client. For /v1/responses, keep OpenAI-style. Otherwise
    use the legacy ErrorResponse shape.
    """
    if request.url.path.startswith("/v1/messages"):
        return _anthropic_error_response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            error_type="invalid_request_error",
            message=_anthropic_validation_message(exc.errors()),
        )

    exc_str = str(exc)
    errors_str = str(exc.errors())

    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    if request.url.path.startswith("/v1/responses"):
        # adapt specially, for v1/responses API only (notice the error key is different)
        nested_error = {
            "message": message,
            "type": HTTPStatus.BAD_REQUEST.phrase,
            "param": None,
            "code": HTTPStatus.BAD_REQUEST.value,
        }
        return ORJSONResponse(status_code=400, content={"error": nested_error})

    err = ErrorResponse(
        message=message,
        type=HTTPStatus.BAD_REQUEST.phrase,
        code=HTTPStatus.BAD_REQUEST.value,
    )

    return ORJSONResponse(
        status_code=400,
        content=err.model_dump(),
    )


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


##### Native API endpoints #####


@app.get("/health")
@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """

    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    # uuid keeps rids unique across tokenizer workers (a bare time.time() can
    # collide and crash the shared DetokenizerManager decode_status).
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL.value
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())

    # As long as we receive any response from the detokenizer/scheduler, we consider the server is healthy.
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.server_status = ServerStatus.Up
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
    return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information (deprecated - use /model_info instead)."""
    logger.warning(
        "Endpoint '/get_model_info' is deprecated and will be removed in a future version. "
        "Please use '/model_info' instead."
    )
    return await model_info()


@app.get("/model_info")
async def model_info():
    """Get the model information."""
    model_config = _global_state.tokenizer_manager.model_config
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": _global_state.tokenizer_manager.is_generation,
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
        "has_image_understanding": model_config.is_image_understandable_model,
        "has_audio_understanding": model_config.is_audio_understandable_model,
        "model_type": getattr(model_config.hf_config, "model_type", None),
        "architectures": getattr(model_config.hf_config, "architectures", None),
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
        # "hf_config": model_config.hf_config.to_dict(),
    }
    embedding_model_spec = getattr(model_config, "embedding_model_spec", None)
    if embedding_model_spec is not None:
        result["embedding"] = resolved_embedding_plan(
            embedding_model_spec,
            server_args=_global_state.tokenizer_manager.server_args,
            model_config=model_config,
        )
    return result


@app.get("/get_weight_version")
@app.get("/weight_version")
async def weight_version():
    """Get the current weight version."""
    raise HTTPException(
        status_code=404,
        detail="Endpoint '/get_weight_version' or '/weight_version' is deprecated. Please use '/model_info' instead.",
    )


@app.get("/get_server_info")
async def get_server_info():
    """Get the server information (deprecated - use /server_info instead)."""
    logger.warning(
        "Endpoint '/get_server_info' is deprecated and will be removed in a future version. "
        "Please use '/server_info' instead."
    )
    return await server_info()


def _summarize_dsv4_sm86_small_batch_gemm(
    internal_states: List[Dict[Any, Any]],
    configured: bool,
    expected_worker_count: int,
) -> Dict[str, Any]:
    """Aggregate worker-local patch/selection proof for `/server_info`."""

    gathered_worker_telemetry: List[Dict[str, Any]] = []
    for state in internal_states:
        if not isinstance(state, dict):
            continue
        gathered = state.get("dsv4_sm86_small_batch_gemm_workers")
        if isinstance(gathered, list):
            gathered_worker_telemetry.extend(
                dict(telemetry) for telemetry in gathered if isinstance(telemetry, dict)
            )
            continue
        # Compatibility with a single-rank runtime, or a worker predating the
        # TP all-gather field. It cannot prove TP2 by itself because the
        # topology-derived expected count below remains two.
        telemetry = state.get("dsv4_sm86_small_batch_gemm")
        if isinstance(telemetry, dict):
            gathered_worker_telemetry.append(dict(telemetry))

    telemetry_by_rank = {
        (
            telemetry.get("dp_rank"),
            telemetry.get("pp_rank"),
            telemetry.get("tp_rank"),
        ): telemetry
        for telemetry in gathered_worker_telemetry
    }
    worker_telemetry = list(telemetry_by_rank.values())

    def rank_sort_key(telemetry: Dict[str, Any]) -> tuple[int, int, int]:
        return (
            -1 if telemetry.get("dp_rank") is None else int(telemetry["dp_rank"]),
            -1 if telemetry.get("pp_rank") is None else int(telemetry["pp_rank"]),
            -1 if telemetry.get("tp_rank") is None else int(telemetry["tp_rank"]),
        )

    worker_telemetry.sort(key=rank_sort_key)
    active_worker_count = sum(
        telemetry.get("patch_state") == "installed"
        and telemetry.get("patch_installed") is True
        and int(telemetry.get("selection_count", 0)) > 0
        for telemetry in worker_telemetry
    )
    return {
        "dsv4_sm86_small_batch_gemm_worker_telemetry": worker_telemetry,
        "dsv4_sm86_small_batch_gemm_expected_worker_count": expected_worker_count,
        "dsv4_sm86_small_batch_gemm_reporting_worker_count": len(worker_telemetry),
        "dsv4_sm86_small_batch_gemm_active_worker_count": active_worker_count,
        "dsv4_sm86_small_batch_gemm_all_workers_active": (
            configured
            and expected_worker_count > 0
            and len(worker_telemetry) == expected_worker_count
            and active_worker_count == expected_worker_count
        ),
    }


_KT_TASK_QUEUE_AFFINITY_WORKER_KEYS = frozenset(
    {
        "pid",
        "gpu_id",
        "tp_rank",
        "pp_rank",
        "dp_rank",
        "moe_ep_rank",
        "moe_dp_rank",
        "telemetry",
        "validation_error",
    }
)
_KT_TASK_QUEUE_AFFINITY_NATIVE_KEYS = frozenset(
    {
        "environment_enabled",
        "eligible_single_numa_subpool",
        "requested",
        "active",
        "status",
        "numa_id",
        "cpu_id",
        "native_thread_id",
    }
)
_KT_TASK_QUEUE_AFFINITY_TELEMETRY_KEYS = frozenset(
    {
        "environment_enabled",
        "eligible_single_numa_subpool",
        "requested",
        "active",
        "status",
        "numa_id",
        "cpu_id",
        "native_thread_id",
        "expected_numa_id",
        "live_cpu_affinity",
        "cpu_in_expected_numa",
        "singleton_instance_count",
        "registered_configuration_count",
    }
)
_KT_SINGLE_NUMA_INLINE_DISPATCH_TELEMETRY_KEYS = frozenset(
    {
        "environment_enabled",
        "expected_numa_id",
        "required_worker_count",
        "singleton_instance_count",
        "registered_configuration_count",
        "dispatch_count_before_startup_probe",
        "dispatch_count_after_startup_probe",
        "startup_probe_advanced_dispatch_count",
        "task_queue_affinity",
        "single_numa_inline_dispatch",
        "worker_pool_affinity",
        "startup_probe",
        "task_queue_live_cpu_affinity",
        "worker_live_cpu_affinities",
        "all_live_worker_affinities_exact",
        "all_worker_cpus_in_expected_numa",
    }
)
_KT_SINGLE_NUMA_INLINE_NATIVE_KEYS = frozenset(
    {
        "environment_enabled",
        "eligible_single_numa_subpool",
        "requested",
        "active",
        "status",
        "physical_numa_id",
        "configured_worker_count",
        "distributor_worker_count",
        "distributor_thread_elided",
        "dispatch_count",
        "exception_count",
        "last_native_thread_id",
        "last_cpu_id",
        "last_worker_pool_thread_id",
        "task_queue_native_thread_id",
        "task_queue_cpu_id",
        "task_queue_affinity_active",
        "last_dispatch_on_task_queue_thread",
        "last_dispatch_on_task_queue_cpu",
        "logical_worker_zero_proven",
        "collision_free_worker_zero",
    }
)
_KT_WORKER_POOL_AFFINITY_NATIVE_KEYS = frozenset(
    {
        "subpool_count",
        "configured_worker_count",
        "subpools",
        "all_worker_bindings_active",
        "all_worker_cpu_ids_unique",
        "all_workers_on_expected_numa",
    }
)
_KT_WORKER_POOL_SUBPOOL_NATIVE_KEYS = frozenset(
    {
        "logical_subpool_index",
        "physical_numa_id",
        "configured_worker_count",
        "active_worker_count",
        "worker_cpu_ids",
        "worker_native_thread_ids",
        "worker_affinity_statuses",
        "worker_roles",
        "last_caller_native_thread_id",
        "last_caller_cpu_id",
    }
)
_KT_SINGLE_NUMA_INLINE_PROBE_KEYS = frozenset(
    {
        "executed_task_count",
        "worker_native_thread_ids",
        "worker_cpu_ids",
    }
)
_KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT = 56
_KT_MXFP4_AVX_SCALE_FOLD_NATIVE_KEYS = frozenset(
    {
        "schema_version",
        "requested_mode",
        "configuration_valid",
        "architecture_supported",
        "execution_mode",
        "n_block",
        "fold_safe_minimum",
        "fold_safe_maximum",
        "lut_identity",
        "lut_hash_algorithm",
        "lut_hash",
        "lut_bytes",
        "buffers_constructed",
        "buffers_finalized",
        "buffers_admitted",
        "buffers_rejected",
        "whole_buffer_domain_finalized",
        "whole_buffer_domain_admitted",
        "scale_bytes_audited",
        "unsafe_scale_bytes",
        "nan_scale_bytes",
        "invalid_mode_requests",
        "observed_scale_minimum",
        "observed_scale_maximum",
        "decode_dispatch_count",
        "prefill_dispatch_count",
        "real_dispatch_count",
        "scale_fold_dispatch_count",
        "lut_decode_dispatch_count",
        "lut_prefill_dispatch_count",
        "exponent_decode_dispatch_count",
        "exponent_prefill_dispatch_count",
        "fallback_dispatch_count",
        "fallback_decode_dispatch_count",
        "fallback_prefill_dispatch_count",
        "zero_invalid_or_fallback_counts",
    }
)
_KT_MXFP4_AVX_SCALE_FOLD_MODES = frozenset({"lut-v1", "exponent-v1"})
_KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK = 128
_KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MINIMUM = 118
_KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MAXIMUM = 126


def _kt_qualified_two_worker_topology(server_args: Any) -> Optional[str]:
    """Name the qualified local two-worker topology, or fail closed."""

    topology = (
        int(server_args.tp_size),
        int(server_args.ep_size),
        int(server_args.pp_size),
        int(server_args.dp_size),
        int(getattr(server_args, "moe_dp_size", 1)),
    )
    if topology == (2, 2, 1, 1, 1):
        return "tp2-ep2"
    if topology == (1, 1, 2, 1, 1):
        return "pp2-ep1"
    return None


def _kt_expected_two_worker_identities(
    topology: Optional[str],
) -> set[tuple[int, int, int, int, int]]:
    """Return (dp, pp, tp, moe_ep, gpu) identities for one topology."""

    if topology == "tp2-ep2":
        return {(0, 0, 0, 0, 0), (0, 0, 1, 1, 1)}
    if topology == "pp2-ep1":
        return {(0, 0, 0, 0, 0), (0, 1, 0, 0, 1)}
    return set()


def _kt_normalized_worker_identity(
    record: Dict[str, Any],
) -> Optional[tuple[int, int, int, int, int]]:
    dp_rank = record.get("dp_rank")
    if dp_rank is None:
        dp_rank = 0
    values = (
        dp_rank,
        record.get("pp_rank"),
        record.get("tp_rank"),
        record.get("moe_ep_rank"),
        record.get("gpu_id"),
    )
    if any(type(value) is not int for value in values):
        return None
    return (values[0], values[1], values[2], values[3], values[4])


def _kt_mxfp4_avx_scale_fold_telemetry_is_admitted(
    telemetry: Any,
    requested_mode: str,
) -> bool:
    """Validate one immutable native scale-buffer snapshot fail closed."""

    if (
        requested_mode not in _KT_MXFP4_AVX_SCALE_FOLD_MODES
        or not isinstance(telemetry, dict)
        or set(telemetry) != _KT_MXFP4_AVX_SCALE_FOLD_NATIVE_KEYS
    ):
        return False
    boolean_keys = (
        "configuration_valid",
        "architecture_supported",
        "whole_buffer_domain_finalized",
        "whole_buffer_domain_admitted",
        "zero_invalid_or_fallback_counts",
    )
    integer_keys = (
        "schema_version",
        "n_block",
        "fold_safe_minimum",
        "fold_safe_maximum",
        "lut_bytes",
        "buffers_constructed",
        "buffers_finalized",
        "buffers_admitted",
        "buffers_rejected",
        "scale_bytes_audited",
        "unsafe_scale_bytes",
        "nan_scale_bytes",
        "invalid_mode_requests",
        "observed_scale_minimum",
        "observed_scale_maximum",
        "decode_dispatch_count",
        "prefill_dispatch_count",
        "real_dispatch_count",
        "scale_fold_dispatch_count",
        "lut_decode_dispatch_count",
        "lut_prefill_dispatch_count",
        "exponent_decode_dispatch_count",
        "exponent_prefill_dispatch_count",
        "fallback_dispatch_count",
        "fallback_decode_dispatch_count",
        "fallback_prefill_dispatch_count",
    )
    string_keys = (
        "requested_mode",
        "execution_mode",
        "lut_identity",
        "lut_hash_algorithm",
        "lut_hash",
    )
    if (
        any(type(telemetry[key]) is not bool for key in boolean_keys)
        or any(type(telemetry[key]) is not int for key in integer_keys)
        or any(type(telemetry[key]) is not str for key in string_keys)
        or min(telemetry[key] for key in integer_keys) < 0
    ):
        return False
    exact_contract = {
        "schema_version": 1,
        "requested_mode": requested_mode,
        "configuration_valid": True,
        "architecture_supported": True,
        "n_block": _KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK,
        "fold_safe_minimum": 2,
        "fold_safe_maximum": 252,
        "lut_identity": "mxfp4-e2m1-bf16-ue8m0-lut-v1",
        "lut_hash_algorithm": "fnv1a64-le",
        "lut_hash": "06d1a83dbf20f545",
        "lut_bytes": 16_384,
        "whole_buffer_domain_finalized": True,
        "whole_buffer_domain_admitted": True,
        "buffers_rejected": 0,
        "unsafe_scale_bytes": 0,
        "nan_scale_bytes": 0,
        "invalid_mode_requests": 0,
        "fallback_dispatch_count": 0,
        "fallback_decode_dispatch_count": 0,
        "fallback_prefill_dispatch_count": 0,
        "zero_invalid_or_fallback_counts": True,
    }
    if any(telemetry[key] != expected for key, expected in exact_contract.items()):
        return False
    constructed = telemetry["buffers_constructed"]
    if (
        constructed <= 0
        or telemetry["buffers_finalized"] != constructed
        or telemetry["buffers_admitted"] != constructed
        or telemetry["scale_bytes_audited"] <= 0
        or not (
            _KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MINIMUM
            <= telemetry["observed_scale_minimum"]
            <= telemetry["observed_scale_maximum"]
            <= _KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MAXIMUM
        )
    ):
        return False

    decode_dispatches = telemetry["decode_dispatch_count"]
    prefill_dispatches = telemetry["prefill_dispatch_count"]
    real_dispatches = telemetry["real_dispatch_count"]
    selected_prefix = "lut" if requested_mode == "lut-v1" else "exponent"
    other_prefix = "exponent" if selected_prefix == "lut" else "lut"
    expected_execution_mode = "not-executed" if real_dispatches == 0 else requested_mode
    return (
        real_dispatches == decode_dispatches + prefill_dispatches
        and telemetry[f"{selected_prefix}_decode_dispatch_count"] == decode_dispatches
        and telemetry[f"{selected_prefix}_prefill_dispatch_count"] == prefill_dispatches
        and telemetry[f"{other_prefix}_decode_dispatch_count"] == 0
        and telemetry[f"{other_prefix}_prefill_dispatch_count"] == 0
        and telemetry["scale_fold_dispatch_count"] == real_dispatches
        and telemetry["execution_mode"] == expected_execution_mode
    )


def _kt_mxfp4_avx_scale_fold_record_is_active(
    record: Dict[str, Any],
    requested_mode: str,
    server_args: Any,
) -> bool:
    """Validate one rank-tagged admission for qualified EP2 or PP2."""

    if set(record) != _KT_TASK_QUEUE_AFFINITY_WORKER_KEYS:
        return False
    topology = _kt_qualified_two_worker_topology(server_args)
    expected_identities = _kt_expected_two_worker_identities(topology)
    if not expected_identities:
        return False
    if any(
        type(record.get(key)) is not int
        for key in ("pid", "gpu_id", "tp_rank", "pp_rank", "moe_ep_rank")
    ):
        return False
    if (
        record["pid"] <= 0
        or _kt_normalized_worker_identity(record) not in expected_identities
        or record["dp_rank"] not in (None, 0)
        or record["moe_dp_rank"] not in (None, 0)
        or record["validation_error"] is not None
    ):
        return False
    return _kt_mxfp4_avx_scale_fold_telemetry_is_admitted(
        record["telemetry"], requested_mode
    )


def _expected_kt_task_queue_numa_id(
    record: Dict[str, Any], server_args: Any
) -> Optional[int]:
    """Resolve the rank-local NUMA node using the KT placement contract."""

    numa_nodes = getattr(server_args, "kt_numa_nodes", None)
    if not isinstance(numa_nodes, list) or any(
        type(numa_id) is not int or numa_id < 0 for numa_id in numa_nodes
    ):
        return None

    ep_size = int(server_args.ep_size)
    pp_size = int(server_args.pp_size)
    tp_size = int(server_args.tp_size)
    if ep_size > 1:
        moe_ep_rank = record.get("moe_ep_rank")
        if (
            len(numa_nodes) != ep_size
            or type(moe_ep_rank) is not int
            or not 0 <= moe_ep_rank < ep_size
        ):
            return None
        return numa_nodes[moe_ep_rank]
    if pp_size > 1 and tp_size == 1:
        pp_rank = record.get("pp_rank")
        if (
            len(numa_nodes) != pp_size
            or type(pp_rank) is not int
            or not 0 <= pp_rank < pp_size
        ):
            return None
        return numa_nodes[pp_rank]
    if len(numa_nodes) != 1:
        return None
    return numa_nodes[0]


def _kt_task_queue_affinity_record_is_active(
    record: Dict[str, Any], server_args: Any
) -> bool:
    """Validate one serialized native readback without trusting truthiness."""

    if set(record) != _KT_TASK_QUEUE_AFFINITY_WORKER_KEYS:
        return False
    integer_rank_fields = ("pid", "gpu_id", "tp_rank", "pp_rank", "moe_ep_rank")
    if any(type(record.get(key)) is not int for key in integer_rank_fields):
        return False
    if record["pid"] <= 0 or record["gpu_id"] < 0:
        return False
    if not 0 <= record["tp_rank"] < int(server_args.tp_size):
        return False
    if not 0 <= record["pp_rank"] < int(server_args.pp_size):
        return False
    if not 0 <= record["moe_ep_rank"] < int(server_args.ep_size):
        return False
    dp_rank = record["dp_rank"]
    if int(server_args.dp_size) == 1:
        if dp_rank is not None and (type(dp_rank) is not int or dp_rank != 0):
            return False
    elif type(dp_rank) is not int or not 0 <= dp_rank < int(server_args.dp_size):
        return False
    moe_dp_rank = record["moe_dp_rank"]
    if moe_dp_rank is not None and (
        type(moe_dp_rank) is not int
        or not 0 <= moe_dp_rank < int(server_args.moe_dp_size)
    ):
        return False
    if record["validation_error"] is not None:
        return False

    telemetry = record["telemetry"]
    if not isinstance(telemetry, dict):
        return False
    if set(telemetry) != _KT_TASK_QUEUE_AFFINITY_TELEMETRY_KEYS:
        return False
    for key in (
        "environment_enabled",
        "eligible_single_numa_subpool",
        "requested",
        "active",
        "cpu_in_expected_numa",
    ):
        if type(telemetry[key]) is not bool:
            return False
    for key in (
        "numa_id",
        "cpu_id",
        "native_thread_id",
        "expected_numa_id",
        "singleton_instance_count",
        "registered_configuration_count",
    ):
        if type(telemetry[key]) is not int:
            return False
    if type(telemetry["status"]) is not str:
        return False
    if (
        telemetry["environment_enabled"] is not True
        or telemetry["eligible_single_numa_subpool"] is not True
        or telemetry["requested"] is not True
        or telemetry["active"] is not True
        or telemetry["status"] != "active"
        or telemetry["cpu_in_expected_numa"] is not True
        or telemetry["singleton_instance_count"] != 1
        or telemetry["registered_configuration_count"] != 1
        or telemetry["cpu_id"] < 0
        or telemetry["native_thread_id"] <= 0
    ):
        return False
    expected_numa_id = _expected_kt_task_queue_numa_id(record, server_args)
    if (
        expected_numa_id is None
        or telemetry["expected_numa_id"] != expected_numa_id
        or telemetry["numa_id"] != expected_numa_id
    ):
        return False
    live_cpu_affinity = telemetry["live_cpu_affinity"]
    return (
        isinstance(live_cpu_affinity, list)
        and len(live_cpu_affinity) == 1
        and type(live_cpu_affinity[0]) is int
        and live_cpu_affinity[0] == telemetry["cpu_id"]
    )


def _kt_single_numa_inline_dispatch_record_is_active(
    record: Dict[str, Any], server_args: Any
) -> bool:
    """Validate one complete 56-worker proof for qualified EP2 or PP2."""

    if set(record) != _KT_TASK_QUEUE_AFFINITY_WORKER_KEYS:
        return False
    topology = _kt_qualified_two_worker_topology(server_args)
    expected_identities = _kt_expected_two_worker_identities(topology)
    if not expected_identities:
        return False
    integer_rank_fields = ("pid", "gpu_id", "tp_rank", "pp_rank", "moe_ep_rank")
    if any(type(record.get(key)) is not int for key in integer_rank_fields):
        return False
    if (
        record["pid"] <= 0
        or record["gpu_id"] < 0
        or _kt_normalized_worker_identity(record) not in expected_identities
    ):
        return False
    if record["dp_rank"] not in (None, 0) or record["moe_dp_rank"] not in (None, 0):
        return False
    if record["validation_error"] is not None:
        return False

    telemetry = record["telemetry"]
    if (
        not isinstance(telemetry, dict)
        or set(telemetry) != _KT_SINGLE_NUMA_INLINE_DISPATCH_TELEMETRY_KEYS
    ):
        return False
    boolean_telemetry_keys = (
        "environment_enabled",
        "startup_probe_advanced_dispatch_count",
        "all_live_worker_affinities_exact",
        "all_worker_cpus_in_expected_numa",
    )
    integer_telemetry_keys = (
        "expected_numa_id",
        "required_worker_count",
        "singleton_instance_count",
        "registered_configuration_count",
        "dispatch_count_before_startup_probe",
        "dispatch_count_after_startup_probe",
    )
    if any(type(telemetry[key]) is not bool for key in boolean_telemetry_keys):
        return False
    if any(type(telemetry[key]) is not int for key in integer_telemetry_keys):
        return False
    expected_numa_id = _expected_kt_task_queue_numa_id(record, server_args)
    before_dispatch_count = telemetry["dispatch_count_before_startup_probe"]
    after_dispatch_count = telemetry["dispatch_count_after_startup_probe"]
    if (
        telemetry["environment_enabled"] is not True
        or expected_numa_id is None
        or telemetry["expected_numa_id"] != expected_numa_id
        or telemetry["required_worker_count"]
        != _KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT
        or telemetry["singleton_instance_count"] != 1
        or telemetry["registered_configuration_count"] != 1
        or before_dispatch_count < 0
        or after_dispatch_count <= before_dispatch_count
        or telemetry["startup_probe_advanced_dispatch_count"] is not True
        or telemetry["all_live_worker_affinities_exact"] is not True
        or telemetry["all_worker_cpus_in_expected_numa"] is not True
    ):
        return False

    task_queue = telemetry["task_queue_affinity"]
    if (
        not isinstance(task_queue, dict)
        or set(task_queue) != _KT_TASK_QUEUE_AFFINITY_NATIVE_KEYS
    ):
        return False
    for key in (
        "environment_enabled",
        "eligible_single_numa_subpool",
        "requested",
        "active",
    ):
        if type(task_queue[key]) is not bool:
            return False
    for key in ("numa_id", "cpu_id", "native_thread_id"):
        if type(task_queue[key]) is not int:
            return False
    if type(task_queue["status"]) is not str:
        return False
    if (
        task_queue["environment_enabled"] is not True
        or task_queue["eligible_single_numa_subpool"] is not True
        or task_queue["requested"] is not True
        or task_queue["active"] is not True
        or task_queue["status"] != "active"
        or task_queue["numa_id"] != expected_numa_id
        or task_queue["cpu_id"] < 0
        or task_queue["native_thread_id"] <= 0
    ):
        return False
    task_queue_cpu_id = task_queue["cpu_id"]
    task_queue_native_thread_id = task_queue["native_thread_id"]

    inline = telemetry["single_numa_inline_dispatch"]
    if (
        not isinstance(inline, dict)
        or set(inline) != _KT_SINGLE_NUMA_INLINE_NATIVE_KEYS
    ):
        return False
    inline_boolean_keys = (
        "environment_enabled",
        "eligible_single_numa_subpool",
        "requested",
        "active",
        "distributor_thread_elided",
        "task_queue_affinity_active",
        "last_dispatch_on_task_queue_thread",
        "last_dispatch_on_task_queue_cpu",
        "logical_worker_zero_proven",
        "collision_free_worker_zero",
    )
    inline_integer_keys = (
        "physical_numa_id",
        "configured_worker_count",
        "distributor_worker_count",
        "dispatch_count",
        "exception_count",
        "last_native_thread_id",
        "last_cpu_id",
        "last_worker_pool_thread_id",
        "task_queue_native_thread_id",
        "task_queue_cpu_id",
    )
    if any(type(inline[key]) is not bool for key in inline_boolean_keys):
        return False
    if any(type(inline[key]) is not int for key in inline_integer_keys):
        return False
    if type(inline["status"]) is not str:
        return False
    required_inline_values = {
        "environment_enabled": True,
        "eligible_single_numa_subpool": True,
        "requested": True,
        "active": True,
        "status": "active",
        "physical_numa_id": expected_numa_id,
        "configured_worker_count": _KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT,
        "distributor_worker_count": 0,
        "distributor_thread_elided": True,
        "dispatch_count": after_dispatch_count,
        "exception_count": 0,
        "last_native_thread_id": task_queue_native_thread_id,
        "last_cpu_id": task_queue_cpu_id,
        "last_worker_pool_thread_id": 0,
        "task_queue_native_thread_id": task_queue_native_thread_id,
        "task_queue_cpu_id": task_queue_cpu_id,
        "task_queue_affinity_active": True,
        "last_dispatch_on_task_queue_thread": True,
        "last_dispatch_on_task_queue_cpu": True,
        "logical_worker_zero_proven": True,
        "collision_free_worker_zero": True,
    }
    if any(inline[key] != expected for key, expected in required_inline_values.items()):
        return False
    if inline["dispatch_count"] <= 0:
        return False

    worker_pool = telemetry["worker_pool_affinity"]
    if (
        not isinstance(worker_pool, dict)
        or set(worker_pool) != _KT_WORKER_POOL_AFFINITY_NATIVE_KEYS
    ):
        return False
    for key in ("subpool_count", "configured_worker_count"):
        if type(worker_pool[key]) is not int:
            return False
    for key in (
        "all_worker_bindings_active",
        "all_worker_cpu_ids_unique",
        "all_workers_on_expected_numa",
    ):
        if type(worker_pool[key]) is not bool:
            return False
    if (
        worker_pool["subpool_count"] != 1
        or worker_pool["configured_worker_count"]
        != _KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT
        or worker_pool["all_worker_bindings_active"] is not True
        or worker_pool["all_worker_cpu_ids_unique"] is not True
        or worker_pool["all_workers_on_expected_numa"] is not True
    ):
        return False
    subpools = worker_pool["subpools"]
    if not isinstance(subpools, list) or len(subpools) != 1:
        return False
    subpool = subpools[0]
    if (
        not isinstance(subpool, dict)
        or set(subpool) != _KT_WORKER_POOL_SUBPOOL_NATIVE_KEYS
    ):
        return False
    for key in (
        "logical_subpool_index",
        "physical_numa_id",
        "configured_worker_count",
        "active_worker_count",
        "last_caller_native_thread_id",
        "last_caller_cpu_id",
    ):
        if type(subpool[key]) is not int:
            return False
    required_subpool_values = {
        "logical_subpool_index": 0,
        "physical_numa_id": expected_numa_id,
        "configured_worker_count": _KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT,
        "active_worker_count": _KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT,
        "last_caller_native_thread_id": task_queue_native_thread_id,
        "last_caller_cpu_id": task_queue_cpu_id,
    }
    if any(
        subpool[key] != expected for key, expected in required_subpool_values.items()
    ):
        return False
    worker_cpu_ids = subpool["worker_cpu_ids"]
    worker_native_thread_ids = subpool["worker_native_thread_ids"]
    worker_affinity_statuses = subpool["worker_affinity_statuses"]
    worker_roles = subpool["worker_roles"]
    required_worker_count = _KT_SINGLE_NUMA_INLINE_REQUIRED_WORKER_COUNT
    if any(
        not isinstance(values, list) or len(values) != required_worker_count
        for values in (
            worker_cpu_ids,
            worker_native_thread_ids,
            worker_affinity_statuses,
            worker_roles,
        )
    ):
        return False
    if (
        any(type(cpu_id) is not int or cpu_id < 0 for cpu_id in worker_cpu_ids)
        or any(
            type(native_thread_id) is not int or native_thread_id <= 0
            for native_thread_id in worker_native_thread_ids
        )
        or len(set(worker_cpu_ids)) != required_worker_count
        or len(set(worker_native_thread_ids)) != required_worker_count
        or worker_affinity_statuses != ["active"] * required_worker_count
        or worker_roles
        != ["inline_task_queue_worker0"]
        + ["background_worker"] * (required_worker_count - 1)
        or worker_cpu_ids[0] != task_queue_cpu_id
        or worker_native_thread_ids[0] != task_queue_native_thread_id
    ):
        return False

    startup_probe = telemetry["startup_probe"]
    if (
        not isinstance(startup_probe, dict)
        or set(startup_probe) != _KT_SINGLE_NUMA_INLINE_PROBE_KEYS
        or type(startup_probe["executed_task_count"]) is not int
        or startup_probe["executed_task_count"] != required_worker_count
        or startup_probe["worker_native_thread_ids"] != worker_native_thread_ids
        or startup_probe["worker_cpu_ids"] != worker_cpu_ids
    ):
        return False

    task_queue_live_cpu_affinity = telemetry["task_queue_live_cpu_affinity"]
    worker_live_cpu_affinities = telemetry["worker_live_cpu_affinities"]
    if task_queue_live_cpu_affinity != [task_queue_cpu_id]:
        return False
    if (
        not isinstance(worker_live_cpu_affinities, list)
        or len(worker_live_cpu_affinities) != required_worker_count
    ):
        return False
    return all(
        isinstance(live_affinity, list) and live_affinity == [cpu_id]
        for live_affinity, cpu_id in zip(
            worker_live_cpu_affinities,
            worker_cpu_ids,
            strict=True,
        )
    )


def _summarize_kt_task_queue_affinity(
    internal_states: List[Dict[Any, Any]],
    configured: bool,
    server_args: Any,
) -> Dict[str, Any]:
    """Aggregate CPUInfer affinity proof across local TP/EP/PP/DP workers."""

    expected_worker_count = (
        int(server_args.tp_size) * int(server_args.pp_size) * int(server_args.dp_size)
    )
    if not configured:
        return {
            "kt_task_queue_affinity_worker_telemetry": [],
            "kt_task_queue_affinity_expected_worker_count": expected_worker_count,
            "kt_task_queue_affinity_reporting_worker_count": 0,
            "kt_task_queue_affinity_active_worker_count": 0,
            "kt_task_queue_affinity_invalid_worker_count": 0,
            "kt_task_queue_affinity_duplicate_worker_count": 0,
            "kt_task_queue_affinity_rank_coverage_valid": False,
            "kt_task_queue_affinity_all_workers_active": False,
        }

    gathered_records: List[Dict[str, Any]] = []
    malformed_record_count = 0
    for state in internal_states:
        if not isinstance(state, dict):
            malformed_record_count += 1
            continue
        gathered = state.get("kt_task_queue_affinity_workers")
        if isinstance(gathered, list):
            for record in gathered:
                if isinstance(record, dict):
                    gathered_records.append(dict(record))
                else:
                    malformed_record_count += 1
            continue
        local_record = state.get("kt_task_queue_affinity")
        if isinstance(local_record, dict):
            gathered_records.append(dict(local_record))
        else:
            malformed_record_count += 1

    worker_records_by_rank: Dict[tuple[Any, Any, Any], Dict[str, Any]] = {}
    duplicate_worker_count = 0
    for record_index, record in enumerate(gathered_records):
        rank_values = (
            record.get("dp_rank"),
            record.get("pp_rank"),
            record.get("tp_rank"),
        )
        identity = (
            rank_values
            if all(value is None or type(value) is int for value in rank_values)
            else ("malformed", record_index, None)
        )
        if identity in worker_records_by_rank:
            duplicate_worker_count += 1
            continue
        worker_records_by_rank[identity] = record
    worker_records = list(worker_records_by_rank.values())

    def rank_sort_key(record: Dict[str, Any]) -> tuple[int, int, int]:
        def integer_or_negative_one(value: Any) -> int:
            return value if type(value) is int else -1

        return (
            integer_or_negative_one(record.get("dp_rank")),
            integer_or_negative_one(record.get("pp_rank")),
            integer_or_negative_one(record.get("tp_rank")),
        )

    worker_records.sort(key=rank_sort_key)
    active_worker_count = sum(
        _kt_task_queue_affinity_record_is_active(record, server_args)
        for record in worker_records
    )
    rank_fields_are_valid = all(
        (record.get("dp_rank") is None or type(record.get("dp_rank")) is int)
        and type(record.get("pp_rank")) is int
        and type(record.get("tp_rank")) is int
        for record in worker_records
    )
    normalized_rank_coverage = (
        {
            (
                0 if record.get("dp_rank") is None else record.get("dp_rank"),
                record.get("pp_rank"),
                record.get("tp_rank"),
            )
            for record in worker_records
        }
        if rank_fields_are_valid
        else set()
    )
    expected_rank_coverage = {
        (dp_rank, pp_rank, tp_rank)
        for dp_rank in range(int(server_args.dp_size))
        for pp_rank in range(int(server_args.pp_size))
        for tp_rank in range(int(server_args.tp_size))
    }
    rank_coverage_valid = (
        malformed_record_count == 0
        and duplicate_worker_count == 0
        and rank_fields_are_valid
        and normalized_rank_coverage == expected_rank_coverage
    )
    reporting_worker_count = len(worker_records)
    invalid_worker_count = (
        reporting_worker_count
        - active_worker_count
        + malformed_record_count
        + duplicate_worker_count
    )
    return {
        "kt_task_queue_affinity_worker_telemetry": worker_records,
        "kt_task_queue_affinity_expected_worker_count": expected_worker_count,
        "kt_task_queue_affinity_reporting_worker_count": reporting_worker_count,
        "kt_task_queue_affinity_active_worker_count": active_worker_count,
        "kt_task_queue_affinity_invalid_worker_count": invalid_worker_count,
        "kt_task_queue_affinity_duplicate_worker_count": duplicate_worker_count,
        "kt_task_queue_affinity_rank_coverage_valid": rank_coverage_valid,
        "kt_task_queue_affinity_all_workers_active": (
            expected_worker_count > 0
            and reporting_worker_count == expected_worker_count
            and active_worker_count == expected_worker_count
            and rank_coverage_valid
        ),
    }


def _summarize_kt_single_numa_inline_dispatch(
    internal_states: List[Dict[Any, Any]],
    configured: bool,
    server_args: Any,
) -> Dict[str, Any]:
    """Aggregate an exact, fail-closed two-worker inline-dispatch proof."""

    expected_worker_count = 2
    numa_nodes = getattr(server_args, "kt_numa_nodes", None)
    topology = _kt_qualified_two_worker_topology(server_args)
    expected_identities = _kt_expected_two_worker_identities(topology)
    supported_topology_valid = (
        bool(expected_identities)
        and isinstance(numa_nodes, list)
        and len(numa_nodes) == 2
        and all(type(numa_id) is int and numa_id >= 0 for numa_id in numa_nodes)
        and len(set(numa_nodes)) == 2
    )
    empty_summary = {
        "kt_single_numa_inline_dispatch_worker_telemetry": [],
        "kt_single_numa_inline_dispatch_expected_worker_count": (expected_worker_count),
        "kt_single_numa_inline_dispatch_reporting_worker_count": 0,
        "kt_single_numa_inline_dispatch_active_worker_count": 0,
        "kt_single_numa_inline_dispatch_invalid_worker_count": 0,
        "kt_single_numa_inline_dispatch_duplicate_worker_count": 0,
        "kt_single_numa_inline_dispatch_rank_coverage_valid": False,
        "kt_single_numa_inline_dispatch_topology": topology,
        "kt_single_numa_inline_dispatch_supported_topology_valid": (
            supported_topology_valid
        ),
        "kt_single_numa_inline_dispatch_ep2_topology_valid": (
            topology == "tp2-ep2" and supported_topology_valid
        ),
        "kt_single_numa_inline_dispatch_all_workers_active": False,
    }
    if not configured:
        return empty_summary

    gathered_records: List[Dict[str, Any]] = []
    malformed_record_count = 0
    for state in internal_states:
        if not isinstance(state, dict):
            malformed_record_count += 1
            continue
        gathered = state.get("kt_single_numa_inline_dispatch_workers")
        if isinstance(gathered, list):
            for record in gathered:
                if isinstance(record, dict):
                    gathered_records.append(dict(record))
                else:
                    malformed_record_count += 1
            continue
        local_record = state.get("kt_single_numa_inline_dispatch")
        if isinstance(local_record, dict):
            gathered_records.append(dict(local_record))
        else:
            malformed_record_count += 1

    worker_records_by_rank: Dict[tuple[Any, Any, Any], Dict[str, Any]] = {}
    duplicate_worker_count = 0
    for record_index, record in enumerate(gathered_records):
        rank_values = (
            record.get("dp_rank"),
            record.get("pp_rank"),
            record.get("tp_rank"),
        )
        identity = (
            rank_values
            if all(value is None or type(value) is int for value in rank_values)
            else ("malformed", record_index, None)
        )
        if identity in worker_records_by_rank:
            duplicate_worker_count += 1
            continue
        worker_records_by_rank[identity] = record
    worker_records = list(worker_records_by_rank.values())

    def rank_sort_key(record: Dict[str, Any]) -> tuple[int, int, int]:
        def integer_or_negative_one(value: Any) -> int:
            return value if type(value) is int else -1

        return (
            integer_or_negative_one(record.get("dp_rank")),
            integer_or_negative_one(record.get("pp_rank")),
            integer_or_negative_one(record.get("tp_rank")),
        )

    worker_records.sort(key=rank_sort_key)
    active_worker_count = sum(
        _kt_single_numa_inline_dispatch_record_is_active(record, server_args)
        for record in worker_records
    )
    rank_fields_are_valid = all(
        _kt_normalized_worker_identity(record) is not None
        for record in worker_records
    )
    normalized_rank_coverage = (
        {
            _kt_normalized_worker_identity(record)
            for record in worker_records
        }
        if rank_fields_are_valid
        else set()
    )
    rank_coverage_valid = (
        supported_topology_valid
        and malformed_record_count == 0
        and duplicate_worker_count == 0
        and rank_fields_are_valid
        and normalized_rank_coverage == expected_identities
    )
    reporting_worker_count = len(worker_records)
    invalid_worker_count = (
        reporting_worker_count
        - active_worker_count
        + malformed_record_count
        + duplicate_worker_count
    )
    return {
        "kt_single_numa_inline_dispatch_worker_telemetry": worker_records,
        "kt_single_numa_inline_dispatch_expected_worker_count": (expected_worker_count),
        "kt_single_numa_inline_dispatch_reporting_worker_count": (
            reporting_worker_count
        ),
        "kt_single_numa_inline_dispatch_active_worker_count": active_worker_count,
        "kt_single_numa_inline_dispatch_invalid_worker_count": (invalid_worker_count),
        "kt_single_numa_inline_dispatch_duplicate_worker_count": (
            duplicate_worker_count
        ),
        "kt_single_numa_inline_dispatch_rank_coverage_valid": rank_coverage_valid,
        "kt_single_numa_inline_dispatch_topology": topology,
        "kt_single_numa_inline_dispatch_supported_topology_valid": (
            supported_topology_valid
        ),
        "kt_single_numa_inline_dispatch_ep2_topology_valid": (
            topology == "tp2-ep2" and supported_topology_valid
        ),
        "kt_single_numa_inline_dispatch_all_workers_active": (
            configured
            and supported_topology_valid
            and reporting_worker_count == expected_worker_count
            and active_worker_count == expected_worker_count
            and rank_coverage_valid
        ),
    }


def _summarize_kt_mxfp4_avx_scale_fold(
    internal_states: List[Dict[Any, Any]],
    requested_mode: str,
    server_args: Any,
) -> Dict[str, Any]:
    """Aggregate the cached, startup-only MXFP4 scale-buffer admission."""

    configured = requested_mode in _KT_MXFP4_AVX_SCALE_FOLD_MODES
    expected_worker_count = 2
    topology = _kt_qualified_two_worker_topology(server_args)
    expected_identities = _kt_expected_two_worker_identities(topology)
    supported_topology_valid = bool(expected_identities)
    empty_summary = {
        "kt_mxfp4_avx_scale_fold_requested_mode": requested_mode,
        "kt_mxfp4_avx_scale_fold_expected_n_block": (_KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK),
        "kt_mxfp4_avx_scale_fold_worker_telemetry": [],
        "kt_mxfp4_avx_scale_fold_expected_worker_count": expected_worker_count,
        "kt_mxfp4_avx_scale_fold_reporting_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_active_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_invalid_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_duplicate_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_rank_coverage_valid": False,
        "kt_mxfp4_avx_scale_fold_topology": topology,
        "kt_mxfp4_avx_scale_fold_supported_topology_valid": (
            supported_topology_valid
        ),
        "kt_mxfp4_avx_scale_fold_ep2_topology_valid": topology == "tp2-ep2",
        "kt_mxfp4_avx_scale_fold_all_workers_active": False,
    }
    if not configured:
        return empty_summary

    gathered_records: List[Dict[str, Any]] = []
    malformed_record_count = 0
    for state in internal_states:
        if not isinstance(state, dict):
            malformed_record_count += 1
            continue
        gathered = state.get("kt_mxfp4_avx_scale_fold_workers")
        if isinstance(gathered, list):
            for record in gathered:
                if isinstance(record, dict):
                    gathered_records.append(dict(record))
                else:
                    malformed_record_count += 1
            continue
        local_record = state.get("kt_mxfp4_avx_scale_fold")
        if isinstance(local_record, dict):
            gathered_records.append(dict(local_record))
        else:
            malformed_record_count += 1

    worker_records_by_rank: Dict[tuple[Any, Any, Any], Dict[str, Any]] = {}
    duplicate_worker_count = 0
    for record_index, record in enumerate(gathered_records):
        rank_values = (
            record.get("dp_rank"),
            record.get("pp_rank"),
            record.get("tp_rank"),
        )
        identity = (
            rank_values
            if all(value is None or type(value) is int for value in rank_values)
            else ("malformed", record_index, None)
        )
        if identity in worker_records_by_rank:
            duplicate_worker_count += 1
            continue
        worker_records_by_rank[identity] = record
    worker_records = list(worker_records_by_rank.values())

    def rank_sort_key(record: Dict[str, Any]) -> tuple[int, int, int]:
        def integer_or_negative_one(value: Any) -> int:
            return value if type(value) is int else -1

        return (
            integer_or_negative_one(record.get("dp_rank")),
            integer_or_negative_one(record.get("pp_rank")),
            integer_or_negative_one(record.get("tp_rank")),
        )

    worker_records.sort(key=rank_sort_key)
    active_worker_count = sum(
        _kt_mxfp4_avx_scale_fold_record_is_active(record, requested_mode, server_args)
        for record in worker_records
    )
    rank_fields_are_valid = all(
        _kt_normalized_worker_identity(record) is not None
        for record in worker_records
    )
    normalized_rank_coverage = (
        {
            _kt_normalized_worker_identity(record)
            for record in worker_records
        }
        if rank_fields_are_valid
        else set()
    )
    worker_pids = {
        record.get("pid")
        for record in worker_records
        if type(record.get("pid")) is int and record.get("pid", 0) > 0
    }
    rank_coverage_valid = (
        supported_topology_valid
        and malformed_record_count == 0
        and duplicate_worker_count == 0
        and rank_fields_are_valid
        and normalized_rank_coverage == expected_identities
        and len(worker_pids) == expected_worker_count
    )
    reporting_worker_count = len(worker_records)
    invalid_worker_count = (
        reporting_worker_count
        - active_worker_count
        + malformed_record_count
        + duplicate_worker_count
    )
    return {
        **empty_summary,
        "kt_mxfp4_avx_scale_fold_worker_telemetry": worker_records,
        "kt_mxfp4_avx_scale_fold_reporting_worker_count": reporting_worker_count,
        "kt_mxfp4_avx_scale_fold_active_worker_count": active_worker_count,
        "kt_mxfp4_avx_scale_fold_invalid_worker_count": invalid_worker_count,
        "kt_mxfp4_avx_scale_fold_duplicate_worker_count": duplicate_worker_count,
        "kt_mxfp4_avx_scale_fold_rank_coverage_valid": rank_coverage_valid,
        "kt_mxfp4_avx_scale_fold_topology": topology,
        "kt_mxfp4_avx_scale_fold_supported_topology_valid": (
            supported_topology_valid
        ),
        "kt_mxfp4_avx_scale_fold_ep2_topology_valid": topology == "tp2-ep2",
        "kt_mxfp4_avx_scale_fold_all_workers_active": (
            configured
            and supported_topology_valid
            and reporting_worker_count == expected_worker_count
            and active_worker_count == expected_worker_count
            and rank_coverage_valid
        ),
    }


@app.get("/server_info")
async def server_info():
    """Get the server information."""
    # Returns internal states per DP.
    internal_states: List[
        Dict[Any, Any]
    ] = await _global_state.tokenizer_manager.get_internal_state()

    server_args = _global_state.tokenizer_manager.server_args

    hybrid_plan_info: Dict[str, Any] = {}
    hybrid_plan_path = os.environ.get("SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN")
    if hybrid_plan_path:
        real_plan_path = os.path.realpath(hybrid_plan_path)

        def plan_sha256() -> str:
            digest = hashlib.sha256()
            with open(real_plan_path, "rb") as plan_file:
                for chunk in iter(lambda: plan_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        before_sha256 = plan_sha256()
        expected_sha256 = os.environ.get("SGLANG_KT_HYBRID_EXPERT_PLAN_SHA256")
        if expected_sha256 is not None and before_sha256 != expected_sha256:
            raise RuntimeError(
                "live KTransformers hybrid plan no longer matches its "
                "launcher-admitted SHA-256"
            )
        loaded_plan = torch.load(real_plan_path, map_location="cpu", weights_only=True)
        if plan_sha256() != before_sha256:
            raise RuntimeError(
                "live KTransformers hybrid plan changed while reading server info"
            )
        if not isinstance(loaded_plan, dict) or not isinstance(
            (gpu_masks := loaded_plan.get("gpu_experts_mask_by_rank")),
            torch.Tensor,
        ):
            raise RuntimeError("live KTransformers hybrid plan is malformed")
        gpu_counts = gpu_masks.to(device="cpu", dtype=torch.bool).sum(dim=2)
        hybrid_plan_info = {
            "kt_hybrid_expert_plan_format": loaded_plan.get("format"),
            "kt_hybrid_placement_semantics_sha256": loaded_plan.get(
                "placement_semantics_sha256"
            ),
            "kt_hybrid_gpu_rank_counts_by_layer": gpu_counts.tolist(),
            "kt_hybrid_min_gpu_experts_per_rank_per_layer": int(
                gpu_counts.min().item()
            ),
            "kt_hybrid_max_gpu_experts_per_rank_per_layer": int(
                gpu_counts.max().item()
            ),
            "kt_hybrid_total_gpu_expert_layers_by_rank": gpu_counts.sum(dim=1).tolist(),
        }

    draft_hybrid_plan_info: Dict[str, Any] = {}
    draft_hybrid_plan_path = os.environ.get("SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN")
    if draft_hybrid_plan_path:
        real_draft_plan_path = os.path.realpath(draft_hybrid_plan_path)

        def draft_plan_sha256() -> str:
            digest = hashlib.sha256()
            with open(real_draft_plan_path, "rb") as plan_file:
                for chunk in iter(lambda: plan_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        draft_before_sha256 = draft_plan_sha256()
        expected_draft_sha256 = os.environ.get(
            "SGLANG_KT_DRAFT_HYBRID_EXPERT_PLAN_SHA256"
        )
        if (
            expected_draft_sha256 is not None
            and draft_before_sha256 != expected_draft_sha256
        ):
            raise RuntimeError(
                "live KTransformers draft hybrid plan no longer matches its "
                "launcher-admitted SHA-256"
            )
        loaded_draft_plan = torch.load(
            real_draft_plan_path, map_location="cpu", weights_only=True
        )
        if draft_plan_sha256() != draft_before_sha256:
            raise RuntimeError(
                "live KTransformers draft hybrid plan changed while reading server info"
            )
        if not isinstance(loaded_draft_plan, dict) or not isinstance(
            (draft_gpu_masks := loaded_draft_plan.get("gpu_experts_mask_by_rank")),
            torch.Tensor,
        ):
            raise RuntimeError("live KTransformers draft hybrid plan is malformed")
        draft_gpu_counts = draft_gpu_masks.to(device="cpu", dtype=torch.bool).sum(dim=2)
        draft_hybrid_plan_info = {
            "kt_draft_hybrid_expert_plan_sha256": draft_before_sha256,
            "kt_draft_hybrid_expert_plan_format": loaded_draft_plan.get("format"),
            "kt_draft_hybrid_gpu_rank_counts_by_layer": draft_gpu_counts.tolist(),
            "kt_draft_hybrid_min_gpu_experts_per_rank_per_layer": int(
                draft_gpu_counts.min().item()
            ),
            "kt_draft_hybrid_max_gpu_experts_per_rank_per_layer": int(
                draft_gpu_counts.max().item()
            ),
            "kt_draft_hybrid_total_gpu_expert_layers_by_rank": (
                draft_gpu_counts.sum(dim=1).tolist()
            ),
            "kt_draft_hybrid_source_profile_sha256": loaded_draft_plan.get(
                "source_profile_sha256"
            ),
            "kt_draft_hybrid_source_ordering_sha256": loaded_draft_plan.get(
                "source_ordering_sha256"
            ),
            "kt_draft_hybrid_gpu_selection_strategy": loaded_draft_plan.get(
                "gpu_selection_strategy"
            ),
        }

    sm86_small_batch_gemm_configured = (
        os.environ.get("SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM", "0") == "1"
    )
    sm86_small_batch_gemm_info = _summarize_dsv4_sm86_small_batch_gemm(
        internal_states,
        sm86_small_batch_gemm_configured,
        int(server_args.tp_size) * int(server_args.pp_size) * int(server_args.dp_size),
    )
    kt_task_queue_affinity_configured = (
        os.environ.get("KT_TASK_QUEUE_PIN_FIRST_CORE") == "1"
    )
    kt_task_queue_affinity_info = _summarize_kt_task_queue_affinity(
        internal_states,
        kt_task_queue_affinity_configured,
        server_args,
    )
    kt_single_numa_inline_dispatch_configured = (
        os.environ.get("KT_SINGLE_NUMA_INLINE_DISPATCH") == "1"
    )
    kt_single_numa_inline_dispatch_info = _summarize_kt_single_numa_inline_dispatch(
        internal_states,
        kt_single_numa_inline_dispatch_configured,
        server_args,
    )
    kt_mxfp4_avx_scale_fold_mode = os.environ.get("KT_MXFP4_AVX_SCALE_FOLD_MODE", "off")
    kt_mxfp4_avx_scale_fold_configured = kt_mxfp4_avx_scale_fold_mode not in (
        "",
        "off",
    )
    kt_mxfp4_avx_scale_fold_info = _summarize_kt_mxfp4_avx_scale_fold(
        internal_states,
        kt_mxfp4_avx_scale_fold_mode,
        server_args,
    )

    # server_args.model_config is not serializable but should be excluded by asdict.
    return msgspec_to_builtins(
        {
            **dataclasses.asdict(server_args),
            **_global_state.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
            # The launcher computes this digest and the KT loader rejects a
            # file mismatch or mid-load mutation before model admission. This
            # lets benchmark receipts bind caller provenance to the placement
            # actually accepted by every local model worker.
            "kt_hybrid_expert_plan_sha256": os.environ.get(
                "SGLANG_KT_HYBRID_EXPERT_PLAN_SHA256"
            ),
            **hybrid_plan_info,
            **draft_hybrid_plan_info,
            "kt_gpu_expert_admission_ceiling": server_args.kt_num_gpu_experts,
            "dsv4_small_row_routing_configured": os.environ.get(
                "SGLANG_V4_MXFP4_SMALL_ROW_ROUTING", "0"
            )
            == "1",
            "dsv4_sm86_small_batch_gemm_configured": (sm86_small_batch_gemm_configured),
            **sm86_small_batch_gemm_info,
            "kt_task_queue_affinity_configured": (kt_task_queue_affinity_configured),
            **kt_task_queue_affinity_info,
            "kt_single_numa_inline_dispatch_configured": (
                kt_single_numa_inline_dispatch_configured
            ),
            **kt_single_numa_inline_dispatch_info,
            "kt_mxfp4_avx_scale_fold_configured": (kt_mxfp4_avx_scale_fold_configured),
            **kt_mxfp4_avx_scale_fold_info,
            "dsv4_custom_all_reduce_v2_configured": os.environ.get(
                "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2", "0"
            )
            == "1",
            "kt_amx_fine_grained_decode_configured": os.environ.get(
                "KT_AMX_FINE_GRAINED_DECODE", "0"
            )
            == "1",
            "kt_mxfp4_amx_min_expert_tokens": int(
                os.environ.get("KT_MXFP4_AMX_MIN_EXPERT_TOKENS", "0")
            ),
            "kt_mxfp4_avx_tiled_min_expert_tokens": int(
                os.environ.get("KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS", "0")
            ),
            # Structured KV-event publisher descriptor for KV-aware routers.
            # `None` when publishing is disabled or misconfigured; see
            # `ServerArgs.describe_kv_events_publisher` for the precise contract.
            "kv_events": server_args.describe_kv_events_publisher(),
        }
    )


@app.get("/get_load")
async def get_load():
    """Get load metrics (deprecated - use /v1/loads instead).

    Legacy shim backed by /v1/loads. Projects the load snapshot down to the
    historical field shape (dp_rank, num_reqs, num_waiting_reqs, num_tokens,
    num_pending_tokens, ts_tic) so existing clients keep working.
    """
    logger.warning(
        "Endpoint '/get_load' is deprecated and will be removed in a future version. "
        "Please use '/v1/loads' instead."
    )
    load_results = await _global_state.tokenizer_manager.get_loads(include=["core"])
    ts = time.perf_counter()
    return [
        {
            "dp_rank": r.dp_rank,
            "num_reqs": r.num_running_reqs + r.num_waiting_reqs,
            "num_waiting_reqs": r.num_waiting_reqs,
            "num_tokens": r.num_total_tokens,
            "num_pending_tokens": r.num_total_tokens - r.num_used_tokens,
            "ts_tic": ts,
        }
        for r in load_results
    ]


# example usage:
# curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"pp_max_micro_batch_size": 8}}'
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def set_internal_state(
    obj: Annotated[SetInternalStateReq, Body()], request: Request
):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# Do not import `dumper.py` to avoid dependency
if os.environ.get("DUMPER_SERVER_PORT") == "reuse":

    @app.api_route("/dumper/{method}", methods=["POST"])
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def _dumper_control_handler(method: str, request: Request):
        body_bytes = await request.body()
        body = await request.json() if body_bytes else {}
        if not isinstance(body, dict):
            return ORJSONResponse(
                status_code=400,
                content={"error": "Request body must be a JSON object."},
            )
        obj = DumperControlReqInput(method=method, body=body)
        results = await _global_state.tokenizer_manager.dumper_control(obj)
        if any(not r.success for r in results):
            errors = [r.error for r in results if not r.success]
            return ORJSONResponse(status_code=400, content={"error": errors})
        return [x for result in results for x in result.response]


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"
            except ValueError as e:
                # A client disconnect also surfaces here. It's a client-side
                # cancellation, not a server error or bad input -- log it and
                # stop (the request was already aborted upstream) instead of
                # emitting a 400.
                if request is not None and await request.is_disconnected():
                    logger.info(f"[http_server] Client disconnected: {e}")
                    return
                out = {
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": 400,
                        "retryable": False,
                    }
                }
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/flush_cache", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def flush_cache(timeout: float = Query(0.0, ge=0.0)):
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache(timeout_s=timeout)
    if ret.success:
        content = (
            "Cache flushed.\nPlease check backend logs for more details. "
            "(When there are running or waiting requests, the operation will not be performed.)\n"
        )
    else:
        content = ret.message or "Flush cache failed.\n"
    return Response(
        content=content,
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/kt_expert_hotspot", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def kt_expert_hotspot(
    obj: Annotated[KTExpertHotspotReqInput, Body()], request: Request
):
    """Validate or commit an idle-boundary, CUDA-graph-safe expert plan."""

    results = await _global_state.tokenizer_manager.kt_expert_hotspot(obj)
    failures = [result.message for result in results if not result.success]
    if failures:
        return ORJSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"success": False, "errors": failures},
        )
    return ORJSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "success": True,
            "receipts": [result.receipt for result in results],
        },
    )


@app.post("/add_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def add_external_corpus(request: Request):
    """Add an external corpus for ngram speculative decoding."""
    from sglang.srt.managers.io_struct import AddExternalCorpusReqInput

    try:
        obj = AddExternalCorpusReqInput(**(await request.json()))
    except TypeError as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.add_external_corpus(obj)
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_id": result.corpus_id,
            "message": result.message,
            "loaded_token_count": result.loaded_token_count,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


@app.post("/remove_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remove_external_corpus(request: Request):
    """Remove an external corpus by ID."""
    body = await request.json()
    corpus_id = body.get("corpus_id")
    if not corpus_id:
        return ORJSONResponse(
            {"success": False, "message": "corpus_id is required."},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.remove_external_corpus(corpus_id)
    return ORJSONResponse(
        {"success": result.success, "message": result.message},
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


@app.get("/list_external_corpora")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def list_external_corpora():
    """List all active external corpora."""
    result = await _global_state.tokenizer_manager.list_external_corpora()
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_token_counts": result.corpus_token_counts,
            "message": result.message,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/clear_hicache_storage_backend", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend_deprecated():
    """Deprecated: use POST /hicache/storage-backend/clear."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content=(
            "Deprecated endpoint. Use POST /hicache/storage-backend/clear.\n"
            "Hierarchical cache storage backend cleared.\n"
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s -X POST http://127.0.0.1:30000/clear_hicache_storage_backend
@app.api_route("/hicache/storage-backend/clear", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend():
    """Clear the hierarchical cache storage backend."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content="Hierarchical cache storage backend cleared.\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
#  -H 'Content-Type: application/json' \
#   -d '{
#     "hicache_storage_backend": "file",
#     "hicache_storage_backend_extra_config_json": "{}",
#     "hicache_storage_prefetch_policy": "timeout",
#     "hicache_write_policy": "write_through"
#   }'
@app.api_route("/hicache/storage-backend", methods=["PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def attach_hicache_storage_backend(
    obj: Annotated[AttachHiCacheStorageReqInput, Body()],
):
    """Attach (enable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.attach_hicache_storage(
        hicache_storage_backend=obj.hicache_storage_backend,
        hicache_storage_backend_extra_config_json=obj.hicache_storage_backend_extra_config_json,
        hicache_storage_prefetch_policy=obj.hicache_storage_prefetch_policy,
        hicache_write_policy=obj.hicache_write_policy,
    )
    msg = ret.message
    return Response(
        content=(
            (
                "HiCache storage backend attached.\n"
                if ret.success
                else "Failed to attach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
@app.api_route("/hicache/storage-backend", methods=["DELETE"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def detach_hicache_storage_backend():
    """Detach (disable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.detach_hicache_storage()
    msg = ret.message
    return Response(
        content=(
            (
                "HiCache storage backend detached.\n"
                if ret.success
                else "Failed to detach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s http://127.0.0.1:30000/hicache/storage-backend
@app.get("/hicache/storage-backend")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def hicache_storage_backend_status():
    """Get current HiCache storage backend status (tokenizer-side view)."""
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    return {
        "hicache_storage_backend": _global_state.tokenizer_manager.server_args.hicache_storage_backend,
        "hicache_storage_backend_extra_config": _global_state.tokenizer_manager.server_args.hicache_storage_backend_extra_config,
        "hicache_storage_prefetch_policy": _global_state.tokenizer_manager.server_args.hicache_storage_prefetch_policy,
        "hicache_write_policy": _global_state.tokenizer_manager.server_args.hicache_write_policy,
    }


@app.api_route("/start_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_profile_async(obj: Annotated[Optional[ProfileReq], Body()] = None):
    """Start profiling."""
    await _global_state.tokenizer_manager.start_profile(obj or ProfileReq())
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/set_trace_level", methods=["GET", "POST"])
def set_trace_level(level: int = Query(..., ge=0)):
    set_global_trace_level(level)

    return Response(
        content="success",
        status_code=200,
    )


@app.api_route("/freeze_gc", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def freeze_gc_async():
    """
    See engine.freeze_gc for more details.
    """
    await _global_state.tokenizer_manager.freeze_gc()
    return Response(
        content="Garbage collection frozen.\n",
        status_code=200,
    )


@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_disk(
    obj: Annotated[UpdateWeightFromDiskReqInput, Body()], request: Request
):
    """Update the weights from disk inplace without re-launching the server."""
    (
        success,
        message,
        num_paused_requests,
    ) = await _global_state.tokenizer_manager.update_weights_from_disk(obj, request)

    content = {
        "success": success,
        "message": message,
        "num_paused_requests": num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post("/init_weights_send_group_for_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_send_group_for_remote_instance(
    obj: Annotated[InitWeightsSendGroupForRemoteInstanceReqInput, Body()],
    request: Request,
):
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.init_weights_send_group_for_remote_instance(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/send_weights_to_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def send_weights_to_remote_instance(
    obj: Annotated[SendWeightsToRemoteInstanceReqInput, Body()], request: Request
):
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.send_weights_to_remote_instance(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.get("/get_remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_remote_instance_transfer_engine_info(rank: int = None):
    """Get the server information (deprecated - use /remote_instance_transfer_engine_info instead)."""
    logger.warning(
        "Endpoint '/get_remote_instance_transfer_engine_info' is deprecated and will be removed in a future version. "
        "Please use '/remote_instance_transfer_engine_info' instead."
    )
    return await remote_instance_transfer_engine_info(rank=rank)


@app.get("/remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remote_instance_transfer_engine_info(rank: int = None):
    if rank is None or rank < 0:
        return ORJSONResponse(
            {"error": {"message": "Missing or invalid rank parameter"}},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    server_args = _global_state.tokenizer_manager.server_args
    try:
        resp = requests.get(
            f"{server_args.engine_info_bootstrap_url}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Failed to get transfer engine info for rank {rank}: {e}")

    return ORJSONResponse(
        {"error": {"message": f"Failed to get transfer engine info for rank {rank}"}},
        status_code=HTTPStatus.BAD_REQUEST,
    )


@app.post("/init_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_update_group(
    obj: Annotated[InitWeightsUpdateGroupReqInput, Body()], request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/destroy_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def destroy_weights_update_group(
    obj: Annotated[DestroyWeightsUpdateGroupReqInput, Body()], request: Request
):
    """Destroy the parameter update group."""
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.destroy_weights_update_group(obj, request)
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_tensor")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_tensor(
    obj: Annotated[UpdateWeightsFromTensorReqInput, Body()], request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """

    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj, request
    )

    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_distributed")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_distributed(
    obj: Annotated[UpdateWeightsFromDistributedReqInput, Body()], request: Request
):
    """Update model parameter from distributed online."""
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.update_weights_from_distributed(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(
    obj: Annotated[UpdateWeightsFromIPCReqInput, Body()], request: Request
):
    """Update the weights from IPC (Inter-Process Communication) for checkpoint-engine integration."""
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weight_version")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weight_version(
    obj: Annotated[UpdateWeightVersionReqInput, Body()], request: Request
):
    """Update the weight version. This operation requires no active requests."""
    if obj.abort_all_requests:
        _global_state.tokenizer_manager.abort_request(abort_all=True)

    # Use a simple approach without the complex lock mechanism for now
    # since weight_version update is a simple operation that doesn't affect model weights
    try:
        # Update the weight version in server args (the single source of truth)
        _global_state.tokenizer_manager.server_args.override(
            "http.update_weight_version", weight_version=obj.new_version
        )

        return ORJSONResponse(
            {
                "success": True,
                "message": f"Weight version updated to {obj.new_version}",
                "new_version": obj.new_version,
            },
            status_code=HTTPStatus.OK,
        )
    except Exception as e:
        return ORJSONResponse(
            {
                "success": False,
                "message": f"Failed to update weight version: {str(e)}",
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_weights_by_name(
    obj: Annotated[GetWeightsByNameReqInput, Body()], request: Request
):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj, request)
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def release_memory_occupation(
    obj: Annotated[ReleaseMemoryOccupationReqInput, Body()], request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def resume_memory_occupation(
    obj: Annotated[ResumeMemoryOccupationReqInput, Body()], request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/weights_checker", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def check_weights(
    obj: Annotated[Optional[CheckWeightsReqInput], Body()] = None,
    request: Request = None,
):
    if obj is None:
        obj = CheckWeightsReqInput()
    (
        success,
        message,
        ranks,
        per_engine_checksum,
    ) = await _global_state.tokenizer_manager.check_weights(obj, request)
    body = {"success": success, "message": message}
    if ranks is not None:
        body["ranks"] = ranks
    if per_engine_checksum is not None:
        body["per_engine_checksum"] = per_engine_checksum
    return ORJSONResponse(body, status_code=200 if success else HTTPStatus.BAD_REQUEST)


@app.api_route("/slow_down", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def slow_down(obj: Annotated[SlowDownReqInput, Body()], request: Request):
    """Slow down the system deliberately. Only for testing. Example scenario:
    when we want to test performance of D in large-scale PD disaggregation and have no enough nodes for P,
    we can use this to slow down D to let it have enough running sequences, and then disable slowdown
    to let it run in full batch size.
    """
    try:
        await _global_state.tokenizer_manager.slow_down(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/load_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def load_lora_adapter(
    obj: Annotated[LoadLoRAAdapterReqInput, Body()], request: Request
):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter(obj, request)
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/load_lora_adapter_from_tensors", methods=["POST"])
async def load_lora_adapter_from_tensors(
    obj: Annotated[LoadLoRAAdapterFromTensorsReqInput, Body()], request: Request
):
    """Load a new LoRA adapter from tensors without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter_from_tensors(
        obj, request
    )
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/unload_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def unload_lora_adapter(
    obj: Annotated[UnloadLoRAAdapterReqInput, Body()], request: Request
):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.unload_lora_adapter(obj, request)
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: Annotated[OpenSessionReqInput, Body()], request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: Annotated[CloseSessionReqInput, Body()], request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/configure_logging", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def configure_logging(
    obj: Annotated[ConfigureLoggingReq, Body()], request: Request
):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


@app.post("/abort_request")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def abort_request(obj: Annotated[AbortReq, Body()], request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.post("/parse_function_call")
async def parse_function_call_request(
    obj: Annotated[ParseFunctionCallReq, Body()], request: Request
):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = FunctionCallParser(
        tools=obj.tools,
        tool_call_parser=obj.tool_call_parser,
        tokenizer=get_global_state().tokenizer_manager.tokenizer,
    )

    # 2) Call the non-stream parsing method (non-stream)
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # Convert pydantic objects to dictionaries
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/separate_reasoning")
async def separate_reasoning_request(
    obj: Annotated[SeparateReasoningReqInput, Body()], request: Request
):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = ReasoningParser(
        model_type=obj.reasoning_parser,
        request=request,
        tokenizer=get_global_state().tokenizer_manager.tokenizer,
    )

    # 2) Call the non-stream parsing method (non-stream)
    if obj.return_blocks:
        blocks = parser.parse_non_stream_blocks(obj.text)
        reasoning_blocks = [b["text"] for b in blocks if b["type"] == "reasoning"]
        text_blocks = [b["text"] for b in blocks if b["type"] == "text"]
        reasoning_text = "".join(reasoning_blocks)
        normal_text = "".join(text_blocks)
    else:
        reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }
    if obj.return_blocks:
        response_data["reasoning_blocks"] = reasoning_blocks
        response_data["text_blocks"] = text_blocks
        response_data["blocks"] = blocks

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/pause_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def pause_generation(
    obj: Annotated[PauseGenerationReqInput, Body()], request: Request
):
    """Pause generation."""
    await _global_state.tokenizer_manager.pause_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )


@app.post("/continue_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def continue_generation(
    obj: Annotated[ContinueGenerationReqInput, Body()], request: Request
):
    """Continue generation."""
    await _global_state.tokenizer_manager.continue_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/embeddings",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_embeddings(request: EmbeddingRequest, raw_request: Request):
    """OpenAI-compatible embeddings endpoint."""
    return await raw_request.app.state.openai_serving_embedding.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/classify",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_classify(request: ClassifyRequest, raw_request: Request):
    """OpenAI-compatible classification endpoint."""
    return await raw_request.app.state.openai_serving_classify.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_tokenize(request: TokenizeRequest, raw_request: Request):
    """OpenAI-compatible tokenization endpoint."""
    return await raw_request.app.state.openai_serving_tokenize.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_detokenize(request: DetokenizeRequest, raw_request: Request):
    """OpenAI-compatible detokenization endpoint."""
    return await raw_request.app.state.openai_serving_detokenize.handle_request(
        request, raw_request
    )


@app.post("/v1/audio/transcriptions")
async def openai_v1_audio_transcriptions(
    raw_request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="default"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    stream: bool = Form(default=False),
    timestamp_granularities: Optional[List[str]] = Form(
        default=None, alias="timestamp_granularities[]"
    ),
):
    """OpenAI-compatible audio transcription endpoint."""
    if response_format not in ["json", "text", "verbose_json"]:
        return ORJSONResponse(
            content={
                "error": {
                    "message": "Only 'json', 'text', and 'verbose_json' formats supported"
                }
            },
            status_code=400,
        )

    audio_data = await file.read()

    return (
        await raw_request.app.state.openai_serving_transcription.create_transcription(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            stream=stream,
            timestamp_granularities=timestamp_granularities,
            raw_request=raw_request,
        )
    )


@app.websocket("/v1/realtime")
async def openai_v1_realtime_transcription(ws: WebSocket):
    """OpenAI Realtime transcription WebSocket endpoint."""
    # /v1/realtime is OpenAI's unified Realtime URL covering transcription +
    # chat modes. This handler implements the transcription subset only;
    # chat-mode session.update payloads are rejected by the
    # `Literal["transcription"]` constraint on TranscriptionSessionConfig.type
    # (see realtime/protocol.py).
    await ws.app.state.openai_serving_transcription.handle_websocket(ws)


@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]
    model_cards = []

    # Add base model
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            )
        )

    # Add loaded LoRA adapters
    if _global_state.tokenizer_manager.server_args.enable_lora:
        lora_registry = _global_state.tokenizer_manager.lora_registry
        for _, lora_ref in lora_registry.get_all_adapters().items():
            model_cards.append(
                ModelCard(
                    id=lora_ref.lora_name,
                    root=lora_ref.lora_path,
                    parent=served_model_names[0],
                    max_model_len=None,
                )
            )

    return ModelList(data=model_cards)


@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]

    if model not in served_model_names:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    return ModelCard(
        id=model,
        root=model,
        max_model_len=_global_state.tokenizer_manager.model_config.context_len,
    )


@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    """Endpoint for the scoring API. Supports CausalLM (logprob-based) and SequenceClassification (class logit-based) models. See Engine.score() for documentation."""
    return await raw_request.app.state.openai_serving_score.handle_request(
        request, raw_request
    )


@app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def v1_responses_request(request: ResponsesRequest, raw_request: Request):
    """Endpoint for the responses API with reasoning support."""

    result = await raw_request.app.state.openai_serving_responses.create_responses(
        request, raw_request
    )

    # Handle streaming responses
    if isinstance(result, AsyncGenerator):
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return result


@app.get("/v1/responses/{response_id}")
async def v1_retrieve_responses(response_id: str, raw_request: Request):
    """Retrieve a response by ID."""
    return await raw_request.app.state.openai_serving_responses.retrieve_responses(
        response_id
    )


@app.post("/v1/responses/{response_id}/cancel")
async def v1_cancel_responses(response_id: str, raw_request: Request):
    """Cancel a background response."""
    return await raw_request.app.state.openai_serving_responses.cancel_responses(
        response_id
    )


@app.api_route(
    "/v1/rerank", methods=["POST", "PUT"], dependencies=[Depends(validate_json_request)]
)
async def v1_rerank_request(request: V1RerankReqInput, raw_request: Request):
    """Endpoint for reranking documents based on query relevance."""
    return await raw_request.app.state.openai_serving_rerank.handle_request(
        request, raw_request
    )


##### Ollama-compatible API endpoints #####

_ollama_root_route = os.environ.get("SGLANG_OLLAMA_ROOT_ROUTE")
if _ollama_root_route is not None:

    @app.get(_ollama_root_route)
    @app.head(_ollama_root_route)
    async def ollama_root():
        """Ollama-compatible root endpoint."""
        return "Ollama is running"

else:

    @app.get("/")
    @app.head("/")
    async def sglang_root():
        """Default root endpoint."""
        return "SGLang is running"


@app.post(os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"))
async def ollama_chat(request: OllamaChatRequest, raw_request: Request):
    """Ollama-compatible chat endpoint."""
    return await raw_request.app.state.ollama_serving.handle_chat(request, raw_request)


@app.post(os.environ.get("SGLANG_OLLAMA_GENERATE_ROUTE", "/api/generate"))
async def ollama_generate(request: OllamaGenerateRequest, raw_request: Request):
    """Ollama-compatible generate endpoint."""
    return await raw_request.app.state.ollama_serving.handle_generate(
        request, raw_request
    )


@app.get(os.environ.get("SGLANG_OLLAMA_TAGS_ROUTE", "/api/tags"))
async def ollama_tags(raw_request: Request):
    """Ollama-compatible list models endpoint."""
    return raw_request.app.state.ollama_serving.get_tags()


@app.post(os.environ.get("SGLANG_OLLAMA_SHOW_ROUTE", "/api/show"))
async def ollama_show(request: OllamaShowRequest, raw_request: Request):
    """Ollama-compatible show model info endpoint."""
    return raw_request.app.state.ollama_serving.get_show(request.model)


##### Anthropic-compatible API endpoints #####


@app.post("/v1/messages", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_messages(
    request: AnthropicMessagesRequest, raw_request: Request
):
    """Anthropic-compatible Messages API endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_messages(
        request, raw_request
    )


@app.post("/v1/messages/count_tokens", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_count_tokens(
    request: AnthropicCountTokensRequest, raw_request: Request
):
    """Anthropic-compatible token counting endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_count_tokens(
        request, raw_request
    )


## SageMaker API
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


## Vertex AI API
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(
    vertex_req: Annotated[VertexGenerateReqInput, Body()], raw_request: Request
):
    if not vertex_req.instances:
        return []
    inputs = {}
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),
    )
    ret = await generate_request(req, raw_request)
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


# FIXME: In theory we should configure ADMIN_FORCE for some entrypoints, but doing so
# would currently cause all endpoints to go through add_api_key_middleware
# (even when neither api-key nor admin-api-key is configured).
#
# For now, we simulate ADMIN_FORCE by explicitly checking the admin API key parameter.
# Once the auth wiring is refactored so ADMIN_FORCE only affects the intended
# admin endpoints, we should switch this logic to use ADMIN_FORCE directly.
def _admin_api_key_missing_response(
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "error": (
                "This endpoint requires admin API key, but this server was started "
                "without one (admin-api-key). Restart with --admin-api-key to enable."
            )
        },
        status_code=status_code,
    )


# Minimal 32x32 black PNG (base64, GLM4v requires at least 32x32 sized image)
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Kimi K2.5/K2.7 runs its MoonViT position interpolation through torch.compile.
# The minimal image above does not exercise a representative image shape and
# leaves the first client request paying the compilation cost. Keep this
# narrowly scoped: a larger default warmup image would unnecessarily lengthen
# startup for VLMs whose encoders do not have this behavior.
KIMI_VLM_WARMUP_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAADEUlEQVR42u3BgQAAAADDoPlTX+EAVQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMBvArQAAf/YBFAAAAAASUVORK5CYII="

# Kimi K3's native vision preprocessing uses a 14x14 spatial patch size. A
# 448x448 image exercises a representative 32x32 patch grid without padding.
KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAcAAAAHACAIAAAC6Ry8kAAACX0lEQVR42u3BMQEAAADCoPVPbQwfoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAtwEyRwAB32QPDQAAAABJRU5ErkJggg=="


def _get_vlm_warmup_image_base64(model_info: dict) -> str:
    """Choose the VLM image used by the startup warmup request.

    A 512x512 image triggers Kimi K2.5/K2.7's representative compiled
    position-interpolation path during startup. Kimi K3 uses a 448x448 image
    matching its native vision patch grid. This keeps one-time vision setup
    work out of the first external image request.
    Other VLMs retain the minimal image to avoid changing their startup cost.
    """

    architectures = model_info.get("architectures") or []
    if (
        "KimiK3ForConditionalGeneration" in architectures
        or model_info.get("model_type") == "kimi_k3"
    ):
        logger.info(
            "Using a 448x448 image for Kimi K3 VLM startup warmup to exercise "
            "its native 32x32 vision patch grid."
        )
        return KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64
    if "KimiK25ForConditionalGeneration" in architectures:
        logger.info(
            "Using a 512x512 image for Kimi VLM startup warmup to compile "
            "MoonViT position interpolation."
        )
        return KIMI_VLM_WARMUP_PNG_PICTURE_BASE64
    return MINIMUM_PNG_PICTURE_BASE64


async def _send_disaggregation_warmup_requests(
    server_args: ServerArgs,
    url: str,
    headers: Dict[str, str],
    ssl_verify: Union[bool, str],
    timeout: int,
) -> List[int]:
    ssl_context = (
        ssl_verify
        if isinstance(ssl_verify, bool)
        else ssl.create_default_context(cafile=ssl_verify)
    )

    async def send_request(session: aiohttp.ClientSession, dp_rank: int) -> int:
        json_data = {
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "bootstrap_host": FAKE_BOOTSTRAP_HOST,
            "bootstrap_room": dp_rank,
            "input_ids": [10, 11, 12, 13],
            "routed_dp_rank": dp_rank,
        }
        async with session.post(
            url + "/generate", json=json_data, ssl=ssl_context
        ) as response:
            await response.read()
            return response.status

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers=headers,
    ) as session:
        return await asyncio.gather(
            *(send_request(session, dp_rank) for dp_rank in range(server_args.dp_size))
        )


def _execute_server_warmup(server_args: ServerArgs):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    ssl_verify = server_args.ssl_verify()

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(
                url + "/model_info", timeout=5, headers=headers, verify=ssl_verify
            )
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success

    model_info = res.json()

    # Construct a warmup request (MLX: text warmup for VLM-advertising models; TODO: enable image warmup).
    # A language-only worker may advertise VLM capability for encoder
    # disaggregation, but its local warmup must stay on the text path.
    is_vlm = (
        bool(model_info.get("has_image_understanding", False))
        and not server_args.language_only
        and not is_mps()
    )
    if model_info["is_generation"]:
        if is_vlm and not server_args.skip_tokenizer_init:
            request_name = "/v1/chat/completions"
        else:
            request_name = "/generate"
    else:
        request_name = "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    elif (
        is_vlm
        and server_args.disaggregation_mode == "null"
        and model_info["is_generation"]
    ):
        served_model_name = ""
        if not envs.SGLANG_RUST_SERVER.get():
            served_model_name = _global_state.tokenizer_manager.served_model_name
        else:
            # _global_state.tokenizer_manager is not initialized in the rust server,
            # so we need to get the model name from the model_info
            served_model_name = model_info.get(
                "model_path", server_args.served_model_name
            )
            served_model_name = served_model_name or server_args.model_path
        # TODO: ChatCompletionRequest does not have bootstrap info required by disaggregation mode, disable image-warmup for now
        # Only use chat completions format for generation models, not embedding models
        json_data = {
            "model": served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                f"{_get_vlm_warmup_image_base64(model_info)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
            "temperature": 0.0,
        }
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # Config debug dumping
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0

    # Send a warmup request
    warmup_timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=warmup_timeout if warmup_timeout > 0 else 600,
                verify=ssl_verify,
            )
            assert res.status_code == 200, f"{res.text}"
            # Skip server_status update for Rust server
            if not envs.SGLANG_RUST_SERVER.get():
                _global_state.tokenizer_manager.server_status = ServerStatus.Up

        else:
            # TODO: @rainj-me fix this when Rust server supports disaggregation
            assert not envs.SGLANG_RUST_SERVER.get(), (
                "Rust server is not supported for disaggregation warmup for now"
            )
            logger.info(f"Start of pd disaggregation warmup ...")
            status_codes = asyncio.run(
                _send_disaggregation_warmup_requests(
                    server_args=server_args,
                    url=url,
                    headers=headers,
                    ssl_verify=ssl_verify,
                    timeout=warmup_timeout if warmup_timeout > 0 else 1800,
                )
            )
            failed_status_codes = [code for code in status_codes if code != 200]
            if not failed_status_codes:
                logger.info(
                    "Disaggregation warmup requests completed for all %s DP ranks",
                    server_args.dp_size,
                )
                logger.info("End of disaggregation warmup")
                _global_state.tokenizer_manager.server_status = ServerStatus.Up
            else:
                logger.info(
                    "Disaggregation warmup failed (mode=%s), status codes: %s",
                    server_args.disaggregation_mode,
                    failed_status_codes,
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy

    except Exception:
        last_traceback = get_exception_traceback()
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return False

    return success


def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,
    execute_warmup_func: Callable = _execute_server_warmup,
):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()

    # Joiner schedulers are served through the primary after adoption.
    skip_elastic_joiner_warmup = server_args.is_ep_scale_joiner
    if skip_elastic_joiner_warmup:
        logger.debug(
            "[Elastic EP] Skipping server warmup for elastic joiner (ep_join_mode=%s)",
            server_args.ep_join_mode,
        )

    if not server_args.skip_server_warmup and not skip_elastic_joiner_warmup:
        if not execute_warmup_func(server_args):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    # The server is ready for requests
    logger.info("The server is fired up and ready to roll!")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if launch_callback is not None:
        launch_callback()


def _wait_weights_ready():
    """Wait for weights to be ready within the specified timeout."""
    timeout = WAIT_WEIGHTS_READY_TIMEOUT
    start_time = time.time()

    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)

    # Timeout reached without weights being ready
    logger.error(
        f"Weights are not ready after waiting {timeout} seconds. "
        f"Consider increasing SGLANG_WAIT_WEIGHTS_READY_TIMEOUT environment variable. "
        f"Current status: initial_weights_loaded={_global_state.tokenizer_manager.initial_weights_loaded}"
    )


def _run_granian_server(
    host,
    port,
    log_level,
    tokenizer_worker_num=1,
    ssl_certfile=None,
    ssl_keyfile=None,
    ssl_ca_certs=None,
    ssl_keyfile_password=None,
    ssl_verify=False,  # MTls is not supported
    backlog=2048,
    backpressure=2048,
):
    """Serve the in-process ASGI app with Granian (embedded mode) over HTTP/2.

    Unlike Granian's default multi-process server, the embedded server runs a
    single worker as an asyncio task inside the current process. It therefore
    serves the live ``app`` object directly and reuses the already-initialized
    global state (tokenizer manager, templates, ...) through the normal
    single-tokenizer lifespan path -- no shared memory or worker re-init needed.
    The event loop is uvloop. The default backlog and backpressure values are set
    exactly like uvicorn's defaults.
    """
    import signal

    from granian import Granian
    from granian.constants import HTTPModes, Interfaces, Loops
    from granian.server.embed import Server as GranianEmbeddedServer

    Server = GranianEmbeddedServer if tokenizer_worker_num == 1 else Granian
    target = (
        app if tokenizer_worker_num == 1 else "sglang.srt.entrypoints.http_server:app"
    )
    granian_kwargs = dict(
        target=target,
        address=host,
        port=port,
        interface=Interfaces.ASGI,
        http=HTTPModes.auto,
        log_level=log_level,
        ssl_cert=ssl_certfile,
        ssl_key=ssl_keyfile,
        ssl_key_password=ssl_keyfile_password,
        ssl_ca=ssl_ca_certs,
        ssl_client_verify=ssl_verify,
        backlog=backlog,
        backpressure=backpressure,
    )

    if tokenizer_worker_num > 1:
        granian_kwargs["workers"] = tokenizer_worker_num
        granian_kwargs["loop"] = Loops.uvloop

    server = Server(**granian_kwargs)

    if tokenizer_worker_num == 1:

        async def serve():
            # The embedded server does not install its own signal handlers, so wire
            # SIGINT/SIGTERM to a graceful stop, mirroring uvicorn's behavior.
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, server.stop)
                except (NotImplementedError, ValueError):
                    pass
            await server.serve()

        uvloop.run(serve())
    else:
        server.serve()


def _setup_and_run_http_server(
    server_args: ServerArgs,
    tokenizer_manager,
    template_manager,
    port_args: PortArgs,
    scheduler_infos: List[Dict],
    subprocess_watchdog: Optional[SubprocessWatchdog],
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """Set up global state, configure middleware, and run uvicorn.

    Called by launch_server after subprocesses have been launched.
    """
    # Set global states
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )

    # Store watchdog on tokenizer_manager (single source of truth for SIGQUIT handler)
    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    # Pass additional arguments to the lifespan function.
    # They will be used for additional initialization setups.
    if server_args.tokenizer_worker_num == 1:
        # If it is single tokenizer mode, we can pass the arguments by attributes of the app object.
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )

        # Add api key authorization
        # This is only supported in single tokenizer mode.
        #
        # Backward compatibility:
        # - api_key only: behavior matches legacy (all endpoints require api_key)
        # - no keys: legacy had no restriction; ADMIN_FORCE endpoints must still be rejected when
        #   admin_api_key is not configured.
        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
            )
    else:
        # If it is multi-tokenizer mode, we need to write the arguments to shared memory
        # for other worker processes to read.
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )

    try:
        # Update logging configs
        set_uvicorn_logging_configs(server_args)

        if server_args.ssl_certfile:
            logger.info(
                f"SSL enabled: certfile={server_args.ssl_certfile}, "
                f"keyfile={server_args.ssl_keyfile}"
            )

        # Listen for HTTP requests
        if server_args.tokenizer_worker_num == 1:
            if server_args.enable_http2:
                logger.info(
                    f"Starting embedded Granian HTTP/2 server on "
                    f"{server_args.host}:{server_args.port}"
                )
                _run_granian_server(
                    host=server_args.host,
                    port=server_args.port,
                    log_level=server_args.log_level_http or server_args.log_level,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                    ssl_verify=False,  # No MTLS supported for now.
                )
            elif server_args.enable_ssl_refresh:
                # Use Config/Server API for access to the SSLContext.
                config = uvicorn.Config(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
                config.load()  # Creates the SSLContext

                from sglang.srt.entrypoints.ssl_utils import SSLCertRefresher

                server = uvicorn.Server(config)

                async def _run_with_ssl_refresh():
                    refresher = SSLCertRefresher(
                        config.ssl,
                        server_args.ssl_keyfile,
                        server_args.ssl_certfile,
                        server_args.ssl_ca_certs,
                    )
                    logger.info("SSL certificate auto-refresh enabled.")
                    try:
                        await server.serve()
                    finally:
                        refresher.stop()

                import asyncio

                asyncio.run(_run_with_ssl_refresh())
            else:
                # Default case, one tokenizer process
                uvicorn.run(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
        else:
            # Multiple tokenizer and http processes
            from uvicorn.config import LOGGING_CONFIG

            LOGGING_CONFIG["loggers"]["sglang.srt.entrypoints.http_server"] = {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            }

            if server_args.enable_ssl_refresh:
                logger.warning(
                    "--enable-ssl-refresh is not supported with multiple "
                    "tokenizer workers (--tokenizer-worker-num > 1). "
                    "SSL refresh will be disabled."
                )

            if server_args.enable_http2:
                logger.info(
                    f"Starting embedded Granian HTTP/2 server on "
                    f"{server_args.host}:{server_args.port}"
                )
                _run_granian_server(
                    host=server_args.host,
                    port=server_args.port,
                    log_level=server_args.log_level_http or server_args.log_level,
                    tokenizer_worker_num=server_args.tokenizer_worker_num,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
            else:
                uvicorn.run(
                    "sglang.srt.entrypoints.http_server:app",
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    timeout_worker_healthcheck=envs.SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT.get(),
                    loop="uvloop",
                    workers=server_args.tokenizer_worker_num,
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
    finally:
        if server_args.tokenizer_worker_num > 1:
            if multi_tokenizer_args_shm is not None:
                multi_tokenizer_args_shm.unlink()
            if _global_state is not None:
                _global_state.tokenizer_manager.socket_mapping.clear_all_sockets()


def _start_native_grpc_server_for_runtime(
    server_args,
    tokenizer_manager,
    template_manager,
    scheduler_info,
):
    try:
        from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
        from sglang.srt.grpc import _core as grpc_native
    except ImportError as e:
        raise RuntimeError(
            "Native gRPC extension (sglang.srt.grpc._core) not found in this wheel, "
            "but --grpc-port was set. The extension is built from "
            "rust/sglang-grpc/ via setuptools-rust during wheel build. Either "
            "install a wheel that includes the extension or unset --grpc-port."
        ) from e

    runtime_handle = RuntimeHandle(
        tokenizer_manager=tokenizer_manager,
        template_manager=template_manager,
        server_args=server_args,
        scheduler_info=scheduler_info or {},
    )

    grpc_handle = grpc_native.start_server(
        host=server_args.host,
        port=server_args.grpc_port,
        runtime_handle=runtime_handle,
        worker_threads=server_args.grpc_worker_threads,
    )
    logger.info(
        f"Native gRPC server started on {server_args.host}:{server_args.grpc_port}"
    )
    return grpc_handle


def _shutdown_native_grpc_server(grpc_handle) -> None:
    if grpc_handle is None:
        return
    try:
        grpc_handle.shutdown()
    except Exception as e:
        logger.warning(f"Failed to shut down native gRPC server: {e}")


def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # Launch subprocesses
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
        _weight_cache_daemon_procs,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    if envs.SGLANG_RUST_SERVER.get():
        # The Rust server serves api-server, tokenizer, and detokenizer, so the
        # main process has no Python HTTP server / tokenizer manager to run.
        # Run a warmup /generate before advertising readiness: the Rust /health
        # and /get_model_info endpoints are static (200 as soon as the server
        # binds, before any forward pass), so without this the first real request
        # pays the cold-start cost (observed as a >60s first generation).
        if not server_args.skip_server_warmup:
            _execute_server_warmup(server_args)
        logger.info("The server is fired up and ready to roll!")
        if launch_callback is not None:
            launch_callback()
        scheduler_init_result.block_until_scheduler_exits()
    else:
        _setup_and_run_http_server(
            server_args,
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result.scheduler_infos,
            subprocess_watchdog,
            execute_warmup_func=execute_warmup_func,
            launch_callback=launch_callback,
        )
