"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 1-200 (None disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import asyncio
import dataclasses
import json
import logging
import pathlib
import random
import string
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal, assert_never

import fastapi
import pydantic
import torch
import uvicorn
from fastapi import responses
from fastapi.middleware import cors, gzip

from pealm import checkpoint, constants, model, settings, utils
from pealm.chat import engine, state_machine
from pealm.tokenizer import PeashooterTokenizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: engine.Engine
    autocast_ctx: torch.amp.autocast


class WorkerPool:
    def __init__(
        self,
        chat_settings: settings.ChatWeb,
        model_settings: settings.GPT2,
    ) -> None:
        """Load model on each GPU."""
        num_gpus = chat_settings.num_gpus
        # init
        device = utils.compute_init(
            use_accelerator=chat_settings.use_accelerator,
            seed=chat_settings.seed,
            torch_seed=chat_settings.torch_seed,
            fp32_matmul_precision=chat_settings.fp32_matmul_precision,
        )
        # share tokenizer across workers since it's thread-safe
        tokenizer = PeashooterTokenizer.load(chat_settings.tokenizer_dir)

        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if num_gpus > 1 and device.type != "cuda":
            msg = "only CUDA supports multiple GPUs. other accelerators do not."
            raise ValueError(msg)

        self.num_gpus = num_gpus
        self.workers: list[Worker] = []
        self._available_workers: asyncio.Queue[Worker] = asyncio.Queue(maxsize=self.num_gpus)

        logger.info("initializing worker pool with %d GPUs", self.num_gpus)
        for gpu_id in range(self.num_gpus):
            # use different gpu per worker if multiple gpus are available
            if num_gpus > 1 and device.type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")

            logger.info("loading model on %s", device)

            # TODO: replace with actual peashooter model
            _model = model.GPT2(
                context_size=model_settings.context_size,
                # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
                vocab_size=model_settings.vocab_size,
                embedding_size=model_settings.embedding_size,
                num_layers=model_settings.num_layers,
                num_heads=model_settings.num_heads,
            ).to(device)

            # load checkpoint, each gpu gets their own replica
            model_state_dict = checkpoint.Checkpoint.load_weights(chat_settings.ckpt, map_location=device)
            _model.load_state_dict(model_state_dict)

            # only compile after graph is in place and on device
            _model.eval().compile()

            # create engine for efficient generation
            _engine = engine.Engine(_model, tokenizer)
            autocast_ctx = torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=chat_settings.use_mixed_precision
            )

            worker = Worker(gpu_id, device, _engine, autocast_ctx)
            self._available_workers.put_nowait(worker)
            self.workers.append(worker)

    @asynccontextmanager
    async def acquire_worker(self) -> AsyncGenerator[Worker]:
        """Acquire a worker from the pool. The async context manager ensures the worker is released back to the pool
        after use."""
        worker = await self._available_workers.get()
        try:
            yield worker
        finally:
            await self._available_workers.put(worker)

    @property
    def is_ready(self) -> bool:
        """Check if the worker pool is ready to serve requests."""
        return len(self.workers) > 0

    @property
    def num_workers(self) -> int:
        """Get the total number of workers in the pool."""
        return len(self.workers)

    @property
    def num_available(self) -> int:
        """Get the number of available workers in the pool."""
        return self._available_workers.qsize()

    @property
    def num_busy(self) -> int:
        """Get the number of busy workers in the pool."""
        return self.num_workers - self.num_available


# NOTE: only decorates input models for extra validation, not output models


class HealthResponse(pydantic.BaseModel):
    status: str
    ready: bool
    num_gpus: int
    available_workers: int


class GPUInfo(pydantic.BaseModel):
    gpu_id: int
    device: str


class StatsResponse(pydantic.BaseModel):
    total_workers: int
    available_workers: int
    busy_workers: int
    workers: list[GPUInfo]


class ChatMessage(pydantic.BaseModel):
    role: Annotated[
        Literal["system", "user", "assistant"],
        pydantic.Field(
            description="Role of the message sender",
        ),
    ]
    content: Annotated[
        str,
        pydantic.Field(min_length=1, max_length=constants.PS_CHAT_WEB_MAX_MESSAGE_LEN),
    ]


class ChatRequest(pydantic.BaseModel):
    messages: Annotated[list[ChatMessage], pydantic.Field(min_length=1, max_length=constants.PS_CHAT_WEB_MAX_MESSAGES)]
    # user can specify params down here that are specific to their request
    # if none, use system defaults
    max_tokens: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            ge=constants.PS_CHAT_WEB_MIN_TOKENS,
            le=constants.PS_CHAT_WEB_MAX_TOKENS,
        ),
    ] = None
    temperature: Annotated[
        pydantic.NonNegativeFloat | None,
        pydantic.Field(
            le=constants.MAX_TEMPERATURE,
        ),
    ] = None
    top_k: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            le=constants.PS_CHAT_WEB_MAX_TOP_K,
        ),
    ] = None

    @pydantic.field_validator("messages", mode="after")
    @classmethod
    def validate_conversation_length(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if sum(len(msg.content) for msg in messages) > constants.PS_CHAT_WEB_MAX_CONVERSATION_LEN:
            msg = (
                f"total conversation is too long. maximum {constants.PS_CHAT_WEB_MAX_CONVERSATION_LEN} characters "
                "allowed"
            )
            raise ValueError(msg)
        return messages


@dataclasses.dataclass
class AssistantResponseChunk:
    text: str


@dataclasses.dataclass
class AssistantResponseDone:
    done: Literal[True] = True


class ChatWebStateMachine(state_machine.Chat):
    def __init__(
        self,
        engine: engine.Engine,
        autocast_ctx: torch.amp.autocast,
        messages: list[ChatMessage],
    ) -> None:
        super().__init__(engine, autocast_ctx)
        self._initialize_conversation(messages)

    def _initialize_conversation(self, messages: list[ChatMessage]) -> None:
        for message in messages:
            if message.role == "user":
                self.add_user_message(message.content)
            elif message.role == "assistant":
                self.add_assistant_message(message.content)

    async def generate_stream(
        self,
        max_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> AsyncGenerator[AssistantResponseChunk | AssistantResponseDone]:
        """Generate assistant response with streaming."""

        self.conversation_tokens.append(self.assistant_start)
        # accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
        accumulated_tokens = []
        # track the last complete UTF-8 string (without replacement characters)
        last_clean_text = ""

        with self.autocast_ctx:
            for token_column, _ in self.engine.generate(
                self.conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=random.randint(0, 2**31 - 1),  # noqa: S311
            ):
                token = token_column[0]

                # stopping criteria
                if token in (self.assistant_end, self.bos):
                    break

                # append the token to sequence
                accumulated_tokens.append(token)
                # decode all accumulated tokens to get proper UTF-8 handling
                # note that decode is a quite efficient operation, basically table lookup and string concat
                current_text = self.tokenizer.decode(accumulated_tokens)
                # only emit text if it doesn't end with a replacement character
                # this ensures we don't emit incomplete UTF-8 sequences
                if not current_text.endswith("�"):
                    # extract only the new text since last clean decode
                    new_text = current_text[len(last_clean_text) :]
                    # only yield if there's new content
                    if new_text:
                        yield AssistantResponseChunk(text=new_text)
                        last_clean_text = current_text

        yield AssistantResponseDone()


def create_app(chat_settings: settings.ChatWeb, model_settings: settings.GPT2) -> fastapi.FastAPI:
    """Create the FastAPI app and initialize the worker pool."""
    pool = WorkerPool(chat_settings, model_settings)

    async def get_worker() -> AsyncGenerator[Worker]:
        """DI dependency that acquires a worker from the pool for each request that needs it."""
        async with pool.acquire_worker() as worker:
            yield worker

    app = fastapi.FastAPI()

    app.add_middleware(
        cors.CORSMiddleware,  # ty:ignore[invalid-argument-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: fastapi.Request, call_next: Callable) -> fastapi.Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    app.add_middleware(gzip.GZipMiddleware, minimum_size=1000, compresslevel=5)  # ty:ignore[invalid-argument-type]

    @app.get("/")
    async def root() -> responses.HTMLResponse:
        """Serve the chat UI."""
        ui_html_path = pathlib.Path(__file__).parent / "ui.html"
        with ui_html_path.open(encoding="utf-8") as f:
            html_content = f.read()
        # replace the API_URL to use the same origin
        html_content = html_content.replace(
            string.Template("const API_URL = `http://${window.location.hostname}:$port`;").safe_substitute(
                port=chat_settings.port
            ),
            "const API_URL = '';",
        )
        return responses.HTMLResponse(content=html_content)

    @app.get("/logo.svg")
    async def logo() -> responses.FileResponse:
        """Serve the logo for favicon and header."""
        logo_path = pathlib.Path(__file__).parent / "logo.svg"
        return responses.FileResponse(logo_path, media_type="image/svg+xml")

    @app.get("/health")
    async def health() -> HealthResponse:
        """Health check endpoint"""
        return HealthResponse(
            status="ok",
            ready=pool.is_ready,
            num_gpus=pool.num_gpus,
            available_workers=pool.num_available,
        )

    @app.get("/stats")
    async def stats() -> StatsResponse:
        """Get worker pool statistics."""
        return StatsResponse(
            total_workers=pool.num_workers,
            available_workers=pool.num_available,
            busy_workers=pool.num_busy,
            workers=[GPUInfo(gpu_id=w.gpu_id, device=str(w.device)) for w in list(pool.workers)],
        )

    @app.post("/chat/completions")
    async def chat_completions(
        request: ChatRequest, worker: Annotated[Worker, fastapi.Depends(get_worker)]
    ) -> responses.StreamingResponse:
        """Chat completions endpoint that streams responses as they are generated."""

        # log incoming conversation to console
        logger.info("=" * 20)
        for message in request.messages:
            logger.info("[%s]: %s", message.role.upper(), message.content)
        logger.info("-" * 20)

        # build conversation tokens
        chat_sm = ChatWebStateMachine(worker.engine, worker.autocast_ctx, request.messages)
        response_chunks: list[str] = []

        async def format_and_log() -> AsyncGenerator[str]:
            """
            Formats chunks from the state machine into nanochat SSE format and logs the full response after generation
            is done.
            """
            try:
                async for chunk in chat_sm.generate_stream(
                    max_tokens=request.max_tokens if request.max_tokens is not None else chat_settings.max_tokens,
                    temperature=request.temperature if request.temperature is not None else chat_settings.temperature,
                    top_k=request.top_k if request.top_k is not None else chat_settings.top_k,
                ):
                    match chunk:
                        case AssistantResponseChunk():
                            # accumulate response for logging
                            response_chunks.append(chunk.text)
                            yield f"data: {json.dumps({'token': chunk.text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"  # noqa: E501
                        case AssistantResponseDone():
                            yield f"data: {json.dumps({'done': True})}\n\n"
                        case _:
                            assert_never(chunk)
            finally:
                # log the assistant response to console
                logger.info("[ASSISTANT] (GPU %d): %s", worker.gpu_id, "".join(response_chunks))
                logger.info("=" * 20)

        return responses.StreamingResponse(format_and_log(), media_type="text/event-stream")

    logger.info(
        "temperature: %f, top-k: %d, max tokens: %d",
        chat_settings.temperature,
        chat_settings.top_k,
        chat_settings.max_tokens,
    )
    return app


def chat_web(chat_settings: settings.ChatWeb, model_settings: settings.GPT2) -> None:
    """Launch the FastAPI web server for Peashooter chat."""
    app = create_app(chat_settings, model_settings)
    uvicorn.run(app, host=str(chat_settings.host), port=chat_settings.port)
