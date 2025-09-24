import ray
import asyncio
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Set

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm

# import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.api_server import (
    TIMEOUT_KEEP_ALIVE,
    parse_args,
)

_running_tasks: Set[asyncio.Task[Any]] = set()


@asynccontextmanager
async def lifespan(
    app: fastapi.FastAPI, engine: AsyncLLMEngine, engine_args: AsyncEngineArgs
):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield

# @ray.remote(num_gpus=1)
class FastAPIApp:
    def __init__(self, args):
        self.args = args
        self.app = fastapi.FastAPI()
        # self.openai_serving_chat: OpenAIServingChat
        # self.openai_serving_completion: OpenAIServingCompletion
        self._setup_routes()
        self._setup_middleware()
        self._setup_exception_handlers()
        self._initialize_components()
        self.app.router.lifespan_context_manager = lifespan(
            self.app, self.engine, self.engine_args
        )

    async def validation_exception_handler(self, _, exc):
        err = self.openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    async def health(self):
        await self.openai_serving_chat.engine.check_health()
        return Response(status_code=200)

    async def show_available_models(self):
        models = await self.openai_serving_chat.show_available_models()
        return JSONResponse(content=models.model_dump())

    async def show_version(self):
        ver = {"version": vllm.__version__}
        return JSONResponse(content=ver)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return JSONResponse(content=generator.model_dump())

    def _setup_routes(self):
        # Define all your routes here
        self.app.get("/health")(self.health)
        self.app.get("/v1/models")(self.show_available_models)
        self.app.get("/version")(self.show_version)
        self.app.post("/v1/chat/completions")(self.create_chat_completion)
        self.app.post("/v1/completions")(self.create_completion)
        self.app.exception_handler(RequestValidationError)(
            self.validation_exception_handler
        )

        # Prometheus metrics route
        route = Mount("/metrics", make_asgi_app())
        route.path_regex = re.compile("^/metrics(?P<path>.*)$")
        self.app.routes.append(route)

    def _setup_middleware(self):
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.args.allowed_origins,
            allow_credentials=self.args.allow_credentials,
            allow_methods=self.args.allowed_methods,
            allow_headers=self.args.allowed_headers,
        )

        # Dynamic middleware loading
        for middleware in self.args.middleware:
            module_path, object_name = middleware.rsplit(".", 1)
            imported = getattr(importlib.import_module(module_path), object_name)
            if inspect.isclass(imported):
                self.app.add_middleware(imported)
            elif inspect.iscoroutinefunction(imported):
                self.app.middleware("http")(imported)
            else:
                raise ValueError(
                    f"Invalid middleware {middleware}. Must be a function or a class."
                )

    def _setup_exception_handlers(self):
        self.app.exception_handler(RequestValidationError)(
            self.validation_exception_handler
        )

    def _initialize_components(self):
        self.logger = init_logger(__name__)
        self.engine_args = AsyncEngineArgs.from_cli_args(self.args)
        self.engine = AsyncLLMEngine.from_engine_args(
            self.engine_args, usage_context=UsageContext.OPENAI_API_SERVER
        )
        if args.served_model_name is not None:
            self.served_model_names = args.served_model_name
        else:
            self.served_model_names = [args.model]

        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            self.served_model_names,
            self.args.response_role,
            self.args.lora_modules,
            self.args.chat_template,
        )
        self.openai_serving_completion = OpenAIServingCompletion(
            self.engine, self.served_model_names, self.args.lora_modules
        )

    def run(self):
        uvicorn.run(
            self.app,
            host=self.args.host,
            port=self.args.port,
            log_level=self.args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self.args.ssl_keyfile,
            ssl_certfile=self.args.ssl_certfile,
            ssl_ca_certs=self.args.ssl_ca_certs,
            ssl_cert_reqs=self.args.ssl_cert_reqs,
        )


if __name__ == "__main__":
    args = parse_args()
    # app_instance = FastAPIApp.remote(args)
    # ray.get(app_instance.run.remote())
    app_instance = FastAPIApp(args)
    app_instance.run()
