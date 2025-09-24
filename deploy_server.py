import importlib
import inspect

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.usage.usage_lib import UsageContext
import vllm.entrypoints.openai.api_server
from vllm.entrypoints.openai.api_server import (
    lifespan,
    TIMEOUT_KEEP_ALIVE,
    parse_args,
    app,
    openai_serving_chat,
    openai_serving_completion,
)


def spin_up(args):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    print("vLLM API server version %s", vllm.__version__)
    print("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    openai_serving_chat = OpenAIServingChat(
        engine,
        served_model_names,
        args.response_role,
        args.lora_modules,
        args.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names, args.lora_modules
    )

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    spin_up(args)

# @ray.remote
# class VllmServer:
#     def __init__(self):
#         self.model = None
#         self.port = None
#         self.server = None

#     def spin_up(self, model_name, port_number):
#         # to spin up a vllm api server, we can use command python -m vllm.entrypoints.openai.api_server --model model_name --port port_number


# @ray.remote
# class VllmGroup:
#     def __init__(self, num_of_servers):
#         self.num_of_servers = num_of_servers
#         self.servers = [VllmServer.remote() for _ in range(num_of_servers)]
