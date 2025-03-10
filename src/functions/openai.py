"""
title: OpenAI Manifold
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from typing import Union, Any, Awaitable, Callable
import sys

MODULE_OPENAI_COMPATIBLE_PIPE = "function_module_openai_compatible_pipe"

AVAILABLE_MODELS = [
    {"id": "o1", "name": "o1"},
    {"id": "o1-preview", "name": "o1-preview"},
    {
        "id": "o3-mini-low",
        "name": "o3-mini (Low Effort)",
        "base_model": "o3-mini",
        "reasoning_effort": "low",
    },
    {
        "id": "o3-mini-medium",
        "name": "o3-mini (Medium Effort)",
        "base_model": "o3-mini",
        "reasoning_effort": "medium",
    },
    {
        "id": "o3-mini-high",
        "name": "o3-mini (High Effort)",
        "base_model": "o3-mini",
        "reasoning_effort": "high",
    },
    {"id": "o1-mini", "name": "o1-mini"},
    {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
    {"id": "gpt-4o", "name": "gpt-4o"},
    {"id": "chatgpt-4o-latest", "name": "chatgpt-4o-latest"},
    {"id": "gpt-4o-2024-05-13", "name": "gpt-4o-2024-05-13"},
    {"id": "gpt-4o-2024-08-06", "name": "gpt-4o-2024-08-06"},
    {"id": "gpt-4o-2024-11-20", "name": "gpt-4o-2024-11-20"},
    {"id": "gpt-4-turbo", "name": "gpt-4-turbo"},
    {"id": "gpt-4.5-preview", "name": "gpt-4.5-preview"},
]

DEFAULT_ENABLED_MODELS = ",".join([model["id"] for model in AVAILABLE_MODELS])


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="The base URL for OpenAI API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="Required API key to retrieve the model list.",
        )
        ENABLED_MODELS: str = Field(
            default=DEFAULT_ENABLED_MODELS,
            description="List of enabled model IDs",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "openai"
        self.name = "openai/"
        self.valves = self.Valves()
        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

    def get_openai_pipe(self):
        module_name = MODULE_OPENAI_COMPATIBLE_PIPE
        if module_name not in sys.modules:
            raise Exception(f"Module {module_name} is not loaded")

        module = sys.modules[module_name]
        return module.OpenAIPipe(
            debug=self.valves.DEBUG,
            debug_logging_prefix=self.debug_logging_prefix,
            api_base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )

    def pipes(self):
        enabled_models = [
            model
            for model in AVAILABLE_MODELS
            if model["id"] in self.valves.ENABLED_MODELS.split(",")
        ]
        return enabled_models

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        # Find model configuration
        model_id = body["model"].split(".")[-1]

        model_config = next(
            (model for model in self.pipes() if model["id"] == model_id), None
        )

        if model_config and "base_model" in model_config:

            if "reasoning_effort" in model_config:
                body["reasoning_effort"] = model_config["reasoning_effort"]
            body["model"] = model_config["base_model"]

        if "o1" in body["model"] and body["messages"][0]["role"] == "system":
            print(
                "OpenAI Manifold: o1 models do not currently support System message. Converting System Prompt to User role."
            )
            body["messages"][0]["role"] = "user"

        return await self.get_openai_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__,
        )
