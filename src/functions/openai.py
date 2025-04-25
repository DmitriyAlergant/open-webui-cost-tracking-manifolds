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
    {"id": "o3", "name": "o3", "generate_thinking_block": True},
    {"id": "o1", "name": "o1", "generate_thinking_block": True},
    {"id": "o4-mini", "name": "o4-mini", "generate_thinking_block": True},
    {"id": "o3-mini", "name": "o3-mini", "generate_thinking_block": True},
    {"id": "o1-mini", "name": "o1-mini", "generate_thinking_block": True},
    {"id": "gpt-4o-mini", "name": "gpt-4o-mini", "generate_thinking_block": False},
    {"id": "gpt-4.1", "name": "gpt-4.1", "generate_thinking_block": False},
    {"id": "gpt-4.1-mini", "name": "gpt-4.1-mini", "generate_thinking_block": False},
    {"id": "gpt-4.1-nano", "name": "gpt-4.1-nano", "generate_thinking_block": False},
    {"id": "chatgpt-4o-latest", "name": "chatgpt-4o-latest", "generate_thinking_block": False},
    {"id": "gpt-4o", "name": "gpt-4o", "generate_thinking_block": False},
    {"id": "gpt-4o-2024-05-13", "name": "gpt-4o-2024-05-13", "generate_thinking_block": False},
    {"id": "gpt-4o-2024-08-06", "name": "gpt-4o-2024-08-06", "generate_thinking_block": False},
    {"id": "gpt-4o-2024-11-20", "name": "gpt-4o-2024-11-20", "generate_thinking_block": False},
    {"id": "gpt-4-turbo", "name": "gpt-4-turbo", "generate_thinking_block": False},
    {"id": "gpt-4.5-preview", "name": "gpt-4.5-preview", "generate_thinking_block": False},
]

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
        enabled_models = [ model for model in AVAILABLE_MODELS ]
        return enabled_models

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:
        
        # Retrieve the model ID suffix from the body
        full_model_id = body.get("model", "")
        model_id_suffix = full_model_id.split(".")[-1] # Get the part after the last period

        # Find the model in AVAILABLE_MODELS and get its generate_thinking_block flag
        generate_thinking_block = False # Default value
        for model_def in AVAILABLE_MODELS:
            if model_def.get("id") == model_id_suffix:
                generate_thinking_block = model_def.get("generate_thinking_block", False)
                break

        body["generate_thinking_block"] = generate_thinking_block

        # See https://community.openai.com/t/developer-role-not-accepted-for-o1-o1-mini-o3-mini/1110750/6
        if body["messages"][0]["role"] == "system" and ("o1-mini" in body["model"] or "o1-preview" in body["model"]):
            print(
                "OpenAI Manifold: this model do not currently support System message. Converting System Prompt to User role."
            )
            body["messages"][0]["role"] = "user"

        if body["stream"]:
            body["stream_options"] = {"include_usage": True}

        return await self.get_openai_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__,
        )
