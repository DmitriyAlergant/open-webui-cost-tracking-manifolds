"""
title: OpenRouter Manifold
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


class Pipe:

    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="Base URL for OpenAI-compatible API endpoint",
        )
        API_KEY: str = Field(
            default="",
            description="API Key",
        )

        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "openrouter"
        self.name = "openrouter/"

        self.valves = self.Valves()

        self.debug_logging_prefix = "DEBUG:    " + __name__ + " -"

    def pipes(self):
        return [
            {
                "id": "google/gemini-2.5-pro-preview",
                "name": "gemini-2.5-pro-preview",
                "generate_thinking_block": True
            },
            {
                "id": "google/gemini-2.5-flash-preview",
                "name": "gemini-2.5-flash-preview",
                "generate_thinking_block": True
            },
            {
                "id": "google/gemini-2.5-pro-exp-03-25",
                "name": "gemini-2.5-pro-exp-free",
                "generate_thinking_block": True
            },
            {
                "id": "anthropic/claude-3.7-sonnet",
                "name": "claude-3.7-sonnet",
                "generate_thinking_block": False
            },
            {
                "id": "anthropic/claude-3.7-sonnet:thinking",
                "name": "claude-3.7-sonnet-thinking",
                "generate_thinking_block": True
            },
            {
                "id": "openai/gpt-4.1",
                "name": "gpt-4.1",
                "generate_thinking_block": False
            },
            {
                "id": "x-ai/grok-3-beta",
                "name": "grok-3-beta",
                "generate_thinking_block": False
            }
        ]

    def get_openai_pipe(self):

        module_name = MODULE_OPENAI_COMPATIBLE_PIPE

        if module_name not in sys.modules:
            raise Exception(f"Module {module_name} is not loaded")
        
        module = sys.modules[module_name]

        return module.OpenAIPipe (
                debug=self.valves.DEBUG,
                debug_logging_prefix=self.debug_logging_prefix, 
                api_base_url=self.valves.API_BASE_URL,
                api_key=self.valves.API_KEY)

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
        model_id_without_prefix = full_model_id.split(".", 1)[1] if "." in full_model_id else full_model_id

        # Find the model in self.pipes() and get its generate_thinking_block flag
        generate_thinking_block = False 
        for model_def in self.pipes():
            if model_def.get("id") == model_id_without_prefix:
                generate_thinking_block = model_def.get("generate_thinking_block", False)
                break

        body["generate_thinking_block"] = generate_thinking_block

        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
        
        return await self.get_openai_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__)