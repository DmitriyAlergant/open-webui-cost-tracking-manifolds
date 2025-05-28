"""
title: LiteLLM Manifold (OpenAI)
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.6.9
license: MIT
"""

from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from typing import Union, Any, Awaitable, Callable
import sys

MODULE_LITELLM_PIPE = "function_module_litellm_pipe"

AVAILABLE_MODELS = [
    {
        "id": "o3",
        "litellm_model_id": "openai/o3",
        "name": "OpenAI o3",
        "generate_thinking_block": True
    },
    {
        "id": "o1",
        "litellm_model_id": "openai/o1",
        "name": "OpenAI o1",
        "generate_thinking_block": True
    },
    {
        "id": "o4-mini",
        "litellm_model_id": "openai/o4-mini",
        "name": "OpenAI o4-mini",
        "generate_thinking_block": True
    },
    {
        "id": "o3-mini",
        "litellm_model_id": "openai/o3-mini",
        "name": "OpenAI o3-mini",
        "generate_thinking_block": True
    },
    {
        "id": "o1-mini",
        "litellm_model_id": "openai/o1-mini",
        "name": "OpenAI o1-mini",
        "generate_thinking_block": True
    },
    {
        "id": "gpt-4o-mini",
        "litellm_model_id": "openai/gpt-4o-mini",
        "name": "OpenAI gpt-4o-mini",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4.1",
        "litellm_model_id": "openai/gpt-4.1",
        "name": "OpenAI gpt-4.1",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4.1-mini",
        "litellm_model_id": "openai/gpt-4.1-mini",
        "name": "OpenAI gpt-4.1-mini",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4.1-nano",
        "litellm_model_id": "openai/gpt-4.1-nano",
        "name": "OpenAI gpt-4.1-nano",
        "generate_thinking_block": False
    },
    {
        "id": "chatgpt-4o-latest",
        "litellm_model_id": "openai/chatgpt-4o-latest",
        "name": "OpenAI chatgpt-4o-latest",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4o",
        "litellm_model_id": "openai/gpt-4o",
        "name": "OpenAI gpt-4o",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4o-2024-05-13",
        "litellm_model_id": "openai/gpt-4o-2024-05-13",
        "name": "OpenAI gpt-4o-2024-05-13",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4o-2024-08-06",
        "litellm_model_id": "openai/gpt-4o-2024-08-06",
        "name": "OpenAI gpt-4o-2024-08-06",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4o-2024-11-20",
        "litellm_model_id": "openai/gpt-4o-2024-11-20",
        "name": "OpenAI gpt-4o-2024-11-20",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4-turbo",
        "litellm_model_id": "openai/gpt-4-turbo",
        "name": "OpenAI gpt-4-turbo",
        "generate_thinking_block": False
    },
    {
        "id": "gpt-4.5-preview",
        "litellm_model_id": "openai/gpt-4.5-preview",
        "name": "OpenAI gpt-4.5-preview",
        "generate_thinking_block": False
    },
]

class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default="",
            description="API key for OpenAI models.",
        )
        OPENAI_API_BASE: str = Field(
            default="https://api.openai.com/v1",
            description="The base URL for OpenAI API endpoints.",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")
        
        LITELLM_SETTINGS: dict = Field(
            default={},
            description="LiteLLM provider-specific settings (e.g. API base, versions, extra headers)"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "openai"
        self.name = "openai/"
        self.valves = self.Valves()
        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

    def get_litellm_pipe(self, full_model_id=None, provider=None):
        module_name = MODULE_LITELLM_PIPE
        if module_name not in sys.modules:
            try:
                __import__(module_name)
            except ImportError as e:
                print(f"Failed to import {module_name}: {e}")
                raise Exception(f"Module {module_name} is not loaded and could not be imported.")
        
        module = sys.modules[module_name]
        
        litellm_settings = self.valves.LITELLM_SETTINGS.copy()
        
        if self.valves.OPENAI_API_KEY:
            litellm_settings["api_key"] = self.valves.OPENAI_API_KEY
        
        if self.valves.OPENAI_API_BASE:
            litellm_settings["base_url"] = self.valves.OPENAI_API_BASE
        
        return module.LiteLLMPipe(
            debug=self.valves.DEBUG,
            debug_logging_prefix=self.debug_logging_prefix,
            litellm_settings=litellm_settings,
            full_model_id=full_model_id,
            provider=provider
        )

    def pipes(self):
        return [
            {**model}
            for model in AVAILABLE_MODELS
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:
        
        if self.valves.DEBUG:
            print(f"{self.debug_logging_prefix} LiteLLM Manifold received request body: {body}")

        full_model_id = body.get("model", "")
        
        model_name_without_prefix = full_model_id.split(f"{self.id}.", 1)[1] if f"{self.id}." in full_model_id else full_model_id
        
        model_def = None
        
        for model_def_candidate in AVAILABLE_MODELS:
            if model_def_candidate.get("id") == model_name_without_prefix:
                model_def = model_def_candidate
                break

        if model_def is None:
            raise ValueError(f"Model {model_name_without_prefix} not found in AVAILABLE_MODELS")

        body["model"] = model_def.get("litellm_model_id")

        generate_thinking_block = model_def.get("generate_thinking_block", False)
        body["generate_thinking_block"] = generate_thinking_block

        # See https://community.openai.com/t/developer-role-not-accepted-for-o1-o1-mini-o3-mini/1110750/6
        if body["messages"][0]["role"] == "system" and ("o1-mini" in body["model"] or "o1-preview" in body["model"]):
            body["messages"][0]["role"] = "user"

        if body["stream"]:
            body["stream_options"] = {"include_usage": True}

        if self.valves.DEBUG:
            print(f"{self.debug_logging_prefix} Calling LiteLLM pipe with payload: {body}")

        return await self.get_litellm_pipe(full_model_id=full_model_id, provider="openai") \
                .chat_completion(
                        body=body,
                        __user__=__user__,
                        __metadata__=__metadata__,
                        __event_emitter__=__event_emitter__,
                        __task__=__task__
                    )
