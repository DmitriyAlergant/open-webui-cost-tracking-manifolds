"""
title: OpenRouter Manifold
author: 
author_url: 
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
        "id": "gemini-2.5-pro",
        "litellm_model_id": "openrouter/google/gemini-2.5-pro",
        "name": "Gemini 2.5 Pro via OpenRouter",
        "generate_thinking_block": True
    },
    {
        "id": "gemini-2.5-flash",
        "litellm_model_id": "openrouter/google/gemini-2.5-flash",
        "name": "Gemini 2.5 Flash via OpenRouter",
        "generate_thinking_block": True
    },
    {
        "id": "x-ai/grok-4",
        "litellm_model_id": "openrouter/x-ai/grok-4",
        "name": "Grok 4",
        "generate_thinking_block": False
    },
    {
        "id": "deepseek/deepseek-r1",
        "litellm_model_id": "openrouter/deepseek/deepseek-r1",
        "name": "Deepseek R1 2005-01-20",
        "generate_thinking_block": True
    },
    {
        "id": "deepseek/deepseek-chat-v3-0324",
        "litellm_model_id": "openrouter/deepseek/deepseek-chat-v3-0324",
        "name": "Deepseek Chat v3 2025-03-24",
        "generate_thinking_block": False
    },
    {
        "id": "deepseek/deepseek-r1-0528",
        "litellm_model_id": "openrouter/deepseek/deepseek-r1-0528",
        "name": "Deepseek R1 2025-05-28",
        "generate_thinking_block": True
    }
]

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
        
        # Prepare isolated LiteLLM settings that won't affect other parts of the system
        litellm_settings = {}
        if self.valves.API_KEY:
            litellm_settings["api_key"] = self.valves.API_KEY
        if self.valves.API_BASE_URL:
            litellm_settings["base_url"] = self.valves.API_BASE_URL
        
        return module.LiteLLMPipe(
            debug=self.valves.DEBUG,
            debug_logging_prefix=self.debug_logging_prefix,
            litellm_settings=litellm_settings,
            full_model_id=full_model_id,
            provider=provider
        )

    def pipes(self):
        return [{"id": model["id"], "name": model["name"]} for model in AVAILABLE_MODELS]

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

        # Find the model configuration
        model_config = next((m for m in AVAILABLE_MODELS if m["id"] == model_id_without_prefix), None)
        
        if not model_config:
            raise ValueError(f"Model {model_id_without_prefix} not found in available models")

        # Update the body with the LiteLLM model ID
        body["model"] = model_config["litellm_model_id"]
        
        # Set generate_thinking_block based on model configuration
        body["generate_thinking_block"] = model_config.get("generate_thinking_block", False)

        if "stream" in body and body["stream"]:
            body["stream_options"] = {"include_usage": True}
        
        return await self.get_litellm_pipe(full_model_id=full_model_id, provider="openrouter").chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__
        )