"""
title: Google Gemini Manifold
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
        "id": "gemini-2.5-pro",
        "litellm_model_id": "gemini/gemini-2.5-pro-preview-05-06",
        "name": "Gemini 2.5 Pro",
        "generate_thinking_block": True
    },
    {
        "id": "gemini-2.5-flash",
        "litellm_model_id": "gemini/gemini-2.5-pro-preview-05-06",
        "name": "Gemini 2.5 Flash",
        "generate_thinking_block": True
    },
    {
        "id": "gemini-2.5-pro-exp",
        "litellm_model_id": "gemini/gemini-2.5-pro-exp-03-25",
        "name": "Gemini 2.5 Pro (free)",
        "generate_thinking_block": True
    },
]

class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com/v1beta/",
            description="Base URL for OpenAI-compatible API endpoint",
        )
        API_KEY: str = Field(
            default="",
            description="Google Gemini API Key",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "google"
        self.name = "google/"
        self.valves = self.Valves()
        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

    def get_litellm_pipe(self):
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

        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
        
        return await self.get_litellm_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__,
        )