"""
title: DeepSeek Manifold
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
        "id": "deepseek-reasoner",
        "litellm_model_id": "deepseek/deepseek-reasoner",
        "name": "DeepSeek R1",
        "generate_thinking_block": True
    },
    {
        "id": "deepseek-chat",
        "litellm_model_id": "deepseek/deepseek-chat",
        "name": "DeepSeek V3",
        "generate_thinking_block": False
    },
]

class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.deepseek.com",
            description="The base URL for DeepSeek API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="Required API key to retrieve the model list.",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "deepseek"
        self.name = "deepseek/"
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
        if self.valves.OPENAI_API_KEY:
            litellm_settings["api_key"] = self.valves.OPENAI_API_KEY
        if self.valves.OPENAI_API_BASE_URL:
            litellm_settings["base_url"] = self.valves.OPENAI_API_BASE_URL
        
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
        
        full_model_id = body.get("model", "")

        model_name_without_prefix = full_model_id.split(f"{self.id}.", 1)[1] if f"{self.id}." in full_model_id else full_model_id

        model_def = None
        
        for model_def_candidate in AVAILABLE_MODELS:
            if model_def_candidate.get("id") == model_name_without_prefix:
                model_def = model_def_candidate
                break
        
        if not model_def:
            raise ValueError(f"Model {model_id} not found in available models")

        body["model"] = model_def["litellm_model_id"]
        
        body["generate_thinking_block"] = model_def.get("generate_thinking_block", False)

        if body["stream"]:
            body["stream_options"] = {"include_usage": True}

        return await self.get_litellm_pipe(full_model_id=full_model_id, provider="deepseek") \
                .chat_completion(
                        body=body,
                        __user__=__user__,
                        __metadata__=__metadata__,
                        __event_emitter__=__event_emitter__,
                        __task__=__task__
                    )
