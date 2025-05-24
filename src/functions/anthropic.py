"""
title: LiteLLM Manifold (Anthropic)
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
import os

MODULE_LITELLM_PIPE = "function_module_litellm_pipe"

AVAILABLE_MODELS = [
    {
        "id": "claude-opus-4-20250514",
        "litellm_model_id": "anthropic/claude-opus-4-20250514",
        "name": "Claude 4 Opus",
        "max_output_tokens": 32000,
        "supports_extended_thinking": True,
    },
    {
        "id": "claude-opus-4-20250514-websearch",
        "litellm_model_id": "anthropic/claude-opus-4-20250514",
        "name": "Claude 4 Opus with Web Search",
        "max_output_tokens": 32000,
        "supports_extended_thinking": True,
        "additional_body_params": {
            "web_search_options": {
                "search_context_size": "medium"
            }
        },
    },
    {
        "id": "claude-sonnet-4-20250514",
        "litellm_model_id": "anthropic/claude-sonnet-4-20250514",
        "name": "Claude 4 Sonnet",
        "max_output_tokens": 64000,
        "supports_extended_thinking": True,
    },
    {
        "id": "claude-sonnet-4-20250514-websearch",
        "litellm_model_id": "anthropic/claude-sonnet-4-20250514",
        "name": "Claude 4 Sonnet with Web Search",
        "max_output_tokens": 64000,
        "supports_extended_thinking": True,
        "additional_body_params": {
            "web_search_options": {
                "search_context_size": "medium"
            }
        },
    },
    {
        "id": "claude-3-7-sonnet-20250219",
        "litellm_model_id": "anthropic/claude-3-7-sonnet-20250219",
        "name": "Claude 3.7 Sonnet",
        "max_output_tokens": 64000,
        "supports_extended_thinking": True,
    },
    {
        "id": "claude-3-5-sonnet-20241022",
        "litellm_model_id": "anthropic/claude-3-5-sonnet-20241022",
        "name": "Claude 3.5 Sonnet",
        "max_output_tokens": 8192,
        "supports_extended_thinking": False,
    },
    {
        "id": "claude-3-5-haiku-20241022",
        "litellm_model_id": "anthropic/claude-3-5-haiku-20241022",
        "name": "Claude 3.5 Haiku",
        "max_output_tokens": 8192,
        "supports_extended_thinking": False,
    }
]


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="API key for Anthropic models.",
        )

        ANTHROPIC_API_BASE: str = Field(
            default="https://api.anthropic.com",
            description="API base for Anthropic models.",
        )

        DEBUG: bool = Field(default=False, description="Display debugging messages")

        LITELLM_SETTINGS: dict = Field(
            default={},
            description="LiteLLM provider-specific settings (e.g. API base, versions, extra headers)"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves()
        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

        # Set environment variables for LiteLLM
        self._update_environment_variables()

    def _update_environment_variables(self):
        """Update environment variables for LiteLLM from valves"""
        if self.valves.ANTHROPIC_API_KEY:
            os.environ["ANTHROPIC_API_KEY"] = self.valves.ANTHROPIC_API_KEY

        if self.valves.ANTHROPIC_API_BASE:
            os.environ["ANTHROPIC_API_BASE"] = self.valves.ANTHROPIC_API_BASE

        # Set environment variables from LITELLM_SETTINGS
        for key, value in self.valves.LITELLM_SETTINGS.items():
            if key.endswith("_API_KEY") or key.endswith("_API_BASE") or key.endswith("_API_VERSION"):
                if os.environ.get(key) != value:
                    os.environ[key] = str(value)
                    if self.valves.DEBUG:
                        print(f"{self.debug_logging_prefix} Set/Updated env var from LITELLM_SETTINGS: {key}")

    def get_litellm_pipe(self):
        module_name = MODULE_LITELLM_PIPE
        if module_name not in sys.modules:
            try:
                __import__(module_name)
            except ImportError as e:
                print(f"Failed to import {module_name}: {e}")
                raise Exception(f"Module {module_name} is not loaded and could not be imported.")
        
        module = sys.modules[module_name]
        
        return module.LiteLLMPipe(
            debug=self.valves.DEBUG,
            debug_logging_prefix=self.debug_logging_prefix,
            litellm_settings=self.valves.LITELLM_SETTINGS
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

        # Update environment variables in case valves were changed
        self._update_environment_variables()

        # Get the model ID from the request
        full_model_id = body.get("model", "")
        
        # Extract the model name without the manifold prefix
        model_name_without_prefix = full_model_id.split(f"{self.id}.", 1)[1] if f"{self.id}." in full_model_id else full_model_id
        
        # Find the model definition to get the actual LiteLLM model ID and settings
        model_def = None
        
        for model_def_candidate in AVAILABLE_MODELS:
            if model_def_candidate.get("id") == model_name_without_prefix:
                model_def = model_def_candidate

        if model_def is None:
            raise ValueError(f"Model {model_name_without_prefix} not found in AVAILABLE_MODELS")
    

        body["model"] = model_def.get("litellm_model_id")

        body["stream_options"] = {"include_usage": True}


        # remove "reasoning_effort" from body
        reasoning_effort = body.pop("reasoning_effort",  model_def.get("reasoning_effort", None))

        thinking_params = {}
        max_tokens_override = body.get("max_tokens", 4096)      # default max tokens
        generate_thinking_block = False

        if model_def.get("supports_extended_thinking"):
            if reasoning_effort == "low":
                thinking_params = {"type": "enabled", "budget_tokens": 2048}
                max_tokens_override = min(max(body.get("max_tokens", 0), 8192), model_def.get("max_output_tokens"))          # At least 8k tokens (and 4k reasoning budget)
                generate_thinking_block = True

            elif reasoning_effort == "medium":
                thinking_params = {"type": "enabled", "budget_tokens": 8192}
                max_tokens_override = min(max(body.get("max_tokens", 0), 24000), model_def.get("max_output_tokens"))         # At least 24k tokens (and 16k reasoning budget)
                generate_thinking_block = True

            elif reasoning_effort == "high":
                thinking_params = {"type": "enabled", "budget_tokens": 24000}
                max_tokens_override = min(max(body.get("max_tokens", 0), 32000), model_def.get("max_output_tokens"))         # At least 32k tokens (and 24k reasoning budget)
                generate_thinking_block = True
        
            elif not reasoning_effort or reasoning_effort == "none" or reasoning_effort == "off" or reasoning_effort == "disabled" or reasoning_effort == "false" or reasoning_effort == "no":
                thinking_params = {"type": "disabled"}
                generate_thinking_block = False
            else:
                raise ValueError(f"Unrecognized reasoning_effort: {reasoning_effort}")


        if max_tokens_override:
            body["max_tokens"] = max_tokens_override

        if model_def.get("supports_extended_thinking") and thinking_params:
            body["thinking"] = thinking_params
            body["allowed_openai_params"]=['thinking']

        body["generate_thinking_block"] = generate_thinking_block

        # Merge additional_body_params from model definition into request body
        additional_params = model_def.get("additional_body_params", {})
        if additional_params:
            for key, value in additional_params.items():
                if key not in body:  # Don't override if already present in request
                    body[key] = value
                    if self.valves.DEBUG:
                        print(f"{self.debug_logging_prefix} Added additional_body_param: {key}={value}")

        if self.valves.DEBUG:
            print(f"{self.debug_logging_prefix} Calling LiteLLM pipe with payload: {body}")

        return await self.get_litellm_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__,
        ) 