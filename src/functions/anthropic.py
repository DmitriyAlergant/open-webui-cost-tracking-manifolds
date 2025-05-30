"""
title: LiteLLM Manifold (Anthropic)
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

        ENABLE_CACHE_WRITING: bool = Field(
            default=False, 
            description="Enable Anthropic cache control on system message and last two user messages"
        )

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

    def get_litellm_pipe(self, full_model_id=None, provider=None):
        module_name = MODULE_LITELLM_PIPE
        if module_name not in sys.modules:
            try:
                __import__(module_name)
            except ImportError as e:
                print(f"Failed to import {module_name}: {e}")
                raise Exception(f"Module {module_name} is not loaded and could not be imported.")
        
        module = sys.modules[module_name]
        
        # Prepare isolated LiteLLM settings that include API credentials
        isolated_litellm_settings = self.valves.LITELLM_SETTINGS.copy()
        
        # Add API credentials directly to the settings if provided
        if self.valves.ANTHROPIC_API_KEY:
            isolated_litellm_settings["api_key"] = self.valves.ANTHROPIC_API_KEY
        
        if self.valves.ANTHROPIC_API_BASE:
            isolated_litellm_settings["base_url"] = self.valves.ANTHROPIC_API_BASE
        
        return module.LiteLLMPipe(
            debug=self.valves.DEBUG,
            debug_logging_prefix=self.debug_logging_prefix,
            litellm_settings=isolated_litellm_settings,
            full_model_id=full_model_id,
            provider=provider
        )

    def _add_cache_control_to_message(self, message: dict, message_identifier: str):
        """Add cache control to the last content block of a message"""
        content = message.get("content")
        
        if isinstance(content, str):
            # Convert string to content blocks format
            content = [{"type": "text", "text": content}]
            message["content"] = content
        
        if isinstance(content, list) and content:
            last_content_block = content[-1]
            last_content_block["cache_control"] = {"type": "ephemeral"}
            if self.valves.DEBUG:
                print(f"{self.debug_logging_prefix} Added cache control to {message_identifier}")

    def _apply_cache_control(self, body: dict):
        """Apply Anthropic cache control to system message and last two user messages"""
        messages = body.get("messages", [])
        if not messages:
            return

        if self.valves.DEBUG:
            print(f"{self.debug_logging_prefix} Applying cache control to messages")

        # Apply cache control to system message (if exists)
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                self._add_cache_control_to_message(message, f"system message {i}")
                break  

        # Find last two user messages (iterate backwards)
        user_message_indices = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                user_message_indices.append(i)
                if len(user_message_indices) == 2:
                    break

        # Apply cache control to last content block of each user message
        for idx in user_message_indices:
            message = messages[idx]
            self._add_cache_control_to_message(message, f"user message {idx}")

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

        # Apply cache control if enabled
        if self.valves.ENABLE_CACHE_WRITING:
            self._apply_cache_control(body)

        if self.valves.DEBUG:
            print(f"{self.debug_logging_prefix} Calling LiteLLM pipe with payload: {body}")

        return await self.get_litellm_pipe(full_model_id=full_model_id, provider="anthropic") \
            .chat_completion(
                body=body,
                __user__=__user__,
                __metadata__=__metadata__,
                __event_emitter__=__event_emitter__,
                __task__=__task__
            ) 