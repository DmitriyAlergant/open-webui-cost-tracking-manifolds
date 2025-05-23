"""
title: LiteLLM Manifold
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

# For now, a static list. We can move this to JSON configuration later.
AVAILABLE_MODELS = [
    {
        "id": "anthropic/claude-sonnet-4-20250514", 
        "name": "claude-sonnet-4-20250514", 
        "generate_thinking_block": True,
        "additional_body_params": {
            "web_search_options": {
                "search_context_size": "medium"
            }
        },
    }
]

class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="API key for Anthropic models.",
        )
        OPENAI_API_KEY: str = Field( # For OpenAI models via LiteLLM
            default="",
            description="API key for OpenAI models (if used via LiteLLM).",
        )
        # We might add other provider keys here later, e.g., COHERE_API_KEY
        DEBUG: bool = Field(default=False, description="Display debugging messages")
        LITELLM_SETTINGS: dict = Field(default={}, description="LiteLLM provider-specific settings (e.g. API base, versions)")


    def __init__(self):
        self.type = "manifold"
        self.id = "litellm" # This will be the prefix for model IDs, e.g., litellm.claude-3-opus
        self.name = "litellm/"
        self.valves = self.Valves()
        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

        # Set API keys as environment variables for LiteLLM
        # This should ideally be done more dynamically if keys change during runtime,
        # but for now, setting them at initialization.
        if self.valves.ANTHROPIC_API_KEY:
            os.environ["ANTHROPIC_API_KEY"] = self.valves.ANTHROPIC_API_KEY
        if self.valves.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
        
        # TODO: Iterate over LITELLM_SETTINGS and set any other env vars from there
        # For example, if LITELLM_SETTINGS = {"COHERE_API_KEY": "...", "AZURE_API_KEY": "..."}
        # This allows for a more flexible way to add new provider keys without changing Valves.


    def get_litellm_pipe(self):
        module_name = MODULE_LITELLM_PIPE
        # Ensure the module is loaded. This basic check might need enhancement
        # if dynamic loading/reloading of the pipe module is required.
        if module_name not in sys.modules:
            # This basic import assumes the module is in the Python path.
            # OpenWebUI's plugin system should handle making it available.
            try:
                __import__(module_name)
            except ImportError as e:
                print(f"Failed to import {module_name}: {e}")
                raise Exception(f"Module {module_name} is not loaded and could not be imported.")
        
        if module_name not in sys.modules: # Check again after attempting import
             raise Exception(f"Module {module_name} is still not loaded after import attempt.")


        module = sys.modules[module_name]
        return module.LiteLLMPipe( # Assuming LiteLLMPipe class in the module
            debug=self.valves.DEBUG,
            debug_logging_prefix=self.debug_logging_prefix,
            # Pass any necessary config from Valves to the LiteLLMPipe constructor
            # For example, API keys if not solely relying on env vars, or base URLs
            litellm_settings=self.valves.LITELLM_SETTINGS
        )

    def pipes(self):
        # Return models with the manifold's ID as a prefix
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

        # Update API keys from valves, in case they were updated in OpenWebUI settings
        if self.valves.ANTHROPIC_API_KEY and os.environ.get("ANTHROPIC_API_KEY") != self.valves.ANTHROPIC_API_KEY:
            os.environ["ANTHROPIC_API_KEY"] = self.valves.ANTHROPIC_API_KEY
            if self.valves.DEBUG:
                print(f"{self.debug_logging_prefix} Updated ANTHROPIC_API_KEY from valves.")
        if self.valves.OPENAI_API_KEY and os.environ.get("OPENAI_API_KEY") != self.valves.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
            if self.valves.DEBUG:
                print(f"{self.debug_logging_prefix} Updated OPENAI_API_KEY from valves.")
        
        # Logic to handle LITELLM_SETTINGS and update os.environ accordingly
        # This ensures that any API keys or specific provider settings passed via LITELLM_SETTINGS
        # are available as environment variables for LiteLLM.
        for key, value in self.valves.LITELLM_SETTINGS.items():
            if key.endswith("_API_KEY") or key.endswith("_API_BASE") or key.endswith("_API_VERSION"): # Common patterns for LiteLLM env vars
                if os.environ.get(key) != value:
                    os.environ[key] = str(value) # Ensure value is a string for env var
                    if self.valves.DEBUG:
                        print(f"{self.debug_logging_prefix} Set/Updated env var from LITELLM_SETTINGS: {key}")


        # Retrieve the model ID suffix from the body
        full_model_id = body.get("model", "")
        # The model ID in the body will be like "litellm.claude-sonnet-4-20250514"
        # We need to extract the part after "litellm." and then lookup the original ID from AVAILABLE_MODELS
        model_name_without_prefix = full_model_id.split(f"{self.id}.", 1)[1] if f"{self.id}." in full_model_id else full_model_id
        
        # Find the model in AVAILABLE_MODELS by matching the "id" field to get the original "id" with provider prefix
        model_id_for_litellm = model_name_without_prefix  # Default fallback
        generate_thinking_block = False # Default value
        additional_body_params = {} # Default empty dict
        
        for model_def in AVAILABLE_MODELS:
            if model_def.get("id") == model_name_without_prefix:
                model_id_for_litellm = model_def.get("id")  # This should be like "anthropic/claude-sonnet-4-20250514"
                generate_thinking_block = model_def.get("generate_thinking_block", False)
                additional_body_params = model_def.get("additional_body_params", {})
                break

        print ("full_model_id: ", full_model_id)
        print ("model_name_without_prefix: ", model_name_without_prefix)
        print ("model_id_for_litellm: ", model_id_for_litellm)
        print ("generate_thinking_block: ", generate_thinking_block)
        print ("additional_body_params: ", additional_body_params)
        
        body["model"] = model_id_for_litellm # Update body with the actual model ID for LiteLLM
        body["generate_thinking_block"] = generate_thinking_block

        # Merge additional body parameters from model definition
        # Use a deep merge to handle nested dictionaries properly
        def deep_merge_dict(target, source):
            """Deep merge source dict into target dict"""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge_dict(target[key], value)
                else:
                    target[key] = value
        
        deep_merge_dict(body, additional_body_params)

        if body.get("stream", False) and "stream_options" not in body:
             # LiteLLM's acompletion with stream=True doesn't directly use OpenAI's stream_options for usage.
             # Usage with streaming in LiteLLM might be handled differently or might not be available in all provider responses.
             # We will need to handle this in the module_litellm_pipe.py
             pass


        # Pass relevant provider-specific params. LiteLLM passes non-OpenAI params automatically.
        # Example: "reasoning_effort" for Anthropic.
        # These could come from body if OpenWebUI frontend sends them, or be configured.
        # For now, we assume they are in 'body' if needed.
        
        # If 'reasoning_effort' is in the payload, LiteLLM should pass it to compatible models.
        # Example: body['reasoning_effort'] = "low" (user might set this via advanced model settings in UI)

        if self.valves.DEBUG:
            print(f"{self.debug_logging_prefix} Calling LiteLLM pipe with model: {model_id_for_litellm}, generate_thinking_block: {generate_thinking_block}")
            print(f"{self.debug_logging_prefix} Payload for LiteLLM pipe: {body}")

        return await self.get_litellm_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__,
        ) 