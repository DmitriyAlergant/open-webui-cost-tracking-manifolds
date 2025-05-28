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
import sys,os

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
        "litellm_model_id": "gemini/gemini-2.5-flash-preview-05-20",
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
            default="https://generativelanguage.googleapis.com",
            description="Base URL for OpenAI-compatible API endpoint",
        )
        API_KEY: str = Field(
            default="",
            description="Google Gemini API Key",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "gemini"
        self.name = "gemini/"
        self.valves = self.Valves()
        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

    def get_litellm_pipe(self, full_model_id=None, url_model_id=None, provider=None):
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
            os.environ["GEMINI_API_KEY"] = self.valves.API_KEY
        if self.valves.API_BASE_URL:
            #litellm_settings["base_url"] = self.valves.API_BASE_URL
            pass
        
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

        # Allow disabling thinking for Gemini 2.5 flash models, and disalbe by default

        if "gemini-2.5-flash" in model_id_without_prefix:

            reasoning_effort = body.get("reasoning_effort")
            if reasoning_effort is None or str(reasoning_effort).lower() in ("none", "off", "disabled", "false", "no"):

                body["thinking"] = {"type": "disabled", "budget_tokens": 0}
                body.pop("reasoning_effort") if body.get("reasoning_effort") else None
                print ("Removing reasoning_effort from body")

            body["generate_thinking_block"] = False # either thinking is dsiabled, or it is enabled but LiteLLM currently has a bug and will stream thoughts as normal text - so no block needed

        if "gemini-2.5-pro" in model_id_without_prefix:

            if body.get("reasoning_effort") and body.get("reasoning_effort") in ("low", "medium", "high"):
                body["generate_thinking_block"] = False # LiteLLM currently has a bug and will stream thoughts as normal text. User will see thoughts. No block needed
            else:
                body["generate_thinking_block"] = True # Gemini 2.5 Pro will still be thinking but LiteLLM will not be returning thoughts 
                body.pop("reasoning_effort") if body.get("reasoning_effort") else None
                print ("Removing reasoning_effort from body")

        if self.valves.DEBUG and "gemini" in model_id_without_prefix:
            print("generate_thinking_block: ", body["generate_thinking_block"])       
            print("reasoning_effort: ", body.get("reasoning_effort"))    
        
        url_model_id = model_config["litellm_model_id"].split("/", 1)[1] if "/" in model_config["litellm_model_id"] else model_config["litellm_model_id"]

        return await self.get_litellm_pipe(full_model_id=full_model_id, url_model_id=url_model_id, provider="gemini") \
                            .chat_completion(
                                    body=body,
                                    __user__=__user__,
                                    __metadata__=__metadata__,
                                    __event_emitter__=__event_emitter__,
                                    __task__=__task__
                                )