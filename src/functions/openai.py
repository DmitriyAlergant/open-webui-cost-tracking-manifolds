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
        ENABLED_MODELS: str = Field(
            default="gpt-4o,gpt-4o-mini,gpt-4-turbo,o1-preview,o1-mini",
            description="List of enabled model IDs",
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

        return module.OpenAIPipe (
                debug=self.valves.DEBUG,
                debug_logging_prefix=self.debug_logging_prefix, 
                api_base_url=self.valves.OPENAI_API_BASE_URL,
                api_key=self.valves.OPENAI_API_KEY)

    def pipes(self):

        # models = self.get_openai_pipe().get_models()

        # Requesting models list from OpenAI works (worked) - see above; But introduced performance delays in the application.
        
        # A static hardcoded list is faster

        models = [
            {"id": "o1-mini", "name": "o1-mini"},
            {"id": "o1-preview", "name": "o1-preview"},
            {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
            {"id": "gpt-4o", "name": "gpt-4o"},
            {"id": "chatgpt-4o-latest", "name": "chatgpt-4o-latest"},
            {"id": "gpt-4o-2024-05-13", "name": "gpt-4o-2024-05-13"},
            {"id": "gpt-4o-2024-08-06", "name": "gpt-4o-2024-08-06"},
            {"id": "gpt-4o-2024-11-20", "name": "gpt-4o-2024-11-20"},
            {"id": "gpt-4-turbo", "name": "gpt-4-turbo"}  
        ]

        enabled_models = [
            model
            for model in models
            if model["id"] in self.valves.ENABLED_MODELS.split(",")
        ]

        return enabled_models

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:
        
        if "o1" in body["model"] and body["messages"][0]["role"] == 'system':

            print (f"OpenAI Manifold: o1 models do not currently support System message. Converting System Prompt to User role.")

            body["messages"][0]["role"] = 'user'
        
        return await self.get_openai_pipe().chat_completion(
            body=body,
            __user__ = __user__,
            __metadata__ = __metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__)