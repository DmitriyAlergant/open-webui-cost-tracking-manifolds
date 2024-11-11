"""
title: Databricks Manifold Pipe with Cost Tracking
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
        DATABRICKS_SERVING_ENDPOINTS_BASE_URL: str = Field(
            default="https://adb-..../serving-endpoints",
            description="The base URL for Databricks Serving Endpoints.",
        )
        DATABRICKS_API_KEY: str = Field(
            default="",
            description="API Key (typically, PAT token)",
        )

        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "databricks"
        self.name = "databricks/"

        self.valves = self.Valves()

        self.debug_logging_prefix = "DEBUG:    " + __name__ + " -"

    def get_openai_pipe(self):

        module_name = MODULE_OPENAI_COMPATIBLE_PIPE

        if module_name not in sys.modules:
            raise Exception(f"Module {module_name} is not loaded")
        
        module = sys.modules[module_name]

        return module.OpenAIPipe (
                debug=self.valves.DEBUG,
                debug_logging_prefix=self.debug_logging_prefix, 
                api_base_url=self.valves.DATABRICKS_SERVING_ENDPOINTS_BASE_URL,
                api_key=self.valves.DATABRICKS_API_KEY)

    def pipes(self):
        return [
            {
                "id": "databricks-meta-llama-3-1-70b-instruct",
                "name": "databricks-meta-llama-3-1-70b-instruct",
            },
            {
                "id": "databricks-meta-llama-3-1-405b-instruct",
                "name": "databricks-meta-llama-3-1-405b-instruct",
            },
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:
        
        return await self.get_openai_pipe().chat_completion(
            body=body,
            __user__=__user__,
            __event_emitter__=__event_emitter__,
            __task__=__task__)