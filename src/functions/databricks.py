"""
title: Databricks Manifold Pipe with Cost Tracking
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field

from typing import Union, List

from open_webui.utils.misc import get_messages_content

from typing import Any, Awaitable, Callable

import sys
import asyncio
import time
import json

from fastapi.responses import StreamingResponse

from openai import OpenAI, AsyncOpenAI


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

        self.debug_prefix = "DEBUG:    " + __name__ + " -"

    def init_openai_client(self, async_client=False):

        if not self.valves.DATABRICKS_API_KEY:
            raise Exception(
                f"Databricks Manifold function: DATABRICKS_API_KEY valve not provided"
            )

        if async_client:
            return AsyncOpenAI(
                api_key=self.valves.DATABRICKS_API_KEY,
                base_url=self.valves.DATABRICKS_SERVING_ENDPOINTS_BASE_URL,
            )

        else:
            return OpenAI(
                api_key=self.valves.DATABRICKS_API_KEY,
                base_url=self.valves.DATABRICKS_SERVING_ENDPOINTS_BASE_URL,
            )

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

        # Initialize CostTrackingManager from "usage_tracking_util" function

        cost_tracker_module_name = "function_usage_tracking_util"
        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is not loaded")
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            body["model"], __user__, __task__, debug=self.valves.DEBUG
        )

        model_id = body["model"][body["model"].find(".") + 1 :]

        payload = {**body, "model": model_id}

        # Calculate Input Tokens (Estimated)
        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(
            body["messages"]
        )

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} Request Payload: {payload}")

        try:
            start_time = time.time()

            cost_tracking_manager.calculate_costs_update_status_and_persist(
                input_tokens=input_tokens,
                generated_tokens=None,
                reasoning_tokens=None,
                start_time=start_time,
                __event_emitter__=__event_emitter__,
                status="Requested...",
                persist_usage=False,
            )

            # Init OpenAI API Client (Async)

            openai_client = self.init_openai_client(async_client=True)

            if body.get("stream", False):
                # STREAMING REQUEST

                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} returning streaming response")

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    last_update_time = 0
                    stream_completed = False

                    try:

                        response = await openai_client.chat.completions.create(
                            **payload
                        )

                        async for chunk in response:

                            # OpenAI API is not supposed to have that, but some proxy providers encounter this
                            if len(chunk.choices) == 0:
                                continue

                            content = chunk.choices[0].delta.content or ""

                            # Buffer content for costs calculation (once a second)

                            streamed_content_buffer += content

                            # Return emulating an SSE stream

                            content_json = {
                                "choices": [{"index": 0, "delta": {"content": content}}]
                            }
                            yield f"data: {json.dumps(content_json)}\n\n"

                            # Update status every ~1 second

                            current_time = time.time()
                            if current_time - last_update_time >= 1:

                                generated_tokens += cost_tracking_manager.count_tokens(
                                    streamed_content_buffer
                                )

                                cost_tracking_manager.calculate_costs_update_status_and_persist(
                                    input_tokens=input_tokens,
                                    generated_tokens=generated_tokens,
                                    reasoning_tokens=None,
                                    start_time=start_time,
                                    __event_emitter__=__event_emitter__,
                                    status="Streaming...",
                                    persist_usage=False,
                                )

                                streamed_content_buffer = ""

                                last_update_time = current_time

                        stream_completed = True
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        if self.valves.DEBUG:
                            print("DEBUG stream_response wrapper... aborted by client")
                    except Exception as e:
                        _, _, tb = sys.exc_info()
                        line_number = tb.tb_lineno
                        print(f"Error on line {line_number}: {e}")
                        raise e

                    finally:
                        generated_tokens += cost_tracking_manager.count_tokens(
                            streamed_content_buffer
                        )

                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=input_tokens,
                            generated_tokens=generated_tokens,
                            reasoning_tokens=None,
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed else "Stopped",
                            persist_usage=True,
                        )

                        if self.valves.DEBUG:
                            print(
                                f"DEBUG Finalized stream (completed: {stream_completed})"
                            )

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # BATCH REQUEST

                response = await openai_client.chat.completions.create(**payload)

                response_json = response.to_dict()

                if self.valves.DEBUG:
                    print(
                        f"{self.debug_prefix} returning batch response: {response_json}"
                    )

                input_tokens = response_json.get("usage", {}).get(
                    "prompt_tokens", input_tokens
                )
                generated_tokens = response_json.get("usage", {}).get(
                    "completion_tokens", 0
                )
                reasoning_tokens = (
                    response_json.get("usage", {})
                    .get("completion_tokens_details", {})
                    .get("reasoning_tokens", 0)
                )

                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=reasoning_tokens,
                    start_time=start_time,
                    __event_emitter__=__event_emitter__,
                    status="Completed",
                    persist_usage=True,
                )

                return response_json

        except Exception as e:
            _, _, tb = sys.exc_info()
            line_number = tb.tb_lineno
            print(f"Error on line {line_number}: {e}")
            raise e
