"""
title: OpenAI Manifold Pipe with Cost Tracking
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field

from typing import Union, List, Any, Awaitable, Callable

import sys
import asyncio
import time
import json

from fastapi.responses import StreamingResponse

from open_webui.utils.misc import get_messages_content

from openai import OpenAI, AsyncOpenAI


MODULE_USAGE_TRACKING = "function_module_usage_tracking"


class OpenAIPipe:

    def __init__(self, debug=False, debug_logging_prefix="", api_base_url="", api_key=""):

        self.debug = debug
        self.debug_logging_prefix = debug_logging_prefix

        self.api_base_url = api_base_url
        self.api_key = api_key

        self.sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
            )
        
        self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
            )

    def get_models(self):

        try:
            models = []

            for model in self.sync_client.models.list():
                models.append(model)

            return [
                {
                    "id": model.id,
                    "name": model.id,
                }
                for model in models
            ]

        except Exception as e:
            _, _, tb = sys.exc_info()
            print(f"OpenAI API Shared Module error on line {tb.tb_lineno}: {e}")

            return [ ]

    async def chat_completion(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        # Initialize CostTrackingManager from "usage_tracking_util" function

        cost_tracker_module_name = MODULE_USAGE_TRACKING

        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is not loaded")
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            body["model"], __user__, __task__, debug=self.debug
        )

        model_id = body["model"][body["model"].find(".") + 1 :]

        payload = {**body, "model": model_id}

        # Calculate Input Tokens (Estimated)
        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(
            body["messages"]
        )

        if self.debug:
            print(f"{self.debug_logging_prefix} Request Payload: {payload}")

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


            if body.get("stream", False):
                # STREAMING REQUEST

                if self.debug:
                    print(f"{self.debug_logging_prefix} returning streaming response")

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    last_update_time = 0
                    stream_completed = False

                    try:

                        response = await self.async_client.chat.completions.create(
                            **payload
                        )

                        async for chunk in response:

                            #OpenAI API is not supposed to have that, but some proxy providers encounter this
                            if len(chunk.choices)==0:
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
                        if self.debug:
                            print(
                                f"{self.debug_logging_prefix} stream_response wrapper... aborted by client"
                            )
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

                        if self.debug:
                            print(
                                f"{self.debug_logging_prefix} Finalized stream (completed: {stream_completed})"
                            )

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # BATCH REQUEST

                response = await self.async_client.chat.completions.create(**payload)

                response_json = response.to_dict()

                if self.debug:
                    print(
                        f"{self.debug_logging_prefix} returning batch response: {response_json}"
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
            print(f"OpenAI API Shared Module error on line {line_number}: {e}")
            raise e



# For OpenWebUI to accept this as a Function Module, there has to be a Filter or Pipe or Action class
# Pipe class with empty pipes list is the option with minimal impact

class Pipe:
    def __init__(self):
        self.type = "manifold"
        self.id = "openai_api_shared"
        self.name = "OpenAI API Shared Module"
        
        pass

    def pipes(self) -> list[dict]:
        return []