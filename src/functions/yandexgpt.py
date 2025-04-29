"""
title: YandexGPT Manifold
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

import sys
import os
import json
import time
import asyncio
import aiohttp
from typing import Any, Awaitable, Callable, Union, List
from pydantic import BaseModel, Field
from open_webui.utils.misc import get_messages_content, pop_system_message
from fastapi.responses import StreamingResponse

MODULE_USAGE_TRACKING = "function_module_usage_tracking"

class Pipe:
    class Valves(BaseModel):
        YANDEX_API_KEY: str = Field(default="", description="Yandex API Key")
        YANDEX_CATALOG_ID: str = Field(default="", description="Yandex Catalog ID")
        PROXY_URL: str = Field(default="", description="Proxy URL if needed")
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "yandex"
        self.name = "yandex/"

        self.valves = self.Valves()

        self.debug_prefix = "DEBUG:    " + __name__ + " -"

    def pipes(self) -> List[dict]:
        return [
            {"id": "yandexgpt-rc", "name": "YandexGPT 5 (rc) Pro", "api-suffix":"/yandexgpt/rc"},        
            {"id": "yandexgpt-latest", "name": "YandexGPT 4 Pro", "api-suffix":"/yandexgpt/latest"},
            {"id": "yandexgpt-lite-rc", "name": "YandexGPT 5 (rc) Lite", "api-suffix":"/yandexgpt-lite/rc"},
            {"id": "yandexgpt-lite-latest", "name": "YandexGPT 4 Lite", "api-suffix":"/yandexgpt-lite/latest"}
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        # Initialize CostTrackingManager from "usage_tracking_util" module

        cost_tracker_module_name = MODULE_USAGE_TRACKING
        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is not loaded")
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            model=body["model"], 
            __user__=__user__, 
            __metadata__=__metadata__, 
            task=__task__, 
            debug=self.valves.DEBUG
        )

        # Process messages and calculate input tokens
        system_message, messages = pop_system_message(body["messages"])
        processed_messages = []
        for message in messages:
            processed_messages.append(
                {"role": message["role"], "text": message.get("content", "")}
            )
        if system_message:
            processed_messages.insert(
                0, {"role": "system", "text": str(system_message)}
            )

        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(
            body["messages"]
        )

        # Determine the correct model URI
        full_model_id = body["model"]
        # Strip the manifold ID prefix (e.g., "yandexgpt.") by splitting at the first '.'
        if '.' in full_model_id:
            selected_model_id = full_model_id.split('.', 1)[1]
        else:
            selected_model_id = full_model_id # Should not happen based on observed format, but handle defensively

        print(f"DEBUG: Full model ID from body: {full_model_id}")
        print(f"DEBUG: Selected model ID (stripped): {selected_model_id}")
        model_info = next((p for p in self.pipes() if p["id"] == selected_model_id), None)

        print(f"DEBUG: Model info: {model_info}")
        
        if not model_info or "api-suffix" not in model_info:
            # Raise error if model info or api-suffix is missing
            raise ValueError(f"Model info for '{selected_model_id}' (derived from '{full_model_id}') not found or missing 'api-suffix'. Please select a valid model from {self.pipes()}.")
        else:
            model_suffix = model_info["api-suffix"]
            
        model_uri = f"gpt://{self.valves.YANDEX_CATALOG_ID}{model_suffix}"

        payload = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": body.get("stream", False),
                "temperature": body.get("temperature", 0.7),
                "maxTokens": body.get("max_tokens", 4096),
            },
            "messages": processed_messages,
        }

        headers = {
            "Authorization": f"Api-Key {self.valves.YANDEX_API_KEY}",
            "x-folder-id": self.valves.YANDEX_CATALOG_ID,
            "Content-Type": "application/json",
        }

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

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

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} Payload: {payload}")

        try:
            if body.get("stream", False):
                # STREAMING REQUEST

                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} Returning streaming response")

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    last_update_time = 0
                    stream_completed = False
                    full_response = ""

                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                url, headers=headers, json=payload, proxy=self.valves.PROXY_URL
                            ) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    raise Exception(
                                        f"HTTP Error {response.status}: {error_text}"
                                    )
                                # Process the streaming content
                                async for line in response.content:
                                    if line:
                                        try:
                                            data = json.loads(line)
                                            chunk = data["result"]["alternatives"][0]["message"]["text"]

                                            # Only yield the new content
                                            new_content = chunk[len(full_response):]
                                            full_response = chunk

                                            if new_content:
                                                # Buffer content for costs calculation
                                                streamed_content_buffer += new_content

                                                # Return the chunk to the client
                                                # Emulate SSE stream (as in OpenAI code)
                                                content_json = {
                                                    "choices": [{"delta": {"content": new_content}, "index": 0}]
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
                                                    context_messages_count=len(messages)
                                                )

                                                streamed_content_buffer = ""

                                                last_update_time = current_time

                                        except json.JSONDecodeError:
                                            print(f"Failed to parse JSON: {line}")

                                stream_completed = True
                                yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        if self.valves.DEBUG:
                            print(
                                f"{self.debug_prefix} YandexGPT streaming... aborted by client"
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
                            context_messages_count=len(messages)
                        )
                        
                        if self.valves.DEBUG:
                            print(
                                f"{self.debug_prefix} Finalized stream (completed: {stream_completed})"
                            )


                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # BATCH REQUEST

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers, proxy=self.valves.PROXY_URL
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(
                                f"HTTP Error {response.status}: {error_text}"
                            )

                        response_json = await response.json()

                        if self.valves.DEBUG:
                            print(
                                f"{self.debug_prefix} Received response: {response_json}"
                            )

                        full_response = response_json["result"]["alternatives"][0][
                            "message"
                        ]["text"]

                        generated_tokens = cost_tracking_manager.count_tokens(
                            full_response
                        )

                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=input_tokens,
                            generated_tokens=generated_tokens,
                            reasoning_tokens=None,
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed",
                            persist_usage=True,
                        )

                        # Return the response in OpenAI-like format
                        return {
                            "choices": [
                                {"message": {"content": full_response}, "index": 0}
                            ]
                        }

        except Exception as e:
            _, _, tb = sys.exc_info()
            line_number = tb.tb_lineno
            print(f"Error on line {line_number}: {e}")
            raise e
