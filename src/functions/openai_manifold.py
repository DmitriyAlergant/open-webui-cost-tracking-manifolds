"""
title: OpenAI Manifold Pipe
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.2
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator, AsyncGenerator

from open_webui.utils.misc import (
    get_last_user_message,
    get_last_assistant_message,
    get_messages_content,
)

from typing import Any, Awaitable, Callable, Optional

import os
import requests
import sys
import asyncio
import time
import json

import weakref
import atexit

import httpx

from fastapi.responses import StreamingResponse


from collections.abc import Iterator


class Pipe:

    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="openai/",
            description="The prefix applied before the model names.",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="The base URL for OpenAI API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="Required API key to retrieve the model list.",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "openai"
        self.name = "openai/"

        self.valves = self.Valves()

        self.debug_prefix = "DEBUG:    " + __name__ + " -"

        pass

    def pipes(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {}
                headers["Authorization"] = f"Bearer {self.valves.OPENAI_API_KEY}"
                headers["Content-Type"] = "application/json"

                r = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )

                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": f'{model["name"] if "name" in model else model["id"]}',
                    }
                    for model in models["data"]
                    if model["id"]
                    in (
                        "chatgpt-4o-latest",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "gpt-4o-audio-preview",
                        "gpt-4o-realtime-preview",
                        "o1-preview",
                        "o1-mini",
                    )
                ]

            except Exception as e:

                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from OpenAI, please update the API Key in the valves.",
                    },
                ]
        else:
            print(f"OpenAI Manifold function: OPENAI_API_KEY valve not provided")
            return [
                {
                    "id": "error",
                    "name": f"OpenAI Manifold function: OPENAI_API_KEY valve not provided",
                },
            ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        if not self.valves.OPENAI_API_KEY:
            raise Exception(
                f"OpenAI Manifold function: OPENAI_API_KEY valve not provided"
            )

        # Initialize CostTrackingManager from "costs_tracking_util" function

        cost_tracker_module_name = "function_costs_tracking_util"
        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is not loaded")
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            body["model"], __user__, __task__
        )

        model_id = body["model"][body["model"].find(".") + 1 :]

        payload = {**body, "model": model_id}

        # Calculate Input Tokens (Estimated)
        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(
            body["messages"]
        )

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} OpenAI Request Payload: {payload}")

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

            # Actual OpenAI API Request

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            response = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=body.get("stream", False),
            )

            response.raise_for_status()

            if body.get("stream", False):
                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} returning streaming response")

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    start_time = time.time()
                    last_update_time = 0
                    stream_completed = False

                    try:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    json_str = decoded_line[6:].strip()

                                    if json_str == "[DONE]":
                                        break

                                    try:
                                        data = json.loads(json_str)
                                        if (
                                            data.get("choices")
                                            and len(data["choices"]) > 0
                                        ):
                                            delta = data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                streamed_content_buffer += delta[
                                                    "content"
                                                ]

                                        yield f"{decoded_line}\n\n"
                                    except json.JSONDecodeError:
                                        if self.valves.DEBUG:
                                            print(
                                                "DEBUG OpenAI stream_response wrapper... invalid JSON on a data chunk:"
                                            )
                                            print(json_str)
                                        pass
                                else:
                                    if self.valves.DEBUG:
                                        print(
                                            "DEBUG OpenAI stream_response wrapper... decoded_line does not start with 'data:'. Added data prefix."
                                        )
                                        print(decoded_line)
                                    yield f"data: {decoded_line}\n\n"

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

                    except GeneratorExit:
                        if self.valves.DEBUG:
                            print(
                                "DEBUG OpenAI stream_response wrapper... aborted by client"
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

                        if self.valves.DEBUG:
                            print(
                                f"DEBUG Finalized stream (completed: {stream_completed})"
                            )

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # BATCH RESPONSE

                response_json = response.json()

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
