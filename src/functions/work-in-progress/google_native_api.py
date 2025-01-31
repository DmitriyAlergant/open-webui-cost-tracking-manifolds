"""
title: Google AI Manifold with Cost Tracking
author: Assistant
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Union, List, Any, Awaitable, Callable
import sys
import time
import json
import aiohttp
from fastapi.responses import StreamingResponse
from open_webui.utils.misc import get_messages_content, pop_system_message

MODULE_USAGE_TRACKING = "function_module_usage_tracking"

class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="", description="Google AI API Key")
        GOOGLE_API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com/v1",
            description="Google AI API Base URL"
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "google"
        self.name = "google/"
        
        self.valves = self.Valves()
        self.debug_prefix = "DEBUG:    " + __name__ + " -"

    def pipes(self) -> List[dict]:
        if not self.valves.GOOGLE_API_KEY:
            return [{
                "id": "error",
                "name": "Google AI Manifold: GOOGLE_API_KEY valve not provided"
            }]

        return [
            {
                "id": "gemini-1.5-flash",
                "name": "gemini-1.5-flash",
            },
            {
                "id": "gemini-2.0-flash-exp",
                "name": "gemini-2.0-flash-exp",
            },
            {
                "id": "gemini-2.0-flash-thinking-exp",
                "name": "gemini-2.0-flash-thinking-exp",
            },
            {
                "id": "gemini-1.5-pro",
                "name": "gemini-1.5-pro",
            },
            {
                "id": "gemini-exp-1206",
                "name": "gemini-exp-1206",
            },
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:
        # Initialize cost tracking
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

        # Process messages
        system_message, messages = pop_system_message(body["messages"])
        
        # Strip the "google." prefix from model name if present
        model = body["model"]
        if "." in model:
            model = model.split(".", 1)[1]  # Get everything after the first dot
        
        # Convert messages to Google AI format
        contents = []
        if system_message:
            contents.append({
                "role": "system",
                "parts": [{"text": str(system_message)}]
            })
            
        for message in messages:
            content = {"role": message["role"]}
            
            if isinstance(message.get("content"), list):
                parts = []
                for item in message["content"]:
                    if item["type"] == "text":
                        parts.append({"text": item["text"]})
                    elif item["type"] == "image_url":
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": item["image_url"]["url"].split(",")[1]
                            }
                        })
                content["parts"] = parts
            else:
                content["parts"] = [{"text": message.get("content", "")}]
                
            contents.append(content)

        # Calculate input tokens
        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(messages)

        # Prepare request payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": body.get("temperature", 0.7),
                "maxOutputTokens": body.get("max_tokens", 2048),
                "topP": body.get("top_p", 0.8),
                "topK": body.get("top_k", 40)
            }
        }

        base_url = self.valves.GOOGLE_API_BASE_URL.rstrip('/')
        
        # Add API key as query parameter instead of Authorization header
        url_with_key = lambda endpoint: f"{base_url}/models/{model}:{endpoint}?key={self.valves.GOOGLE_API_KEY}"
        
        headers = {
            "Content-Type": "application/json"
        }  # Remove Authorization header

        start_time = time.time()
        
        cost_tracking_manager.calculate_costs_update_status_and_persist(
            input_tokens=input_tokens,
            generated_tokens=None,
            reasoning_tokens=None,
            start_time=start_time,
            __event_emitter__=__event_emitter__,
            status="Requested...",
            persist_usage=False,
            context_messages_count=len(messages)
        )

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} Payload: {payload}")

        try:
            if body.get("stream", False):
                # STREAMING REQUEST
                url = url_with_key("streamGenerateContent")  # Use the new URL construction

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    last_update_time = 0
                    stream_completed = False
                    json_buffer = ""

                    try:
                        if self.valves.DEBUG:
                            print(f"{self.debug_prefix} Starting stream request to {url}")
                            
                        async with aiohttp.ClientSession() as session:
                            async with session.post(url, json=payload, headers=headers) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    raise Exception(f"HTTP Error {response.status}: {error_text}")

                                if self.valves.DEBUG:
                                    print(f"{self.debug_prefix} Connected to stream, status: {response.status}")

                                async for line in response.content:
                                    if line:
                                        line_str = line.decode('utf-8').strip()
                                        if not line_str:
                                            continue
                                            
                                        if self.valves.DEBUG:
                                            print(f"{self.debug_prefix} Received chunk: {line_str}")
                                        
                                        # Accumulate JSON chunks
                                        json_buffer += line_str
                                        
                                        try:
                                            # Try to parse complete JSON objects
                                            response_obj = json.loads(json_buffer)
                                            json_buffer = ""  # Reset buffer after successful parse
                                            
                                            # Extract text from the parsed response
                                            if "candidates" in response_obj and response_obj["candidates"]:
                                                candidate = response_obj["candidates"][0]
                                                if "content" in candidate and "parts" in candidate["content"]:
                                                    for part in candidate["content"]["parts"]:
                                                        if "text" in part:
                                                            content = part["text"]
                                                            streamed_content_buffer += content

                                                            # Emulate OpenAI format
                                                            content_json = {
                                                                "choices": [{
                                                                    "delta": {"content": content},
                                                                    "index": 0
                                                                }]
                                                            }
                                                            
                                                            yield f"data: {json.dumps(content_json)}\n\n"

                                                            # Update status every ~1 second
                                                            current_time = time.time()
                                                            if current_time - last_update_time >= 1:
                                                                generated_tokens = cost_tracking_manager.count_tokens(
                                                                    streamed_content_buffer
                                                                )

                                                                if self.valves.DEBUG:
                                                                    print(f"{self.debug_prefix} Updating tokens count: {generated_tokens}")

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
                                            # Incomplete JSON, continue accumulating
                                            continue
                                        except Exception as e:
                                            if self.valves.DEBUG:
                                                print(f"{self.debug_prefix} Error processing chunk: {e}")
                                            continue

                        stream_completed = True
                        if self.valves.DEBUG:
                            print(f"{self.debug_prefix} Stream completed successfully")
                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        _, _, tb = sys.exc_info()
                        print(f"{self.debug_prefix} Error on line {tb.tb_lineno}: {e}")
                        if self.valves.DEBUG:
                            print(f"{self.debug_prefix} Current chunk at error: {current_chunk}")
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
                                f"DEBUG Finalized stream (completed: {stream_completed})"
                            )

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # BATCH REQUEST
                url = url_with_key("generateContent")  # Use the new URL construction

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"HTTP Error {response.status}: {error_text}")

                        response_json = await response.json()

                        if self.valves.DEBUG:
                            print(f"{self.debug_prefix} Response: {response_json}")

                        if "candidates" not in response_json or not response_json["candidates"]:
                            raise Exception("No response candidates received")

                        generated_text = ""
                        for part in response_json["candidates"][0]["content"]["parts"]:
                            if "text" in part:
                                generated_text += part["text"]

                        generated_tokens = cost_tracking_manager.count_tokens(generated_text)

                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=input_tokens,
                            generated_tokens=generated_tokens,
                            reasoning_tokens=None,
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed",
                            persist_usage=True,
                            context_messages_count=len(messages)
                        )

                        # Return in OpenAI-compatible format
                        return {
                            "choices": [{
                                "message": {"content": generated_text},
                                "index": 0
                            }]
                        }

        except Exception as e:
            _, _, tb = sys.exc_info()
            print(f"Error on line {tb.tb_lineno}: {e}")
            raise e