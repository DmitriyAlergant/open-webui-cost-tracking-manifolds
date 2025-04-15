"""
title: Anthropic Manifold with Cost Tracking
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.2.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Union, Dict, Any, List, Optional

from open_webui.utils.misc import (
    get_messages_content,
    pop_system_message,
)

from typing import Any, Awaitable, Callable

import os
import sys
import time
import json
import aiohttp
import asyncio
from fastapi.responses import StreamingResponse

MODULE_USAGE_TRACKING = "function_module_usage_tracking"


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_BASE_URL: str = Field(
            default="https://api.anthropic.com",
            description="The base URL for Anthropic API endpoints.",
        )
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Required API key to access Anthropic API.",
        )
        DEBUG: bool = Field(default=False, description="Display debugging messages")

        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"

        self.valves = self.Valves()

        self.debug_prefix = "DEBUG:    " + __name__ + " -"

        pass

    def get_anthropic_models(self):
        return [
            # Standard models
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "max_tokens": 4096, "thinking": None},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "max_tokens": 4096, "thinking": None},
            {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet (2024-06-20)", "max_tokens": 4096, "thinking": None},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.6 Sonnet (2024-10-22)", "max_tokens": 8192, "thinking": None},
            {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet (2025-02-19) non-thinking", "max_tokens": 16384, "thinking": None},
            
            {"id": "claude-3-7-sonnet-thinking-small", "name": "Claude 3.7 Sonnet + Thinking (4K)", 
             "api_model_id": "claude-3-7-sonnet-20250219", "max_tokens": 16384, 
             "thinking": {"type": "enabled", "budget_tokens": 4000},
             "description": "Claude 3.7 with extended thinking (forces temperature=1.0, removes top_k/top_p)"},
            
            {"id": "claude-3-7-sonnet-thinking-medium", "name": "Claude 3.7 Sonnet + Thinking (16K)", 
             "api_model_id": "claude-3-7-sonnet-20250219", "max_tokens": 64000, 
             "thinking": {"type": "enabled", "budget_tokens": 16000},
             "description": "Claude 3.7 with extended thinking (forces temperature=1.0, removes top_k/top_p)"},
            
            {"id": "claude-3-7-sonnet-thinking-high", "name": "Claude 3.7 Sonnet + Thinking (64K)", 
             "api_model_id": "claude-3-7-sonnet-20250219", "max_tokens": 128000, 
             "thinking": {"type": "enabled", "budget_tokens": 64000},
             "description": "Claude 3.7 with extended thinking (forces temperature=1.0, removes top_k/top_p)"},
        ]

    def pipes(self):
        if self.valves.ANTHROPIC_API_KEY:
            models = self.get_anthropic_models()
            return models
        else:
            print(f"Anthropic Manifold function: ANTHROPIC_API_KEY valve not provided")
            return [
                {
                    "id": "error",
                    "name": f"Anthropic Manifold function: ANTHROPIC_API_KEY valve not provided",
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

        # Initialize CostTrackingManager from "usage_tracking_util" function
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

        # Get the selected model configuration
        selected_model = None
        
        # Get username for logging
        username = __user__.get("email", "unknown")
        
        if self.valves.DEBUG:
            print(f"{self.debug_prefix} User '{username}' requested model: '{body['model']}'")
            
        # Check if model ID has a prefix (like 'anthropic.')
        normalized_model_id = body["model"]
        if "." in normalized_model_id:
            normalized_model_id = normalized_model_id[normalized_model_id.find(".") + 1:]

        models = self.get_anthropic_models()
        
        selected_model = next((model for model in models if model["id"] == normalized_model_id), None)

        if not selected_model:
            raise Exception(f"Anthropic Manifold function: Model '{body['model']}' not found")

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} User '{username}' using model: {selected_model['name']} (max_tokens={selected_model['max_tokens']})")

        api_model_id = selected_model.get("api_model_id", selected_model["id"])
        max_tokens = selected_model.get("max_tokens", body.get("max_tokens", 4096))
        thinking = selected_model.get("thinking")
        if self.valves.DEBUG:
            print(f"{self.debug_prefix} User '{username}' using model: {selected_model['name']}, api_model_id={api_model_id}, thinking={thinking}, max_tokens={max_tokens}")

        # Thinking mode does not currently support temperature, top_k, top_p
        if thinking:
            body["temperature"] = 1.0
            body.pop("top_k", None)
            body.pop("top_p", None)

        # Extract system message and pop from messages
        system_message, messages = pop_system_message(body["messages"])

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} User '{username}' using system message: {system_message}")

        # Pre-process chat messages as per Anthropic requirements
        processed_messages = []
        image_count = 0
        total_image_size = 0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        if image_count >= 5:
                            raise ValueError(
                                "Maximum of 5 images per API call exceeded"
                            )

                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                        else:
                            image_size = 0

                        total_image_size += image_size
                        if total_image_size > 100 * 1024 * 1024:
                            raise ValueError(
                                "Total size of images exceeds 100 MB limit"
                            )

                        image_count += 1
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        payload = {
            "model": api_model_id,
            "messages": processed_messages,
            "max_tokens": max_tokens,
            "temperature": body.get("temperature", 0.8),
            **({"top_k": body.get("top_k")} if body.get("top_k") is not None else {}),
            **({"top_p": body.get("top_p")} if body.get("top_p") is not None else {}),
            "stop_sequences": body.get("stop", []),
            **({"system": str(system_message)} if system_message else {}),
            "stream": body.get("stream", False),
        }
        
        if thinking and isinstance(thinking, dict) and thinking.get("type") == "enabled":

            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking.get("budget_tokens", 16000)
            }
            
            payload["temperature"] = 1.0
            payload.pop("top_k", None)
            payload.pop("top_p", None)

        # Calculate Input Tokens (Estimated)
        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(
            body["messages"]
        )

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
                context_messages_count=len(messages)
            )

            # Set up headers for Anthropic API request
            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            # For debugging
            if self.valves.DEBUG:
                os.environ["ANTHROPIC_LOG"] = "debug"

            if body.get("stream", False) == True:
                # STREAMING REQUEST
                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} User '{username}' sending streaming request to {api_model_id}" + 
                          (f" with thinking budget: {thinking.get('budget_tokens')} tokens" if thinking else ""))

                # Set stream parameter for API
                payload["stream"] = True

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    stream_completed = False
                    last_update_time = time.time()
                    current_text_content = ""
                    reasoning_tokens = None
                    in_thinking_block = False

                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"{self.valves.ANTHROPIC_API_BASE_URL}/v1/messages",
                                headers=headers,
                                json=payload,
                                timeout=60,
                            ) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    if self.valves.DEBUG:
                                        print(f"{self.debug_prefix} API Error for user '{username}': {error_text}")
                                    
                                    try:
                                        error_json = json.loads(error_text)
                                        error_message = error_json.get("error", {}).get("message", "Unknown error")
                                        error_type = error_json.get("error", {}).get("type", "unknown_error")
                                        error_msg = {
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": f"Error ({error_type}): {error_message}"},
                                                }
                                            ]
                                        }
                                    except:
                                        error_msg = {
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": f"Error: {error_text}"},
                                                }
                                            ]
                                        }
                                    
                                    yield f"data: {json.dumps(error_msg)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                                # Process the SSE stream
                                async for line in response.content:
                                    line = line.decode('utf-8')
                                    if not line.strip():
                                        continue
                                    
                                    if line.startswith('data: '):
                                        data = line[6:].strip()
                                        if not data or data == '[DONE]':
                                            continue
                                        
                                        try:
                                            event_data = json.loads(data)
                                            event_type = event_data.get('type')
                                            
                                            # Handle thinking delta event
                                            if event_type == 'content_block_delta' and event_data.get('delta', {}).get('type') == 'thinking_delta':
                                                thinking_delta = event_data['delta']['thinking']
                                                
                                                if self.valves.DEBUG:
                                                    print(f"{self.debug_prefix} User '{username}' received thinking delta: {thinking_delta[:30]}...")
                                                
                                                # Format for OpenAI-compatible streaming with <think> tags
                                                # Only send the opening tag for the first thinking delta
                                                if not in_thinking_block:
                                                    thinking_prefix = "<thinking>\n"
                                                    content_json = {
                                                        "choices": [
                                                            {"index": 0, "delta": {"content": thinking_prefix}}
                                                        ]
                                                    }
                                                    yield f"data: {json.dumps(content_json)}\n\n"
                                                    streamed_content_buffer += thinking_prefix
                                                    in_thinking_block = True
                                                
                                                # Send the thinking content
                                                content_json = {
                                                    "choices": [
                                                        {"index": 0, "delta": {"content": thinking_delta}}
                                                    ]
                                                }
                                                yield f"data: {json.dumps(content_json)}\n\n"
                                                streamed_content_buffer += thinking_delta
                                            
                                            # Handle signature delta event (end of thinking block)
                                            elif event_type == 'content_block_delta' and event_data.get('delta', {}).get('type') == 'signature_delta':
                                                # Close the thinking block with </think> tag
                                                if self.valves.DEBUG:
                                                    print(f"{self.debug_prefix} User '{username}' received signature delta, closing thinking block")
                                                
                                                if in_thinking_block:
                                                    thinking_suffix = "\n</thinking>\n\n"
                                                    content_json = {
                                                        "choices": [
                                                            {"index": 0, "delta": {"content": thinking_suffix}}
                                                        ]
                                                    }
                                                    yield f"data: {json.dumps(content_json)}\n\n"
                                                    streamed_content_buffer += thinking_suffix
                                                    in_thinking_block = False
                                            
                                            # Handle redacted thinking block
                                            elif event_type == 'content_block_start' and event_data.get('content_block', {}).get('type') == 'redacted_thinking':
                                                if self.valves.DEBUG:
                                                    print(f"{self.debug_prefix} User '{username}' received redacted thinking block")
                                                
                                                # Send a placeholder for redacted thinking with <think> tags
                                                redacted_message = "<think>\n[Redacted thinking content - encrypted for safety reasons]\n</think>\n\n"
                                                content_json = {
                                                    "choices": [
                                                        {"index": 0, "delta": {"content": redacted_message}}
                                                    ]
                                                }
                                                yield f"data: {json.dumps(content_json)}\n\n"
                                                streamed_content_buffer += redacted_message
                                            
                                            # Handle content block start for text blocks
                                            elif event_type == 'content_block_start' and event_data.get('content_block', {}).get('type') == 'text':
                                                # If we were in a thinking block and didn't get a signature delta, close it
                                                if in_thinking_block:
                                                    if self.valves.DEBUG:
                                                        print(f"{self.debug_prefix} User '{username}' transitioning from thinking to text block")
                                                    
                                                    thinking_suffix = "\n</think>\n\n"
                                                    content_json = {
                                                        "choices": [
                                                            {"index": 0, "delta": {"content": thinking_suffix}}
                                                        ]
                                                    }
                                                    yield f"data: {json.dumps(content_json)}\n\n"
                                                    streamed_content_buffer += thinking_suffix
                                                    in_thinking_block = False
                                            
                                            # Handle text delta event
                                            elif event_type == 'content_block_delta' and event_data.get('delta', {}).get('type') == 'text_delta':
                                                text_delta = event_data['delta']['text']
                                                current_text_content += text_delta
                                                streamed_content_buffer += text_delta
                                                
                                                # Format for OpenAI-compatible streaming
                                                content_json = {
                                                    "choices": [
                                                        {"index": 0, "delta": {"content": text_delta}}
                                                    ]
                                                }
                                                yield f"data: {json.dumps(content_json)}\n\n"
                                                
                                            # Handle final message delta event with usage information
                                            elif event_type == 'message_delta' and 'usage' in event_data:
                                                # Update token counts if available
                                                if 'output_tokens' in event_data.get('usage', {}):
                                                    generated_tokens = event_data['usage']['output_tokens']
                                                
                                                # Get reasoning tokens if available
                                                if thinking and 'reasoning_tokens' in event_data.get('usage', {}):
                                                    reasoning_tokens = event_data['usage']['reasoning_tokens']
                                                    
                                                    if self.valves.DEBUG:
                                                        print(f"{self.debug_prefix} User '{username}' received {reasoning_tokens} reasoning tokens")
                                                    
                                                    # Update status with reasoning tokens
                                                    cost_tracking_manager.calculate_costs_update_status_and_persist(
                                                        input_tokens=input_tokens,
                                                        generated_tokens=generated_tokens,
                                                        reasoning_tokens=reasoning_tokens,
                                                        start_time=start_time,
                                                        __event_emitter__=__event_emitter__,
                                                        status="Streaming...",
                                                        persist_usage=False,
                                                        context_messages_count=len(messages)
                                                    )
                                            
                                            # Handle message stop event
                                            elif event_type == 'message_stop':
                                                stream_completed = True
                                                yield "data: [DONE]\n\n"

                                            # Periodically update tokens and cost - on all events
                                            current_time = time.time()
                                            if current_time - last_update_time >= 1:
                                                generated_tokens += cost_tracking_manager.count_tokens(
                                                    streamed_content_buffer
                                                )
                                                
                                                cost_tracking_manager.calculate_costs_update_status_and_persist(
                                                    input_tokens=input_tokens,
                                                    generated_tokens=generated_tokens,
                                                    reasoning_tokens=reasoning_tokens,
                                                    start_time=start_time,
                                                    __event_emitter__=__event_emitter__,
                                                    status="Streaming...",
                                                    persist_usage=False,
                                                    context_messages_count=len(messages)
                                                )
                                                
                                                streamed_content_buffer = ""
                                                last_update_time = current_time
                                                
                                        except json.JSONDecodeError:
                                            if self.valves.DEBUG:
                                                print(f"{self.debug_prefix} Error parsing SSE data: {data}")
                            
                    except GeneratorExit:
                        if self.valves.DEBUG:
                            print(f"{self.debug_prefix} Stream aborted by user '{username}'")
                    except Exception as e:
                        _, _, tb = sys.exc_info()
                        line_number = tb.tb_lineno
                        print(f"Error on line {line_number}: {e}")
                        raise e
                    finally:
                        # Calculate final token count from any remaining buffer
                        generated_tokens += cost_tracking_manager.count_tokens(
                            streamed_content_buffer
                        )

                        # Update final stats and persist usage data
                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=input_tokens,
                            generated_tokens=generated_tokens,
                            reasoning_tokens=reasoning_tokens,
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed else "Stopped",
                            persist_usage=True,
                            context_messages_count=len(messages)
                        )

                        if self.valves.DEBUG:
                            msg = f"Completed" if stream_completed else "Stopped"
                            print(f"{self.debug_prefix} User '{username}' streaming request {msg.lower()}")

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # BATCH REQUEST
                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} User '{username}' sending batch request to {api_model_id}" + 
                          (f" with thinking budget: {thinking.get('budget_tokens')} tokens" if thinking else ""))

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.valves.ANTHROPIC_API_BASE_URL}/v1/messages",
                        headers=headers,
                        json=payload,
                        timeout=60,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            if self.valves.DEBUG:
                                print(f"{self.debug_prefix} API Error for user '{username}': {error_text}")
                            
                            try:
                                error_json = json.loads(error_text)
                                error_message = error_json.get("error", {}).get("message", "Unknown error")
                                error_type = error_json.get("error", {}).get("type", "unknown_error")
                                raise Exception(f"Anthropic API Error ({error_type}): {error_message}")
                            except json.JSONDecodeError:
                                raise Exception(f"Anthropic API Error: {error_text}")
                        
                        res = await response.json()

                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} Anthropic batch response: {res}")

                generated_text = ""

                # Extract text content from the response
                for content_item in res.get("content", []):
                    if content_item["type"] == "text":
                        generated_text += content_item["text"]
                    elif content_item["type"] == "thinking":
                        # Format thinking content with <think> tags
                        thinking_text = content_item.get("thinking", "")
                        if thinking_text:
                            if self.valves.DEBUG:
                                print(f"{self.debug_prefix} User '{username}' received thinking block: {thinking_text[:50]}...")
                            generated_text += f"<think>\n{thinking_text}\n</think>\n\n"
                    elif content_item["type"] == "redacted_thinking":
                        # Add a placeholder for redacted thinking
                        if self.valves.DEBUG:
                            print(f"{self.debug_prefix} User '{username}' received redacted thinking block")
                        generated_text += "<think>\n[Redacted thinking content - encrypted for safety reasons]\n</think>\n\n"

                # Get token counts from response usage data
                input_tokens = res.get("usage", {}).get("input_tokens", input_tokens)
                generated_tokens = res.get("usage", {}).get(
                    "output_tokens", cost_tracking_manager.count_tokens(generated_text)
                )
                
                # Get reasoning tokens if available (for thinking feature)
                reasoning_tokens = None
                if thinking and "usage" in res:
                    reasoning_tokens = res.get("usage", {}).get("reasoning_tokens", None)
                    if reasoning_tokens and self.valves.DEBUG:
                        print(f"{self.debug_prefix} User '{username}' received {reasoning_tokens} reasoning tokens")

                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=reasoning_tokens,
                    start_time=start_time,
                    __event_emitter__=__event_emitter__,
                    status="Completed",
                    persist_usage=True,
                    context_messages_count=len(messages)
                )
                
                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} User '{username}' batch request completed")

                return generated_text

        except Exception as e:
            _, _, tb = sys.exc_info()
            line_number = tb.tb_lineno
            print(f"Error on line {line_number}: {e}")
            raise e

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["image_url"]["url"]},
            }
