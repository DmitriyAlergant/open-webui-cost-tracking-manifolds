"""
title: Anthropic Manifold with Cost Tracking
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Union

from open_webui.utils.misc import (
    get_messages_content,
    pop_system_message,
)

from typing import Any, Awaitable, Callable

import os
import sys
import time
import json

from fastapi.responses import StreamingResponse

from anthropic import Anthropic, AsyncAnthropic

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
        ENABLE_WEB_SEARCH: bool = Field(
            default=False, description="Enable Web Search tool for Anthropic"
        )
        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"

        self.valves = self.Valves()

        self.debug_prefix = "DEBUG:    " + __name__ + " -"

        pass

    def init_anthropic_client(self, async_client=False):

        if not self.valves.ANTHROPIC_API_KEY:
            raise Exception(
                f"Anthropic Manifold function: ANTHROPIC_API_KEY valve not provided"
            )

        if async_client:
            return AsyncAnthropic(
                api_key=self.valves.ANTHROPIC_API_KEY,
                base_url=self.valves.ANTHROPIC_API_BASE_URL,
            )

        else:
            return Anthropic(
                api_key=self.valves.OPENAI_API_KEY,
                base_url=self.valves.OPENAI_API_BASE_URL,
            )

    def get_anthropic_models(self):
        return [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "max_tokens": 4096, "thinking": None},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "max_tokens": 4096, "thinking": None},
            {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet (2024-06-20)", "max_tokens": 4096, "thinking": None},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.6 Sonnet (2024-10-22)", "max_tokens": 8192, "thinking": None},
            {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet (2025-02-19) non-thinking", "max_tokens": 16384, "thinking": None},
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

        # Remove the "anthropic." prefix from the model name
        model_id = body["model"][body["model"].find(".") + 1 :]

        # Extract system message and pop from messages
        system_message, messages = pop_system_message(body["messages"])

        # Check for reasoning_effort parameter
        reasoning_effort = body.pop("reasoning_effort", None)
        thinking_params = {}
        max_tokens_override = None

        if reasoning_effort:
            if reasoning_effort == "low":
                thinking_params = {"type": "enabled", "budget_tokens": 8000}
                max_tokens_override = 16384 
            elif reasoning_effort == "medium":
                thinking_params = {"type": "enabled", "budget_tokens": 24000}
                max_tokens_override = 64000 
            elif reasoning_effort == "high":
                thinking_params = {"type": "enabled", "budget_tokens": 48000}
                max_tokens_override = 64000
            else:
                print(f"{self.debug_prefix} Unknown reasoning_effort: {reasoning_effort}. Ignoring.")

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

        # Ensure the system_message is pulled up a body-level parameter as per Anthropic Messages API specs.
        payload = {
            "model": model_id,
            "messages": processed_messages,
            **({"max_tokens": max_tokens_override} if max_tokens_override else ({"max_tokens": body["max_tokens"]} if "max_tokens" in body else {"max_tokens": 16384})),
            "stop_sequences": body.get("stop", []),
            **({"system": str(system_message)} if system_message else {}),
            "stream": body.get("stream", False),
        }

        # Inject web search tool if enabled
        if self.valves.ENABLE_WEB_SEARCH:
            payload["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            }]

        # Add thinking params and adjust others if reasoning_effort was provided
        if thinking_params:
            payload["thinking"] = thinking_params
        else:
            # No reasoning_effort, add params only if they exist in the body
            if "temperature" in body:
                payload["temperature"] = body["temperature"]
            if "top_k" in body:
                payload["top_k"] = body["top_k"]
            if "top_p" in body:
                payload["top_p"] = body["top_p"]

        # Calculate Input Tokens (Estimated)
        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(
            body["messages"]
        )

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} Anthropic Request Payload: {payload}")

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

            # Init Anthropic API Client (Async)

            if self.valves.DEBUG:
                os.environ["ANTHROPIC_LOG"] = "debug"

            anthropic_client = self.init_anthropic_client(async_client=True)

            if body.get("stream", False) == True:

                # STREAMING REQUEST

                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} sending a streaming request...")

                async def stream_generator():
                    streamed_content_buffer = ""
                    generated_tokens = 0
                    start_time = time.time()
                    last_update_time = 0
                    stream_completed = False
                    thinking_content_buffer = ""  # Buffer for thinking content
                    reasoning_tokens = 0  # Accumulator for reasoning tokens
                    is_thinking_block = False # Flag to track if we are inside a thinking block
                    # Variables to handle tool usage (e.g., web_search)
                    tool_use_active = False      # Inside a server_tool_use content block
                    tool_input_buffer = ""       # Collect partial_json for tool arguments
                    current_tool_index = None     # Index of the current tool block so we can reference it when yielding

                    try:

                        payload.pop("stream")  # avoid duplicate json attribute

                        async with anthropic_client.messages.stream(
                            **payload
                        ) as stream:
                            # Use the async iterator directly
                            async for event in stream:

                                #print(f"DEBUG: Anthropic event: {event}")

                                if event.type == "content_block_start":
                                    if event.content_block.type == "thinking":
                                        is_thinking_block = True
                                        # Yield opening <think> tag
                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': '<think>'}}]})}\n\n"
                                    elif event.content_block.type == "text":
                                        is_thinking_block = False
                                    elif event.content_block.type == "server_tool_use":
                                        # Begin a tool invocation block – collect its JSON arguments
                                        tool_use_active = True
                                        tool_input_buffer = ""
                                        current_tool_index = event.index
                                    elif hasattr(event.content_block, "type") and str(event.content_block.type).endswith("_tool_result"):
                                        # Handle tool results (for now, only web_search_tool_result is expected)
                                        if event.content_block.type == "web_search_tool_result":
                                            results_text = "\n\n🌐 Web search results:\n"
                                            if isinstance(event.content_block.content, list):
                                                for i, res in enumerate(event.content_block.content, 1):
                                                    if isinstance(res, dict) and res.get("type") == "web_search_result":
                                                        title = res.get("title", "result")
                                                        url = res.get("url", "")
                                                        results_text += f"{i}. {title} - {url}\n"

                                            # Yield the formatted results for immediate display in Open WebUI
                                            yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': results_text}}]})}\n\n"
                                            streamed_content_buffer += results_text
                                    # Ignore redacted_thinking for now

                                elif event.type == "content_block_delta":
                                    if event.delta.type == "thinking_delta":
                                        thinking_text = event.delta.thinking
                                        # Append to thinking buffer for token counting
                                        thinking_content_buffer += thinking_text
                                        # Yield thinking content delta
                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': thinking_text}}]})}\n\n"
                                    elif event.delta.type == "text_delta":
                                        text = event.delta.text
                                        # Append to regular buffer for token counting
                                        streamed_content_buffer += text
                                        # Yield regular content delta
                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': text}}]})}\n\n"
                                    elif tool_use_active and event.delta.type == "input_json_delta":
                                        # Accumulate partial JSON representing tool input
                                        tool_input_buffer += event.delta.partial_json
                                    elif event.delta.type == "citations_delta":
                                        citation = event.delta.citation
                                        # Safely access attributes using getattr
                                        title = getattr(citation, 'title', getattr(citation, 'url', ''))
                                        url = getattr(citation, 'url', '')
                                        citation_text = f" ([{title}]({url}))"
                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': citation_text}}]})}\n\n"
                                        streamed_content_buffer += citation_text

                                elif event.type == "content_block_stop":
                                    if is_thinking_block:
                                        # Yield closing </think> tag
                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': '</think>'}}]})}\n\n"
                                        is_thinking_block = False 
                                    elif tool_use_active and event.index == current_tool_index:
                                        # Conclude server_tool_use block – we now have the full tool input JSON
                                        query = ""
                                        try:
                                            # The accumulated JSON string might be incomplete; attempt best-effort parsing
                                            query_dict = json.loads(tool_input_buffer) if tool_input_buffer.strip() else {}
                                            query = query_dict.get("query", "")
                                        except Exception:
                                            pass

                                        if query:
                                            search_msg = f"\n\n🔎 Searching the web for: {query}\n\n"
                                        else:
                                            search_msg = "\n\n🔎 Performing web search...\n\n"

                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': search_msg}}]})}\n\n"
                                        streamed_content_buffer += search_msg

                                        # Reset tool-use tracking flags
                                        tool_use_active = False
                                        tool_input_buffer = ""

                                # Periodic update for cost tracking (every second)
                                current_time = time.time()
                                if current_time - last_update_time >= 1:
                                    # Count tokens generated in the last second
                                    current_generated = cost_tracking_manager.count_tokens(streamed_content_buffer)
                                    current_reasoning = cost_tracking_manager.count_tokens(thinking_content_buffer)

                                    generated_tokens += current_generated
                                    reasoning_tokens += current_reasoning

                                    cost_tracking_manager.calculate_costs_update_status_and_persist(
                                        input_tokens=input_tokens,
                                        generated_tokens=generated_tokens,
                                        reasoning_tokens=reasoning_tokens, # Pass reasoning tokens
                                        start_time=start_time,
                                        __event_emitter__=__event_emitter__,
                                        status="Streaming...",
                                        persist_usage=False,
                                        context_messages_count=len(messages)
                                    )

                                    # Clear buffers after counting
                                    streamed_content_buffer = ""
                                    thinking_content_buffer = ""
                                    last_update_time = current_time

                            # After loop completes normally
                            stream_completed = True
                            yield "data: [DONE]\n\n"

                    except GeneratorExit:
                        if self.valves.DEBUG:
                            print(
                                "DEBUG Anthropic stream_response wrapper... aborted by client"
                            )
                        stream_completed = False # Mark as stopped if aborted
                    except Exception as e:
                        _, _, tb = sys.exc_info()
                        line_number = tb.tb_lineno
                        print(f"Error processing stream on line {line_number}: {e}")
                        stream_completed = False # Mark as stopped on error
                        # Format error message to match the expected choices array format
                        error_message = str(e)
                        if hasattr(e, 'response') and hasattr(e.response, 'json'):
                            try:
                                error_json = e.response.json()
                                if 'error' in error_json and 'message' in error_json['error']:
                                    error_message = error_json['error']['message']
                            except:
                                pass
                        yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': f'Error: {error_message}'}}]})}\n\n"
                        yield "data: [DONE]\n\n" # Still need to send DONE for OpenWebUI
                        # Re-raise the exception to be caught by the outer handler if needed
                        # raise e # Commented out to allow final cost update

                    finally:
                        # Final token count for any remaining buffer content
                        final_generated = cost_tracking_manager.count_tokens(streamed_content_buffer)
                        final_reasoning = cost_tracking_manager.count_tokens(thinking_content_buffer)

                        generated_tokens += final_generated
                        reasoning_tokens += final_reasoning

                        # Calculate total output tokens for cost calculation
                        total_output_tokens = generated_tokens + reasoning_tokens

                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=input_tokens,
                            generated_tokens=total_output_tokens, # Use total for cost calculation
                            reasoning_tokens=reasoning_tokens, # Pass separate reasoning tokens for status message
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed else "Stopped",
                            persist_usage=True, # Persist final usage
                            context_messages_count=len(messages)
                        )

                        if self.valves.DEBUG:
                            print(
                                f"DEBUG Finalized stream (completed: {stream_completed}, generated: {generated_tokens}, reasoning: {reasoning_tokens}, total output for cost: {total_output_tokens})"
                            )

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:

                # BATCH REQUEST

                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} sending non-stream request...")

                message = await anthropic_client.messages.create(**payload)

                res = message.to_dict()

                if self.valves.DEBUG:
                    print(f"{self.debug_prefix} Anthropic batch response: {res}")

                generated_text = ""

                for content_item in res.get("content", []):
                    if content_item["type"] == "text":
                        generated_text += content_item["text"]

                # Try to get generated tokens from usage response, if not available default to input_tokens we already had
                input_tokens = res.get("usage", {}).get("input_tokens", input_tokens)

                # Try to get generated tokens from usage response, if not available default to our own calculations
                generated_tokens = res.get("usage", {}).get(
                    "output_tokens", cost_tracking_manager.count_tokens(generated_text)
                )

                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=None, # Batch mode doesn't explicitly track reasoning tokens here yet
                    start_time=start_time,
                    __event_emitter__=__event_emitter__,
                    status="Completed",
                    persist_usage=True,
                    context_messages_count=len(messages)
                )

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