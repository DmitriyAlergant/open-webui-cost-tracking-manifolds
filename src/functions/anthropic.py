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
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "max_tokens": 4096,
                "thinking": None,
                "web_search": False,
                "web_search_max_uses": None
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "max_tokens": 4096,
                "thinking": None,
                "web_search": False,
                "web_search_max_uses": None
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.6 Sonnet",
                "max_tokens": 8192,
                "thinking": None,
                "web_search": False,
                "web_search_max_uses": None
            },
            {
                "id": "claude-3-7-sonnet-20250219",
                "name": "Claude 3.7 Sonnet",
                "max_tokens": 16384,
                "thinking": None,
                "web_search": False,
                "web_search_max_uses": None
            },
            {
                "id": "claude-opus-4-20250514",
                "name": "Claude 4 Opus",
                "max_tokens": 16384,
                "thinking": None,
                "web_search": False,
                "web_search_max_uses": None
            },
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude 4 Sonnet",
                "max_tokens": 16384,
                "thinking": None,
                "web_search": False,
                "web_search_max_uses": None
            },
            {
                "id": "claude-sonnet-4-20250514-websearch",
                "name": "Claude 4 Sonnet with Web Search",
                "max_tokens": 16384,
                "thinking": None,
                "web_search": True,
                "web_search_max_uses": 5
            },
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

        # Remove the "anthropic." prefix from the model name
        clean_model_id = body["model"][body["model"].find(".") + 1 :]

        # Find model config from our models list
        models = self.get_anthropic_models()
        model_config = next((m for m in models if m["id"] == clean_model_id), None)

        if self.valves.DEBUG:
            print(f"{self.debug_prefix} Model config: {model_config}")

        # If the model name ends with -websearch, strip that suffix
        if clean_model_id.endswith("-websearch"):
            clean_model_id = clean_model_id.replace("-websearch", "")
            if self.valves.DEBUG:
                print(f"{self.debug_prefix} Stripped -websearch suffix from model name: {body['model']}")

        # Initialize CostTrackingManager from "usage_tracking_util" function

        cost_tracker_module_name = MODULE_USAGE_TRACKING
        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is not loaded")
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            model=clean_model_id, 
            __user__=__user__, 
            __metadata__=__metadata__, 
            task=__task__, 
            debug=self.valves.DEBUG
        )

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
            "model": clean_model_id,
            "messages": processed_messages,
            **({"max_tokens": max_tokens_override} if max_tokens_override else ({"max_tokens": body["max_tokens"]} if "max_tokens" in body else {"max_tokens": 16384})),
            "stop_sequences": body.get("stop", []),
            **({"system": str(system_message)} if system_message else {}),
            "stream": body.get("stream", False),
        }

        # Inject web search tool if enabled for this model        
        if model_config and model_config.get("web_search", False):
            payload["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": model_config.get("web_search_max_uses", 5)
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
                    
                    # Buffer for text content blocks to group text and their citations
                    text_content_blocks_data = {}
                    final_usage_data = None # To store usage from message_delta
                    web_search_requests_count = 0 # Initialize web search counter

                    sse_event_suffix = "\n\n" # Correct SSE formatting

                    try:

                        payload.pop("stream")  # avoid duplicate json attribute

                        async with anthropic_client.messages.stream(
                            **payload
                        ) as stream:
                            # Use the async iterator directly
                            async for event in stream:

                                #print(f"DEBUG: Anthropic event: {event}")

                                if event.type == "content_block_start":
                                    if event.content_block.type == "text":
                                        text_content_blocks_data[event.index] = {"text_buffer": "", "citations_list": []}
                                        # Don't yield yet, buffer this text block
                                    elif event.content_block.type == "thinking":
                                        is_thinking_block = True
                                        # Yield opening <think> tag
                                        yield f"data: {json.dumps({'choices': [{'index': event.index, 'delta': {'content': '<think>'}}]})}{sse_event_suffix}"

                                    elif event.content_block.type == "server_tool_use":
                                        # Begin a tool invocation block – collect its JSON arguments
                                        tool_use_active = True
                                        tool_input_buffer = ""
                                        current_tool_index = event.index
                                        # Tool use message yielded at content_block_stop
                                    elif hasattr(event.content_block, "type") and str(event.content_block.type).endswith("_tool_result"):
                                        # Handle tool results (for now, only web_search_tool_result is expected)
                                        if event.content_block.type == "web_search_tool_result":
                                            # The "web_search_tool_result" block itself doesn't need to yield text to the UI.
                                            # The LLM's subsequent text blocks will reference these results via citations.
                                            pass # Explicitly do nothing for this block type in terms of direct UI output
                                    # Ignore redacted_thinking for now

                                elif event.type == "content_block_delta":
                                    block_idx = event.index 
                                    delta_type = event.delta.type

                                    if block_idx in text_content_blocks_data: # Delta for a text block we are buffering
                                        if delta_type == "text_delta":
                                            text_content_blocks_data[block_idx]["text_buffer"] += event.delta.text
                                        elif delta_type == "citations_delta":
                                            text_content_blocks_data[block_idx]["citations_list"].append(event.delta.citation)
                                        # No yield from here for buffered text blocks, processed at content_block_stop
                                    
                                    # Deltas for non-buffered blocks (thinking, tool input)
                                    elif delta_type == "thinking_delta":
                                        thinking_text = event.delta.thinking
                                        # Append to thinking buffer for token counting
                                        thinking_content_buffer += thinking_text
                                        # Yield thinking content delta
                                        yield f"data: {json.dumps({'choices': [{'index': block_idx, 'delta': {'content': thinking_text}}]})}{sse_event_suffix}"

                                    # tool_use_active implicitly checks if current_tool_index is not None
                                    elif tool_use_active and delta_type == "input_json_delta" and block_idx == current_tool_index:
                                        # Accumulate partial JSON representing tool input
                                        tool_input_buffer += event.delta.partial_json
                                        # Tool input message yielded at content_block_stop

                                elif event.type == "content_block_stop":
                                    block_idx = event.index

                                    if block_idx in text_content_blocks_data: # Stop for a text block we buffered
                                        data = text_content_blocks_data.pop(block_idx) # Use pop to remove it
                                        buffer = data["text_buffer"]
                                        citations_list = data["citations_list"]
                                        
                                        content_to_yield = buffer
                                        assembled_citations_text = ""
                                        unique_block_citations = set() # To track (title, url) tuples for this block

                                        for cit_obj in citations_list:
                                            title = getattr(cit_obj, 'title', '')
                                            url = getattr(cit_obj, 'url', '')
                                            if url: # Only include citation if there's a URL
                                                citation_key = (title, url)
                                                if citation_key not in unique_block_citations:
                                                    display_name = title if title else url 
                                                    assembled_citations_text += f" ([{display_name}]({url}))"
                                                    unique_block_citations.add(citation_key)
                                        
                                        content_to_yield += assembled_citations_text
                                        
                                        if content_to_yield: # Only yield if there's actual content
                                            yield f"data: {json.dumps({'choices': [{'index': block_idx, 'delta': {'content': content_to_yield}}]})}{sse_event_suffix}"

                                            streamed_content_buffer += content_to_yield # Add to global buffer for token counting
                                    
                                    # Stop for non-text blocks (thinking, tool use)
                                    elif is_thinking_block and not tool_use_active: # Ensure it's not a tool block also ending
                                        # This condition for is_thinking_block needs to be robust if thinking can have specific index
                                        # For now, assumes it's a general flag that's true for the thinking block that just ended.
                                        yield f"data: {json.dumps({'choices': [{'index': block_idx, 'delta': {'content': '</think>'}}]})}{sse_event_suffix}"

                                        is_thinking_block = False 
                                    
                                    elif tool_use_active and block_idx == current_tool_index:
                                        # Conclude server_tool_use block – we now have the full tool input JSON
                                        query = ""
                                        try:
                                            query_dict = json.loads(tool_input_buffer) if tool_input_buffer.strip() else {}
                                            query = query_dict.get("query", "")
                                        except Exception:
                                            pass # Ignore malformed JSON, proceed with generic message

                                        if query:
                                            search_msg = f"\n\n🔎 Searching the web for: {query}\n\n"
                                        else:
                                            search_msg = "\n\n🔎 Performing web search...\n\n"

                                        yield f"data: {json.dumps({'choices': [{'index': block_idx, 'delta': {'content': search_msg}}]})}{sse_event_suffix}"

                                        streamed_content_buffer += search_msg

                                        # Reset tool-use tracking flags
                                        tool_use_active = False
                                        tool_input_buffer = ""
                                        current_tool_index = None

                                elif event.type == "message_delta": # Capture final usage data
                                    if hasattr(event, 'usage') and event.usage:
                                        final_usage_data = event.usage
                                        # Update web_search_requests_count if present in the delta's usage
                                        if hasattr(event.usage, 'server_tool_use') and isinstance(event.usage.server_tool_use, dict):
                                            current_api_search_count = event.usage.server_tool_use.get('web_search_requests', 0)
                                            if current_api_search_count > web_search_requests_count: # Ensure we take the highest reported
                                                web_search_requests_count = current_api_search_count
                                    if self.valves.DEBUG:
                                        print(f"{self.debug_prefix} Received message_delta: {event.delta}, Usage: {event.usage}")

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
                                        web_search_requests_count=web_search_requests_count, # Pass current search count
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
                        
                        # Override with API reported tokens if available from final message_delta
                        final_input_tokens = input_tokens # Keep originally calculated if not overridden
                        final_generated_tokens_for_cost = total_output_tokens # Keep self-calculated if not overridden
                        final_reasoning_tokens_for_status = reasoning_tokens # Keep self-calculated for status message

                        if final_usage_data:
                            if hasattr(final_usage_data, 'input_tokens') and final_usage_data.input_tokens is not None:
                                final_input_tokens = final_usage_data.input_tokens
                            if hasattr(final_usage_data, 'output_tokens') and final_usage_data.output_tokens is not None:
                                final_generated_tokens_for_cost = final_usage_data.output_tokens
                                # If API provides total output, we might not have separate reasoning tokens from API.
                                # For status, we can still show our self-calculated reasoning tokens.
                                # For cost, the API's output_tokens should be the source of truth.
                            
                            # Explicitly get the final web_search_requests_count from final_usage_data for the final call
                            if hasattr(final_usage_data, 'server_tool_use') and final_usage_data.server_tool_use is not None:
                                stu = final_usage_data.server_tool_use
                                if hasattr(stu, 'web_search_requests') and isinstance(stu.web_search_requests, int):
                                    web_search_requests_count = stu.web_search_requests
                                elif isinstance(stu, dict): # Fallback if it's a dict
                                    web_search_requests_count = stu.get('web_search_requests', web_search_requests_count)
                                
                            if self.valves.DEBUG:
                                print(f"{self.debug_prefix} Finalizing with API usage data: Input={final_input_tokens}, Output={final_generated_tokens_for_cost}, Updated web_search_count for final call: {web_search_requests_count}")
                        else:
                            if self.valves.DEBUG:
                                print(f"{self.debug_prefix} Finalizing with self-calculated usage data: Input={final_input_tokens}, Output={final_generated_tokens_for_cost}, Reasoning (status only)={final_reasoning_tokens_for_status}, Web Search Count: {web_search_requests_count}")


                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=final_input_tokens,
                            generated_tokens=final_generated_tokens_for_cost, # Use total for cost calculation (API or self-calc)
                            reasoning_tokens=final_reasoning_tokens_for_status, # Pass separate reasoning tokens for status message (self-calc)
                            web_search_requests_count=web_search_requests_count, # Pass final search count
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed else "Stopped",
                            persist_usage=True, # Persist final usage
                            context_messages_count=len(messages)
                        )

                        if self.valves.DEBUG:
                            print(
                                f"DEBUG Finalized stream (completed: {stream_completed}). Original self-calc gen: {generated_tokens}, orig self-calc reas: {reasoning_tokens}. Used for cost: Input={final_input_tokens}, Output={final_generated_tokens_for_cost}, Web Searches: {web_search_requests_count}"
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

                # Get web_search_requests count for batch mode
                web_search_requests_count_batch = res.get("usage", {}).get("server_tool_use", {}).get("web_search_requests", 0)

                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=None, # Batch mode doesn't explicitly track reasoning tokens here yet
                    web_search_requests_count=web_search_requests_count_batch,
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