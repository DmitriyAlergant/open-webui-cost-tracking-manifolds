"""
title: OpenAI-Compatible Pipe (Generic) with Cost Tracking
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
                timeout=600
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
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        # Initialize CostTrackingManager from "usage_tracking_util" function

        cost_tracker_module_name = MODULE_USAGE_TRACKING

        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is not loaded")
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        # Use the user-requested model ID from metadata if available, otherwise fallback to body["model"]
        model_for_cost_tracking = __metadata__.get("user_requested_model_id", body["model"])

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            model=model_for_cost_tracking, 
            __user__=__user__, 
            __metadata__=__metadata__, 
            task=__task__, 
            debug=self.debug
        )

        model_id = body["model"][body["model"].find(".") + 1 :]

        payload = {**body, "model": model_id}
        # Keep reasoning_effort if present
        # payload.pop("reasoning_effort", None) # Removed this line

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
                context_messages_count=len(body["messages"])
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
                    thinking_content_buffer = ""  # Buffer for thinking content
                    reasoning_tokens = 0  # Accumulator for reasoning tokens
                    is_thinking_block = False # Track if we are emitting thinking content
                    final_usage_stats = None # Store final usage stats if provided by the stream

                    try:
                        response = await self.async_client.chat.completions.create(**payload)

                        async for chunk in response:
                            # --- Check for usage FIRST ---
                            if hasattr(chunk, 'usage') and chunk.usage:
                                # Store the latest usage stats received. Convert Pydantic model if necessary.
                                final_usage_stats = chunk.usage if isinstance(chunk.usage, dict) else chunk.usage.model_dump() # Use model_dump
                                if self.debug:
                                    print(f"{self.debug_logging_prefix} Received usage stats in stream chunk: {final_usage_stats}")
                                # This chunk might ONLY contain usage, or usage + final delta. Proceed to check choices.


                            # --- Process content delta ---
                            if not chunk.choices:
                                # If no choices, this might be the final usage chunk, or an empty intermediary chunk.
                                # We've already captured usage if it existed in this chunk.
                                # Continue to the next chunk as there's no delta content to process.
                                continue

                            # If we reach here, chunk.choices exists.
                            delta = chunk.choices[0].delta

                            # Extract regular content
                            content = delta.content or ""
                            if content:
                                # If we were previously thinking, close the block
                                if is_thinking_block:
                                    think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                                    yield f"data: {json.dumps(think_end_json)}\n\n"
                                    is_thinking_block = False

                                streamed_content_buffer += content
                                content_json = {"choices": [{"index": 0, "delta": {"content": content}}]}
                                yield f"data: {json.dumps(content_json)}\n\n"


                            # Extract thinking/reasoning content (prioritize reasoning_content from LiteLLM example)
                            thinking_text = getattr(delta, 'reasoning_content', None)

                            # Fallback check for thinking_blocks (more complex structure)
                            if not thinking_text and hasattr(delta, 'thinking_blocks') and delta.thinking_blocks:
                                # Assuming the first block contains the relevant text
                                if isinstance(delta.thinking_blocks, list) and len(delta.thinking_blocks) > 0:
                                     thinking_block = delta.thinking_blocks[0]
                                     if isinstance(thinking_block, dict):
                                         thinking_text = thinking_block.get('thinking', None)


                            if thinking_text:
                                # If not already thinking, start the block
                                if not is_thinking_block:
                                    think_start_json = {"choices": [{"index": 0, "delta": {"content": "<think>"}}]}
                                    yield f"data: {json.dumps(think_start_json)}\n\n"
                                    is_thinking_block = True

                                thinking_content_buffer += thinking_text
                                # Yield thinking content delta (mimicking Anthropic's direct style)
                                think_content_json = {"choices": [{"index": 0, "delta": {"content": thinking_text}}]}
                                yield f"data: {json.dumps(think_content_json)}\n\n"


                            # Update status every ~1 second
                            current_time = time.time()
                            if current_time - last_update_time >= 1:
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
                                    context_messages_count=len(body["messages"])
                                )

                                streamed_content_buffer = ""
                                thinking_content_buffer = "" # Clear thinking buffer too
                                last_update_time = current_time

                        # If the stream ended while thinking, close the block
                        if is_thinking_block:
                            think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                            yield f"data: {json.dumps(think_end_json)}\n\n"

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
                        # Final token count for any remaining buffer content
                        counted_generated_tokens = generated_tokens + cost_tracking_manager.count_tokens(streamed_content_buffer)
                        counted_reasoning_tokens = reasoning_tokens + cost_tracking_manager.count_tokens(thinking_content_buffer)

                        # Initialize final tokens with counted values or pre-calculated input
                        final_input_tokens = input_tokens
                        final_generated_tokens = counted_generated_tokens
                        final_reasoning_tokens = counted_reasoning_tokens
                        override_occurred = False

                        # Override with final usage stats if available
                        if final_usage_stats:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Found final usage stats in stream: {final_usage_stats}")

                            reported_prompt_tokens = final_usage_stats.get("prompt_tokens")
                            reported_completion_tokens = final_usage_stats.get("completion_tokens")
                            # Use 'reasoning_tokens' if present in nested structure, otherwise default to None
                            completion_details = final_usage_stats.get("completion_tokens_details", {})
                            reported_reasoning_tokens = completion_details.get("reasoning_tokens")

                            if reported_prompt_tokens is not None:
                                final_input_tokens = reported_prompt_tokens
                                override_occurred = True
                            if reported_completion_tokens is not None:
                                final_generated_tokens = reported_completion_tokens
                                override_occurred = True
                            # Only override reasoning tokens if the key exists and has a value
                            if reported_reasoning_tokens is not None:
                                final_reasoning_tokens = reported_reasoning_tokens
                                override_occurred = True

                            if override_occurred and self.debug:
                                print(f"{self.debug_logging_prefix} Overriding counted tokens. Using final stats: Input={final_input_tokens}, Generated={final_generated_tokens}, Reasoning={final_reasoning_tokens}")


                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=final_input_tokens,
                            generated_tokens=final_generated_tokens,
                            reasoning_tokens=final_reasoning_tokens,
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed else "Stopped",
                            persist_usage=True,
                            context_messages_count=len(body["messages"])
                        )

                        if self.debug:
                             print(
                                 f"{self.debug_logging_prefix} Finalized stream (completed: {stream_completed}). Final Tokens - Input: {final_input_tokens}, Generated: {final_generated_tokens}, Reasoning: {final_reasoning_tokens}"
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
                    .get("  ", {})
                    .get("reasoning_tokens", 0)
                )

                # Add potential reasoning content counting for batch responses if usage doesn't provide it
                # This might be needed if 'reasoning_tokens' isn't in 'usage' for batch calls
                if reasoning_tokens == 0 and response_json.get('choices'):
                    try:
                        message_data = response_json['choices'][0].get('message', {})
                        # Check both reasoning_content and thinking_blocks based on LiteLLM example
                        reasoning_content = message_data.get('reasoning_content')
                        if not reasoning_content and message_data.get('thinking_blocks'):
                            thinking_blocks = message_data['thinking_blocks']
                            if isinstance(thinking_blocks, list) and len(thinking_blocks) > 0:
                                thinking_block = thinking_blocks[0]
                                if isinstance(thinking_block, dict):
                                    reasoning_content = thinking_block.get('thinking') # Extract text

                        if reasoning_content:
                             reasoning_tokens = cost_tracking_manager.count_tokens(reasoning_content)
                             if self.debug:
                                 print(f"{self.debug_logging_prefix} Manually counted reasoning tokens for batch response: {reasoning_tokens}")
                    except (IndexError, KeyError, AttributeError, TypeError) as e:
                         if self.debug:
                            print(f"{self.debug_logging_prefix} Could not extract/count reasoning_content from batch response: {e}")


                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=reasoning_tokens,
                    start_time=start_time,
                    __event_emitter__=__event_emitter__,
                    status="Completed",
                    persist_usage=True,
                    context_messages_count=len(body["messages"])
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