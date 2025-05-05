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

        # Check for and remove the generate_thinking_block flag
        generate_thinking_block = payload.pop("generate_thinking_block", False)


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
                    thinking_content_buffer = "" # Buffer for thinking content
                    generated_tokens = 0 # Accumulator for periodic updates (non-heartbeat)
                    reasoning_tokens = 0 # Accumulator for periodic updates (non-heartbeat)
                    is_thinking_block = False # Track if we are emitting thinking content
                    final_usage_stats = None # Store final usage stats if provided by the stream
                    response_stream = None # Placeholder for the stream object
                    last_update_time = 0 # Relevant for non-heartbeat periodic updates
                    stream_completed = False # Track if the stream finished cleanly

                    try:
                        if generate_thinking_block:
                            # --- New Heartbeat Streaming Logic ---
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Starting async streaming request with synthetic thinking block heartbeat....")

                                print(f"Yielding thinking block...")


                            # Yield initial thinking message
                            think_start_json = {"choices": [{"index": 0, "delta": {"content": "<think>"}}]}
                            yield f"data: {json.dumps(think_start_json)}\n\n"
                            think_content_json = {"choices": [{"index": 0, "delta": {"content": "Thinking... Model thoughts are not shown via this API."}}]}
                            yield f"data: {json.dumps(think_content_json)}\n\n"
                            is_thinking_block = True 

                            create_task = asyncio.create_task(self.async_client.chat.completions.create(**payload))
                            # Emit heartbeat every 5 seconds until the call completes
                            while True:
                                try:
                                    done, pending = await asyncio.wait({create_task}, timeout=5)
                                    if create_task in done:
                                        response_stream = create_task.result() # This gets the stream object
                                        break

                                    # Emit the still-thinking heartbeat message
                                    if self.debug:
                                        print(f"Yielding still-thinking heartbeat message...")
                                    think_content_json = {"choices": [{"index": 0, "delta": {"content": "\nStill thinking..."}}]}
                                    yield f"data: {json.dumps(think_content_json)}\n\n"
                                except asyncio.CancelledError:
                                    if self.debug:
                                        print(f"{self.debug_logging_prefix} Heartbeat loop cancelled.")
                                    raise 
                                except Exception as e:
                                    if self.debug:
                                        print(f"{self.debug_logging_prefix} Error during heartbeat wait: {e}")
                                    raise 

                                # This "thinking" block will be closed below based on the "is_thinking_block" variable, similar to providers who generate their own thinking blocks that we capture via the stream

                        else:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Starting streaming request...")
                            response_stream = await self.async_client.chat.completions.create(**payload)
                            last_update_time = time.time() # Initialize for periodic updates

                        if self.debug:
                            print(f"{self.debug_logging_prefix} Starting to consume stream response...")

                        async for chunk in response_stream:
                            # --- Check for usage FIRST ---
                            if hasattr(chunk, 'usage') and chunk.usage:
                                # Store the latest usage stats received. Convert Pydantic model if necessary.
                                final_usage_stats = chunk.usage if isinstance(chunk.usage, dict) else chunk.usage.model_dump() # Use model_dump
                                # if self.debug:
                                #     print(f"{self.debug_logging_prefix} Received usage stats in stream chunk: {final_usage_stats}")


                            # Some kind of an empty chunk - ignore
                            if not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta

                            # Extract regular content
                            content = delta.content or ""
                            if content:
                                # If we were previously thinking, close the block FIRST
                                if is_thinking_block:
                                    think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                                    yield f"data: {json.dumps(think_end_json)}\n\n"
                                    is_thinking_block = False # Ensure it's only closed once

                                # Now yield the actual content
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


                            # Update status every ~2 seconds
                            current_time = time.time()
                            if current_time - last_update_time >= 2:
                                current_generated = cost_tracking_manager.count_tokens(streamed_content_buffer)
                                current_reasoning = cost_tracking_manager.count_tokens(thinking_content_buffer)

                                generated_tokens += current_generated
                                reasoning_tokens += current_reasoning

                                cost_tracking_manager.calculate_costs_update_status_and_persist(
                                    input_tokens=input_tokens,
                                    generated_tokens=generated_tokens,
                                    reasoning_tokens=reasoning_tokens,
                                    start_time=start_time,
                                    __event_emitter__=__event_emitter__,
                                    status="Streaming...",
                                    persist_usage=False,
                                    context_messages_count=len(body["messages"])
                                )

                                streamed_content_buffer = ""
                                thinking_content_buffer = "" # Clear thinking buffer too
                                last_update_time = current_time

                        # If the stream ended while thinking, close the thinkingblock
                        if is_thinking_block:
                            think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                            yield f"data: {json.dumps(think_end_json)}\n\n"

                        stream_completed = True
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        stream_completed = False
                        if self.debug:
                            print(f"{self.debug_logging_prefix} stream_response wrapper aborted by client")

                    except Exception as e:
                        stream_completed = False # Mark as errored
                        _, _, tb = sys.exc_info()
                        line_number = tb.tb_lineno
                        print(f"Error in stream_generator's common processing loop on line {line_number}: {e}")

                        # Yield error message to client
                        error_json = {"error": f"Server error during streaming: {e}"}
                        try:
                            yield f"data: {json.dumps(error_json)}\n\n"
                        except Exception as yield_err:
                             print(f"Error yielding error message to client: {yield_err}")
                        raise e # Re-raise the original exception

                    finally:
                        if self.debug:
                             print(f"{self.debug_logging_prefix} Entering stream finally block. Stream completed: {stream_completed}")

                        # Final token count for any remaining buffer content
                        final_counted_generated = cost_tracking_manager.count_tokens(streamed_content_buffer)
                        final_counted_reasoning = cost_tracking_manager.count_tokens(thinking_content_buffer)

                        # Add remaining buffer counts to accumulated totals (relevant for non-heartbeat)
                        generated_tokens += final_counted_generated
                        reasoning_tokens += final_counted_reasoning

                        # Initialize final tokens with accumulated/counted values or pre-calculated input
                        final_input_tokens = input_tokens
                        # Use accumulated tokens if not heartbeat, otherwise use final buffer count directly
                        final_generated_tokens = generated_tokens if not generate_thinking_block else final_counted_generated
                        final_reasoning_tokens = reasoning_tokens if not generate_thinking_block else final_counted_reasoning
                        override_occurred = False

                        # Override with final usage stats if available (applies to both branches)
                        if final_usage_stats:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Found final usage stats in stream: {final_usage_stats}")

                            reported_prompt_tokens = final_usage_stats.get("prompt_tokens")
                            reported_completion_tokens = final_usage_stats.get("completion_tokens")
                            # Handle potential nested structure for reasoning tokens
                            completion_details = final_usage_stats.get("completion_tokens_details") or {}
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
                            # --- Infer reasoning tokens if not provided but total is --- >
                            elif (reported_reasoning_tokens is None or reported_reasoning_tokens == 0) and \
                                 reported_prompt_tokens is not None and \
                                 reported_completion_tokens is not None:
                                reported_total_tokens = final_usage_stats.get("total_tokens")
                                if reported_total_tokens is not None:
                                    inferred_reasoning = reported_total_tokens - reported_prompt_tokens - reported_completion_tokens
                                    # Use a small threshold (e.g., > 5) to avoid tiny rounding errors causing inference
                                    if inferred_reasoning > 5:
                                        final_reasoning_tokens = inferred_reasoning
                                        override_occurred = True # Mark as override since we used API stats
                                        if self.debug:
                                            print(f"{self.debug_logging_prefix} Inferred reasoning tokens from total: {reported_total_tokens} - {reported_prompt_tokens} - {reported_completion_tokens} = {inferred_reasoning}")
                            # <------------------------------------------------------------

                            if override_occurred and self.debug:
                                print(f"{self.debug_logging_prefix} Overriding counted tokens. Using final stats: Input={final_input_tokens}, Generated={final_generated_tokens}, Reasoning={final_reasoning_tokens}")

                        # If no override occurred, we rely on the final counted tokens from the buffers.
                        elif not override_occurred and self.debug:
                            # For non-heartbeat, this uses the accumulated totals. For heartbeat, it uses the final buffer count.
                            print(f"{self.debug_logging_prefix} No final usage stats from API or no override occurred. Using final calculated tokens: Input={final_input_tokens} (estimated), Generated={final_generated_tokens}, Reasoning={final_reasoning_tokens}")


                        # Determine the final output tokens count for cost calculation
                        output_tokens_for_cost = 0
                        if override_occurred:
                            # Assume API's completion_tokens is the total billable output
                            output_tokens_for_cost = final_generated_tokens
                        else:
                            # Sum our counted tokens if no API stats were used
                            output_tokens_for_cost = final_generated_tokens + final_reasoning_tokens

                        # Final cost calculation and persistence
                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=final_input_tokens,
                            generated_tokens=output_tokens_for_cost, # Use the determined value for cost
                            reasoning_tokens=final_reasoning_tokens, # Pass separate reasoning for status message
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed else "Stopped",
                            persist_usage=True, # Persist final usage
                            context_messages_count=len(body["messages"])
                        )

                        if self.debug:
                             print(
                                 f"{self.debug_logging_prefix} Finalized stream (completed: {stream_completed}). Final Tokens - Input: {final_input_tokens}, Generated (for cost): {output_tokens_for_cost}, Reasoning (for display): {final_reasoning_tokens}"
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

                # Correctly extract tokens, noting that reasoning_tokens is nested
                usage_details = response_json.get("usage", {})
                completion_details = usage_details.get("completion_tokens_details", {}) or {}

                input_tokens = usage_details.get("prompt_tokens", input_tokens) # Use estimated if not provided
                generated_tokens = usage_details.get("completion_tokens", 0)
                reasoning_tokens = completion_details.get("reasoning_tokens", 0) # Get from nested details

                # Infer reasoning tokens if not provided but total is and difference > 50
                if (reasoning_tokens is None or reasoning_tokens == 0) and \
                   input_tokens is not None and generated_tokens is not None:

                    total_tokens = usage_details.get("total_tokens")

                    if total_tokens is not None:
                        inferred_reasoning = total_tokens - input_tokens - generated_tokens
                        if inferred_reasoning > 50: # Apply the threshold
                            reasoning_tokens = inferred_reasoning
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Inferred reasoning tokens for batch from total: {total_tokens} - {input_tokens} - {generated_tokens} = {inferred_reasoning}")

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