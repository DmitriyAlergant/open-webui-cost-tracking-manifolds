"""
title: LiteLLM Pipe (Generic) with Cost Tracking
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
requirements: litellm==1.70.2
license: MIT
"""

import litellm
import asyncio
import time
import json
import sys
from typing import Union, Any, Awaitable, Callable, List
from fastapi.responses import StreamingResponse
from open_webui.utils.misc import get_messages_content

MODULE_USAGE_TRACKING = "function_module_usage_tracking"

class LiteLLMPipe:

    def __init__(self, debug=False, debug_logging_prefix="", litellm_settings=None):
        self.debug = debug
        self.debug_logging_prefix = debug_logging_prefix
        self.litellm_settings = litellm_settings if litellm_settings else {}
        # LiteLLM uses environment variables for API keys, which should be set by the main manifold.
        # Specific provider configs can also be done via litellm.AnthropicConfig etc. if needed,
        # or by passing them directly in the acompletion call if they are non-standard OpenAI params.

    async def chat_completion(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        cost_tracker_module_name = MODULE_USAGE_TRACKING
        if cost_tracker_module_name not in sys.modules:
            # Attempt to import if not loaded, assuming it's in the path
            try:
                __import__(cost_tracker_module_name)
            except ImportError as e:
                print(f"Failed to import {cost_tracker_module_name}: {e}")
                raise Exception(f"Module {cost_tracker_module_name} is not loaded and could not be imported.")

        if cost_tracker_module_name not in sys.modules:
            raise Exception(f"Module {cost_tracker_module_name} is still not loaded after import attempt.")
        
        cost_tracker_module = sys.modules[cost_tracker_module_name]

        model_for_cost_tracking = __metadata__.get("user_requested_model_id", body["model"])

        cost_tracking_manager = cost_tracker_module.CostTrackingManager(
            model=model_for_cost_tracking,
            __user__=__user__,
            __metadata__=__metadata__,
            task=__task__,
            debug=self.debug
        )

        # The model ID in `body["model"]` should already be the one LiteLLM expects (e.g., "anthropic/claude-3-opus-20240229")
        # as prepared by the main litellm.py manifold.
        payload = {**body}
        # LiteLLM handles non-OpenAI params like 'reasoning_effort' automatically.
        # Ensure 'model', 'messages' are present. Others like 'temperature', 'max_tokens' are standard.

        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(body["messages"])

        if self.debug:
            print(f"{self.debug_logging_prefix} LiteLLM Request Payload: {payload}")

        generate_thinking_block_flag = payload.pop("generate_thinking_block", False)

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
                if self.debug:
                    print(f"{self.debug_logging_prefix} LiteLLM returning streaming response")

                async def stream_generator():
                    streamed_content_buffer = ""
                    thinking_content_buffer = ""
                    generated_tokens_count = 0
                    reasoning_tokens_count = 0
                    is_thinking_block_active = False
                    final_usage_stats = None
                    response_stream = None
                    last_update_time = 0
                    stream_completed_successfully = False
                    first_chunk_received = False

                    try:
                        # Prepare LiteLLM call arguments
                        litellm_kwargs = {**payload}
                        # Custom parameters from self.litellm_settings can be added here if necessary
                        # For example, if settings contains api_base for a specific model:
                        # if 'api_base' in self.litellm_settings.get(payload['model'], {}):
                        #     litellm_kwargs['api_base'] = self.litellm_settings[payload['model']]['api_base']
                        
                        # Add provider-specific settings from LITELLM_SETTINGS valve if any
                        # These are passed as top-level kwargs to acompletion if they are not standard OpenAI params.
                        for key, value in self.litellm_settings.items():
                            if key not in litellm_kwargs: # Avoid overriding standard params like 'model', 'messages', 'stream'
                                litellm_kwargs[key] = value
                                if self.debug:
                                    print(f"{self.debug_logging_prefix} Added to LiteLLM call from LITELLM_SETTINGS: {key}={value}")


                        if generate_thinking_block_flag:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Starting LiteLLM async streaming with synthetic thinking block heartbeat....")
                            
                            think_start_json = {"choices": [{"index": 0, "delta": {"content": "<think>"}}]}
                            yield f"data: {json.dumps(think_start_json)}\n\n"
                            is_thinking_block_active = True
                            # Initial thinking message after <think> opening tag
                            initial_think_msg = "Thinking... Model thoughts will appear here if provided by the API.\n\n" 
                            think_content_json = {"choices": [{"index": 0, "delta": {"content": initial_think_msg}}]}
                            yield f"data: {json.dumps(think_content_json)}\n\n"
                            thinking_content_buffer += initial_think_msg # Add to buffer for counting

                            # Create the task for litellm.acompletion
                            acompletion_task = asyncio.create_task(litellm.acompletion(**litellm_kwargs))

                            while not acompletion_task.done():
                                await asyncio.sleep(3) # Heartbeat interval
                                if acompletion_task.done(): # Check again after sleep
                                    break 
                                if not first_chunk_received: # Only send heartbeat if no real content yet
                                    if self.debug:
                                        print(f"{self.debug_logging_prefix} Yielding synthetic still-thinking heartbeat...")
                                    heartbeat_msg = "\nStill thinking..."
                                    think_content_json = {"choices": [{"index": 0, "delta": {"content": heartbeat_msg}}]}
                                    yield f"data: {json.dumps(think_content_json)}\n\n"
                                    thinking_content_buffer += heartbeat_msg # Add to buffer
                                else:
                                    # Once first real chunk is received, we stop synthetic heartbeats
                                    # The actual stream processing loop below will handle further content.
                                    break
                            
                            response_stream = await acompletion_task # Get the stream or raise exception if task failed

                        else: # Not using synthetic thinking block
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Starting LiteLLM streaming request without synthetic thinking block...")
                            response_stream = await litellm.acompletion(**litellm_kwargs)
                        
                        last_update_time = time.time()
                        if self.debug:
                            print(f"{self.debug_logging_prefix} Starting to consume LiteLLM stream response...")

                        async for chunk in response_stream:
                            first_chunk_received = True # Mark that we've received data

                            print ("chunk: ", chunk)

                            # Try to get usage from the chunk (some providers might include it per chunk or at the end)
                            # LiteLLM's standard chunk format doesn't specify 'usage' directly in delta chunks.
                            # It's usually in the *final* non-delta response or a special final chunk.
                            # We will primarily rely on litellm.stream_chunk_builder or manual accumulation
                            # and then check for final_usage_stats if the provider sends it.

                            if not chunk.choices:
                                if hasattr(chunk, 'usage') and chunk.usage: # Handle cases like Groq where final chunk only has usage
                                    final_usage_stats = chunk.usage if isinstance(chunk.usage, dict) else chunk.usage.model_dump()
                                    if self.debug:
                                        print(f"{self.debug_logging_prefix} Received usage-only chunk: {final_usage_stats}")
                                continue

                            delta = chunk.choices[0].delta
                            actual_content = delta.content or ""
                            
                            # LiteLLM standardizes 'reasoning_content' and 'thinking_blocks' in the message object of the *response* model,
                            # not typically in each delta of a stream. However, some providers might stream it.
                            # We need to check if LiteLLM surfaces this in streaming deltas.
                            # Based on the example, it's on `response.choices[0].message.reasoning_content`
                            # For streaming, this might be on `chunk.choices[0].message.reasoning_content` if the chunk represents a full message part
                            # or `chunk.choices[0].delta.custom_fields['reasoning_content']` (less likely but possible)

                            # Let's assume for now that reasoning/thinking comes as part of the 'content' if streamed,
                            # or we capture it if it's a special field in delta.
                            # The primary example shows `reasoning_content` on the *message* object, not delta.

                            # For now, we'll assume thinking content is identified by the provider and streamed as part of 'content',
                            # or rely on the `generate_thinking_block_flag` for synthetic thinking blocks.
                            # If a provider *does* stream specific thinking blocks, we'd need to adapt to its format.

                            if actual_content:
                                if is_thinking_block_active and generate_thinking_block_flag: # If using synthetic and real content arrives
                                    # Close the synthetic thinking block IF it was ours
                                    think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                                    yield f"data: {json.dumps(think_end_json)}\n\n"
                                    is_thinking_block_active = False # Reset for potential provider-driven blocks

                                streamed_content_buffer += actual_content
                                content_json = {"choices": [{"index": 0, "delta": {"content": actual_content}}]}
                                yield f"data: {json.dumps(content_json)}\n\n"
                            
                            # Check for LiteLLM's standardized thinking/reasoning in the chunk if available
                            # This part is speculative as LiteLLM docs emphasize it on the final message object.
                            # However, if a provider streams it and LiteLLM normalizes it into the delta, we might find it here.
                            chunk_message = getattr(chunk.choices[0], 'message', None) # Full message object if present in chunk
                            provider_reasoning_content = None
                            if chunk_message and hasattr(chunk_message, 'reasoning_content'):
                                provider_reasoning_content = chunk_message.reasoning_content
                            elif hasattr(delta, 'reasoning_content'): # Check delta directly
                                 provider_reasoning_content = delta.reasoning_content
                            
                            # Also check for 'thinking_blocks' if streamed by Anthropic via LiteLLM
                            provider_thinking_blocks_text = ""
                            if chunk_message and hasattr(chunk_message, 'thinking_blocks') and chunk_message.thinking_blocks:
                                for tb in chunk_message.thinking_blocks:
                                    if isinstance(tb, dict) and tb.get('type') == 'thinking' and tb.get('thinking'):
                                        provider_thinking_blocks_text += tb['thinking']
                            elif hasattr(delta, 'thinking_blocks') and delta.thinking_blocks: # Check delta directly
                                for tb in delta.thinking_blocks:
                                    if isinstance(tb, dict) and tb.get('type') == 'thinking' and tb.get('thinking'):
                                        provider_thinking_blocks_text += tb['thinking']

                            effective_thinking_text = provider_reasoning_content or provider_thinking_blocks_text

                            if effective_thinking_text:
                                if not is_thinking_block_active:
                                    think_start_json = {"choices": [{"index": 0, "delta": {"content": "<think>"}}]}
                                    yield f"data: {json.dumps(think_start_json)}\n\n"
                                    is_thinking_block_active = True
                                
                                thinking_content_buffer += effective_thinking_text
                                think_delta_json = {"choices": [{"index": 0, "delta": {"content": effective_thinking_text}}]}
                                yield f"data: {json.dumps(think_delta_json)}\n\n"

                            # Periodic cost update
                            current_time = time.time()
                            if current_time - last_update_time >= 2:
                                current_gen_tok = cost_tracking_manager.count_tokens(streamed_content_buffer)
                                current_rea_tok = cost_tracking_manager.count_tokens(thinking_content_buffer)
                                
                                generated_tokens_count += current_gen_tok
                                reasoning_tokens_count += current_rea_tok

                                cost_tracking_manager.calculate_costs_update_status_and_persist(
                                    input_tokens=input_tokens,
                                    generated_tokens=generated_tokens_count,
                                    reasoning_tokens=reasoning_tokens_count,
                                    start_time=start_time,
                                    __event_emitter__=__event_emitter__,
                                    status="Streaming...",
                                    persist_usage=False,
                                    context_messages_count=len(body["messages"])
                                )
                                streamed_content_buffer = ""
                                thinking_content_buffer = ""
                                last_update_time = current_time
                            
                            # Check for final usage data if LiteLLM embeds it in the *last* choice/delta of the stream
                            # This is provider-dependent. Some might send it in `chunk.usage` (handled above) or `chunk.choices[0].usage`.
                            if hasattr(chunk.choices[0], 'usage') and chunk.choices[0].usage:
                                final_usage_stats = chunk.choices[0].usage if isinstance(chunk.choices[0].usage, dict) else chunk.choices[0].usage.model_dump()
                                if self.debug:
                                    print(f"{self.debug_logging_prefix} Received usage in-stream chunk.choices[0].usage: {final_usage_stats}")

                        # End of stream iteration
                        if is_thinking_block_active: # Close any open thinking block
                            think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                            yield f"data: {json.dumps(think_end_json)}\n\n"
                        
                        stream_completed_successfully = True
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        stream_completed_successfully = False
                        if self.debug:
                            print(f"{self.debug_logging_prefix} LiteLLM stream_generator aborted by client.")
                        # Do not re-raise CancelledError, just let it propagate to finally

                    except Exception as e:
                        stream_completed_successfully = False
                        _, _, tb = sys.exc_info()
                        line_number = tb.tb_lineno
                        error_message = f"Error in LiteLLM stream_generator on line {line_number}: {e}"
                        print(error_message)
                        try:
                            error_json = {"error": error_message, "type": "streaming_error"}
                            yield f"data: {json.dumps(error_json)}\n\n"
                        except Exception as yield_err:
                            print(f"Error yielding error message to client: {yield_err}")
                        # Do not re-raise here if we yielded an error, allow finally to run.
                        # If we want the main pipe function to catch it, re-raise e

                    finally:
                        if self.debug:
                            print(f"{self.debug_logging_prefix} Entering LiteLLM stream finally. Completed: {stream_completed_successfully}")

                        # Process any remaining buffer content for token counting
                        final_gen_tok = cost_tracking_manager.count_tokens(streamed_content_buffer)
                        final_rea_tok = cost_tracking_manager.count_tokens(thinking_content_buffer)
                        generated_tokens_count += final_gen_tok
                        reasoning_tokens_count += final_rea_tok

                        final_input_tokens_val = input_tokens
                        final_generated_tokens_val = generated_tokens_count
                        final_reasoning_tokens_val = reasoning_tokens_count
                        override_occurred = False

                        # LiteLLM's `acompletion` when streamed doesn't return a response object with final usage directly.
                        # Usage might be in the last chunk (handled if `final_usage_stats` got populated) or not available.
                        # The `litellm.stream_chunk_builder` can reconstruct the full response and *might* include usage.
                        # For now, we rely on `final_usage_stats` if it was captured.

                        if final_usage_stats:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Found final usage stats from stream: {final_usage_stats}")
                            
                            prompt_tok = final_usage_stats.get("prompt_tokens")
                            completion_tok = final_usage_stats.get("completion_tokens")
                            # LiteLLM standard response for usage doesn't explicitly break out reasoning tokens in the main 'usage' block.
                            # It might be part of 'completion_tokens' or available in a model-specific extension if the provider returns it.
                            # For now, we assume completion_tokens is the sum of actual output and reasoning for cost if not detailed otherwise.

                            if prompt_tok is not None:
                                final_input_tokens_val = prompt_tok
                                override_occurred = True
                            if completion_tok is not None:
                                final_generated_tokens_val = completion_tok # This would be the total output tokens
                                # If API provides total completion, and we counted reasoning separately, 
                                # we might need to adjust or decide which is more accurate.
                                # For now, if API gives completion_tokens, we use that for generated_tokens for cost,
                                # and our counted reasoning_tokens_count for display/status, unless API also details reasoning.
                                override_occurred = True
                            
                            # If the API provided completion_tokens, that's the billable amount.
                            # Our `reasoning_tokens_count` is for display/status unless the API gives a specific breakdown.
                            # If `completion_tok` is the sum of generated + reasoning, then `final_generated_tokens_val` is correct for cost.
                            # `final_reasoning_tokens_val` (our counted one) remains for display if no API override.

                            if override_occurred and self.debug:
                                print(f"{self.debug_logging_prefix} Overriding counted tokens with API stream stats: Input={final_input_tokens_val}, Generated(Total Output)={final_generated_tokens_val}")
                        elif self.debug:
                             print(f"{self.debug_logging_prefix} No final usage stats from API stream. Using calculated: Input={final_input_tokens_val}, Generated={final_generated_tokens_val}, Reasoning={final_reasoning_tokens_val}")

                        output_tokens_for_cost = final_generated_tokens_val # If overridden, this is API's total output. If not, it's our counted generated.
                        if not override_occurred: # If not overridden by API, our generated + reasoning is the total output
                            output_tokens_for_cost = final_generated_tokens_val + final_reasoning_tokens_val

                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=final_input_tokens_val,
                            generated_tokens=output_tokens_for_cost, # Total billable output
                            reasoning_tokens=final_reasoning_tokens_val, # Our counted reasoning for display/status
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed_successfully else "Stopped/Error",
                            persist_usage=True,
                            context_messages_count=len(body["messages"])
                        )
                        if self.debug:
                            print(f"{self.debug_logging_prefix} Finalized LiteLLM stream. Input: {final_input_tokens_val}, Output(cost): {output_tokens_for_cost}, Reasoning(display): {final_reasoning_tokens_val}")
                
                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            else: # BATCH REQUEST (Non-streaming)
                if self.debug:
                    print(f"{self.debug_logging_prefix} LiteLLM making batch request...")
                
                litellm_kwargs = {**payload}
                for key, value in self.litellm_settings.items():
                    if key not in litellm_kwargs:
                        litellm_kwargs[key] = value
                        if self.debug:
                            print(f"{self.debug_logging_prefix} Added to LiteLLM call from LITELLM_SETTINGS: {key}={value}")

                response = await litellm.acompletion(**litellm_kwargs)

                print("response: ", response)
                
                # Convert LiteLLM's ModelResponse to a dictionary that mirrors OpenAI structure for consistency if possible
                # The example response seems to be already in a compatible format.
                response_json = response.model_dump() # LiteLLM responses are Pydantic models

                if self.debug:
                    print(f"{self.debug_logging_prefix} LiteLLM batch response: {json.dumps(response_json, indent=2)}")

                api_usage = response_json.get("usage", {})
                api_input_tokens = api_usage.get("prompt_tokens", input_tokens)
                api_completion_tokens = api_usage.get("completion_tokens", 0)
                
                # Extract reasoning_content from the first choice's message
                # LiteLLM standardizes this here for non-streaming.
                api_reasoning_content = ""
                if response_json.get("choices") and len(response_json["choices"]) > 0:
                    message = response_json["choices"][0].get("message", {})
                    api_reasoning_content = message.get("reasoning_content", "")
                    # also check thinking_blocks for anthropic
                    thinking_blocks_text = ""
                    if message.get("thinking_blocks"):
                         for tb in message.get("thinking_blocks"):
                            if isinstance(tb, dict) and tb.get('type') == 'thinking' and tb.get('thinking'):
                                thinking_blocks_text += tb['thinking']
                    if thinking_blocks_text:
                        api_reasoning_content = (api_reasoning_content + "\n" + thinking_blocks_text).strip()
                
                # Count tokens for API-provided reasoning content
                api_reasoning_tokens = cost_tracking_manager.count_tokens(api_reasoning_content)

                # For batch, completion_tokens from API is total output. 
                # If reasoning tokens are part of it, we display our counted reasoning tokens separately.
                # The cost is based on api_completion_tokens.

                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=api_input_tokens,
                    generated_tokens=api_completion_tokens, # This is the billable amount
                    reasoning_tokens=api_reasoning_tokens,  # This is for display/status
                    start_time=start_time,
                    __event_emitter__=__event_emitter__,
                    status="Completed",
                    persist_usage=True,
                    context_messages_count=len(body["messages"])
                )

                # Add our counted reasoning tokens to the response for UI if not already there and if significant
                # This part is tricky as we don't want to duplicate if the model already includes it in its main content.
                # For now, we rely on LiteLLM's `reasoning_content` field in the message.
                # The UI would need to be adapted to look for `<think>` tags or this field.

                # To ensure the UI can display thinking, if we have api_reasoning_content,
                # and generate_thinking_block_flag was true, we could wrap the main content:
                # However, LiteLLM puts `reasoning_content` at the message level, not in `content`.
                # The UI needs to be aware of `message.reasoning_content` or `message.thinking_blocks`
                # or we inject <think>...</think> into the main content if `generate_thinking_block_flag` was true.

                if generate_thinking_block_flag and api_reasoning_content and response_json.get("choices") and len(response_json["choices"]) > 0:
                    # This is an attempt to inject <think> block for non-streaming if flag was set
                    # and reasoning content was found. This might be redundant if UI handles raw reasoning_content field.
                    main_content = response_json["choices"][0]["message"].get("content", "")
                    response_json["choices"][0]["message"]["content"] = f"<think>{api_reasoning_content}</think>\n{main_content}"
                    if self.debug:
                        print(f"{self.debug_logging_prefix} Injected <think> block into non-streaming response based on api_reasoning_content.")
                
                # If generate_thinking_block_flag was true but NO api_reasoning_content came from model,
                # we might inject a simple <think>Thinking complete.</think> for consistency, but this could be misleading.
                # Best to rely on actual model output for thinking content.

                return response_json

        except litellm.exceptions.AuthenticationError as e:
            print(f"{self.debug_logging_prefix} LiteLLM Authentication Error: {e}")
            # Try to provide a more user-friendly message
            error_detail = str(e)
            if "API key is not set" in error_detail or "auth" in error_detail.lower():
                # Attempt to determine which key might be missing based on the model
                model_name = body.get("model", "").lower()
                missing_key_provider = "the required provider (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY)"
                if "anthropic" in model_name or "claude" in model_name:
                    missing_key_provider = "Anthropic (ANTHROPIC_API_KEY)"
                elif "gpt-" in model_name or "openai" in model_name:
                    missing_key_provider = "OpenAI (OPENAI_API_KEY)"
                # Add more providers as needed
                
                user_message = f"Authentication failed. Please check if the API key for {missing_key_provider} is correctly set in the OpenWebUI manifold settings. Details: {error_detail}"
            else:
                user_message = f"LiteLLM Authentication Error: {e}"
            raise Exception(user_message) from e # Re-raise with a potentially more helpful message

        except litellm.exceptions.APIConnectionError as e:
            print(f"{self.debug_logging_prefix} LiteLLM API Connection Error: {e}")
            raise Exception(f"LiteLLM could not connect to the API. Please check the API endpoint and network connectivity. Details: {e}") from e
        
        except litellm.exceptions.RateLimitError as e:
            print(f"{self.debug_logging_prefix} LiteLLM Rate Limit Error: {e}")
            raise Exception(f"LiteLLM API rate limit exceeded. Please try again later. Details: {e}") from e

        except Exception as e:
            _, _, tb = sys.exc_info()
            line_number = tb.tb_lineno
            error_message = f"LiteLLM Pipe error on line {line_number}: {e}"
            print(error_message)
            # Update cost tracking for failed request
            cost_tracking_manager.calculate_costs_update_status_and_persist(
                input_tokens=input_tokens,
                generated_tokens=0,
                reasoning_tokens=0,
                start_time=start_time,
                __event_emitter__=__event_emitter__,
                status="Error",
                persist_usage=True, # Persist even on error to log the attempt and input tokens
                context_messages_count=len(body["messages"])
            )
            raise Exception(error_message) from e

# Minimal Pipe class for OpenWebUI to recognize this as a loadable module (though it's a helper)
class Pipe:
    def __init__(self):
        self.type = "manifold_helper" # Not a primary manifold
        self.id = "litellm_pipe_module"
        self.name = "LiteLLM Pipe Shared Module"
        pass

    def pipes(self) -> list[dict]:
        return [] # This module doesn't expose models directly 