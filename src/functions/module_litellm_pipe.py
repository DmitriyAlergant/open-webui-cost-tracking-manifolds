"""
title: LiteLLM Pipe (Generic) with Cost Tracking
author: Dmitriy Alergant
author_url: https://github.com/DmitriyAlergant-t1a/open-webui-cost-tracking-manifolds
version: 0.1.0
required_open_webui_version: 0.3.17
requirements: litellm==1.70.2
license: MIT
"""

import asyncio
import time
import json
import sys
import re
from typing import Union, Any, Awaitable, Callable, List
from fastapi.responses import StreamingResponse
from open_webui.utils.misc import get_messages_content

import litellm


MODULE_USAGE_TRACKING = "function_module_usage_tracking"

class LiteLLMPipe:

    def __init__(self, debug=False, debug_logging_prefix="", litellm_settings=None):
        self.debug = debug
        self.debug_logging_prefix = debug_logging_prefix
        self.litellm_settings = litellm_settings if litellm_settings else {}
        
        # Configure LiteLLM settings - remove verbose mode to reduce excessive logging
        # Only enable LiteLLM verbose logging if explicitly requested
        # if self.debug and self.litellm_settings.get("enable_litellm_verbose", False):
        
        litellm.set_verbose = False

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

        # Prepare the payload for LiteLLM
        payload = {**body}

        input_content = get_messages_content(body["messages"])
        input_tokens = cost_tracking_manager.count_tokens(input_content) + len(body["messages"])

        if self.debug:
            print(f"{self.debug_logging_prefix} LiteLLM Request Payload: {payload}")

        generate_thinking_block_flag = payload.pop("generate_thinking_block", False)

        # Add LiteLLM-specific settings from valves
        for key, value in self.litellm_settings.items():
            if key not in payload:
                payload[key] = value
                if self.debug:
                    print(f"{self.debug_logging_prefix} Added to LiteLLM call from LITELLM_SETTINGS: {key}={value}")

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
                context_messages_count=len(body["messages"]),
                web_search_requests_count=0
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
                    # Citation buffering
                    pending_citations = []
                    citation_urls_seen = set()
                    # Tool call tracking for web search
                    tool_call_buffer = ""
                    web_search_requests_count = 0

                    try:
                        if generate_thinking_block_flag:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Starting LiteLLM async streaming with synthetic thinking block heartbeat....")
                            
                            think_start_json = {"choices": [{"index": 0, "delta": {"content": "<think>"}}]}
                            yield f"data: {json.dumps(think_start_json)}\n\n"
                            is_thinking_block_active = True
                            
                            initial_think_msg = "Thinking... Model thoughts will appear here if provided by the API.\n\n"
                            think_content_json = {"choices": [{"index": 0, "delta": {"content": initial_think_msg}}]}
                            yield f"data: {json.dumps(think_content_json)}\n\n"
                            thinking_content_buffer += initial_think_msg

                            acompletion_task = asyncio.create_task(litellm.acompletion(**payload))

                            while not acompletion_task.done():
                                await asyncio.sleep(3)
                                if acompletion_task.done():
                                    break 
                                if not first_chunk_received:
                                    if self.debug:
                                        print(f"{self.debug_logging_prefix} Yielding synthetic still-thinking heartbeat...")
                                    heartbeat_msg = "Still thinking...\n\n"
                                    think_content_json = {"choices": [{"index": 0, "delta": {"content": heartbeat_msg}}]}
                                    yield f"data: {json.dumps(think_content_json)}\n\n"
                                    thinking_content_buffer += heartbeat_msg
                                else:
                                    break
                            
                            response_stream = await acompletion_task

                        else:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Starting LiteLLM streaming request without synthetic thinking block...")
                            response_stream = await litellm.acompletion(**payload)
                        
                        last_update_time = time.time()
                        if self.debug:
                            print(f"{self.debug_logging_prefix} Starting to consume LiteLLM stream response...")

                        async for chunk in response_stream:
                            first_chunk_received = True

                            #if self.debug:
                                #print(f"{self.debug_logging_prefix} Streaming Chunk: {chunk}")

                            # Handle usage-only chunks
                            if hasattr(chunk, 'usage') and chunk.usage and not chunk.choices:
                                final_usage_stats = chunk.usage if isinstance(chunk.usage, dict) else chunk.usage.model_dump()
                                if self.debug:
                                    print(f"{self.debug_logging_prefix} Usage-only chunk: input={final_usage_stats.get('prompt_tokens', 'N/A')}, output={final_usage_stats.get('completion_tokens', 'N/A')}")
                                continue

                            if not chunk.choices:
                                continue

                            # Handle citations from any choice in the chunk first
                            for choice in chunk.choices:
                                choice_delta = choice.delta if hasattr(choice, 'delta') else None
                                if choice_delta and hasattr(choice_delta, 'provider_specific_fields') and choice_delta.provider_specific_fields:
                                    # Handle both 'citations' (array) and 'citation' (single object) formats
                                    citation_data = choice_delta.provider_specific_fields.get('citations')
                                    if not citation_data:
                                        single_citation = choice_delta.provider_specific_fields.get('citation')
                                        if single_citation:
                                            citation_data = [single_citation]  # Convert to list for uniform processing
                                    
                                    if citation_data and isinstance(citation_data, list):
                                        for citation in citation_data:
                                            if isinstance(citation, dict):
                                                title = citation.get('title', '')
                                                url = citation.get('url', '')
                                                if url and url not in citation_urls_seen:
                                                    citation_urls_seen.add(url)
                                                    pending_citations.append((title, url))
                                                    if self.debug:
                                                        print(f"{self.debug_logging_prefix} Citation: {title[:50]}{'...' if len(title) > 50 else ''} - {url}")

                            delta = chunk.choices[0].delta
                            actual_content = delta.content or ""
                            
                            # Handle tool calls (web search detection)
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tool_call in delta.tool_calls:
                                    if hasattr(tool_call, 'function') and tool_call.function:
                                        if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                            tool_call_buffer += tool_call.function.arguments
                                            
                                            # Try to parse accumulated arguments to extract query
                                            try:
                                                # Look for complete JSON query in the buffer
                                                query_match = re.search(r'"query":\s*"([^"]*)"', tool_call_buffer)
                                                if query_match:
                                                    search_query = query_match.group(1)
                                                    if search_query.strip():  # Only report if we have a meaningful query
                                                        search_message = f"\n🔍 **Web Search:** *{search_query}*\n\n"
                                                        search_json = {"choices": [{"index": 0, "delta": {"content": search_message}}]}
                                                        yield f"data: {json.dumps(search_json)}\n\n"
                                                        tool_call_buffer = "" # Reset buffer after reporting
                                                        web_search_requests_count += 1
                                                        if self.debug:
                                                            print(f"{self.debug_logging_prefix} Reported web search: {search_query}")
                                            except Exception as e:
                                                if self.debug:
                                                    print(f"{self.debug_logging_prefix} Error parsing tool call: {e}")
                            
                            if actual_content:
                                if is_thinking_block_active and generate_thinking_block_flag:
                                    think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                                    yield f"data: {json.dumps(think_end_json)}\n\n"
                                    is_thinking_block_active = False

                                # Add any pending citations to this content chunk
                                content_with_citations = actual_content
                                if pending_citations:
                                    citation_text = ""
                                    for title, url in pending_citations:
                                        display_name = title if title else url
                                        # Make citations more readable with better formatting
                                        citation_text += f" [[{display_name}]({url})]"
                                    content_with_citations += citation_text
                                    pending_citations.clear()
                                    if self.debug:
                                        print(f"{self.debug_logging_prefix} Added citations to content: {citation_text}")

                                streamed_content_buffer += content_with_citations
                                content_json = {"choices": [{"index": 0, "delta": {"content": content_with_citations}}]}
                                yield f"data: {json.dumps(content_json)}\n\n"
                            
                            # Check for reasoning content from the chunk
                            reasoning_content = getattr(delta, 'reasoning_content', None)
                            if not reasoning_content and hasattr(chunk.choices[0], 'message'):
                                reasoning_content = getattr(chunk.choices[0].message, 'reasoning_content', None)
                            
                            # Also check for thinking_blocks
                            thinking_blocks_text = ""
                            thinking_blocks = getattr(delta, 'thinking_blocks', None)
                            if not thinking_blocks and hasattr(chunk.choices[0], 'message'):
                                thinking_blocks = getattr(chunk.choices[0].message, 'thinking_blocks', None)
                            
                            if thinking_blocks:
                                for tb in thinking_blocks:
                                    if isinstance(tb, dict) and tb.get('type') == 'thinking' and tb.get('thinking'):
                                        thinking_blocks_text += tb['thinking']

                            effective_thinking_text = reasoning_content or thinking_blocks_text

                            if effective_thinking_text:
                                if self.debug:
                                    print(f"{self.debug_logging_prefix} Reasoning content: {len(effective_thinking_text)} chars")
                                
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
                                    context_messages_count=len(body["messages"]),
                                    web_search_requests_count=web_search_requests_count
                                )
                                streamed_content_buffer = ""
                                thinking_content_buffer = ""
                                last_update_time = current_time
                            
                            # Check for usage data in chunk
                            if hasattr(chunk, 'usage') and chunk.usage:
                                final_usage_stats = chunk.usage if isinstance(chunk.usage, dict) else chunk.usage.model_dump()
                                if self.debug:
                                    print(f"{self.debug_logging_prefix} Usage in chunk: input={final_usage_stats.get('prompt_tokens', 'N/A')}, output={final_usage_stats.get('completion_tokens', 'N/A')}")

                        # End of stream iteration
                        if is_thinking_block_active:
                            think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>"}}]}
                            yield f"data: {json.dumps(think_end_json)}\n\n"
                        
                        # Handle any remaining buffered citations
                        if pending_citations:
                            citation_text = ""
                            for title, url in pending_citations:
                                display_name = title if title else url
                                citation_text += f" [[{display_name}]({url})]"
                            pending_citations.clear()
                            
                            final_citation_json = {"choices": [{"index": 0, "delta": {"content": citation_text}}]}
                            yield f"data: {json.dumps(final_citation_json)}\n\n"
                            
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Added {len(pending_citations)} remaining citations")
                        
                        stream_completed_successfully = True
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        stream_completed_successfully = False
                        if self.debug:
                            print(f"{self.debug_logging_prefix} LiteLLM stream_generator aborted by client.")

                    except Exception as e:
                        stream_completed_successfully = False
                        _, _, tb = sys.exc_info()
                        line_number = tb.tb_lineno
                        error_message = f"Error in LiteLLM stream_generator on line {line_number}: {e}"
                        print(error_message)
                        
                        # Create a more user-friendly error message
                        user_friendly_error = str(e)
                        error_type = "streaming_error"
                        
                        # Handle specific LiteLLM error types
                        if "AuthenticationError" in user_friendly_error or "authentication" in user_friendly_error.lower():
                            user_friendly_error = "Authentication failed. Please check your API key."
                            error_type = "authentication_error"
                        elif "RateLimitError" in user_friendly_error or "rate limit" in user_friendly_error.lower():
                            user_friendly_error = "API rate limit exceeded. Please try again later."
                            error_type = "rate_limit_error"
                        elif "APIConnectionError" in user_friendly_error or "connection" in user_friendly_error.lower():
                            user_friendly_error = "Could not connect to the API. Please check your network connection."
                            error_type = "connection_error"
                        
                        try:
                            # Close any open thinking block before yielding error
                            if is_thinking_block_active:
                                think_end_json = {"choices": [{"index": 0, "delta": {"content": "</think>\n\n"}}]}
                                yield f"data: {json.dumps(think_end_json)}\n\n"
                                is_thinking_block_active = False
                            
                            # Yield user-friendly error message to client
                            error_json = {
                                "choices": [{"index": 0, "delta": {"content": f"**Error:** {user_friendly_error}"}}],
                                "error": error_message,
                                "type": error_type
                            }
                            yield f"data: {json.dumps(error_json)}\n\n"
                            yield "data: [DONE]\n\n"
                        except Exception as yield_err:
                            print(f"Error yielding error message to client: {yield_err}")
                        
                        # Re-raise the exception so it can be caught by the outer handler
                        # This ensures OpenWebUI gets proper error handling
                        raise e

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

                        if final_usage_stats:
                            if self.debug:
                                print(f"{self.debug_logging_prefix} Final usage from API: input={final_usage_stats.get('prompt_tokens', 'N/A')}, output={final_usage_stats.get('completion_tokens', 'N/A')}")
                            
                            prompt_tok = final_usage_stats.get("prompt_tokens")
                            completion_tok = final_usage_stats.get("completion_tokens")

                            if prompt_tok is not None:
                                final_input_tokens_val = prompt_tok
                                override_occurred = True
                            if completion_tok is not None:
                                final_generated_tokens_val = completion_tok
                                override_occurred = True
                            
                            if override_occurred and self.debug:
                                print(f"{self.debug_logging_prefix} Using API token counts: Input={final_input_tokens_val}, Generated={final_generated_tokens_val}")
                        elif self.debug:
                             print(f"{self.debug_logging_prefix} Using calculated tokens: Input={final_input_tokens_val}, Generated={final_generated_tokens_val}, Reasoning={final_reasoning_tokens_val}")

                        output_tokens_for_cost = final_generated_tokens_val
                        if not override_occurred:
                            output_tokens_for_cost = final_generated_tokens_val + final_reasoning_tokens_val

                        cost_tracking_manager.calculate_costs_update_status_and_persist(
                            input_tokens=final_input_tokens_val,
                            generated_tokens=output_tokens_for_cost,
                            reasoning_tokens=final_reasoning_tokens_val,
                            start_time=start_time,
                            __event_emitter__=__event_emitter__,
                            status="Completed" if stream_completed_successfully else "Stopped",
                            persist_usage=True,
                            context_messages_count=len(body["messages"]),
                            web_search_requests_count=web_search_requests_count
                        )
                        if self.debug:
                            print(f"{self.debug_logging_prefix} Stream completed. Final tokens - Input: {final_input_tokens_val}, Output: {output_tokens_for_cost}, Reasoning: {final_reasoning_tokens_val}")
                
                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            else: # BATCH REQUEST (Non-streaming)

                # TBD obtain from the usage JSON when LiteLLM supports it
                web_search_requests_count = 0
                
                if self.debug:
                    print(f"{self.debug_logging_prefix} LiteLLM making batch request...")

                response = await litellm.acompletion(**payload)

                if self.debug:
                    response_summary = {
                        "id": getattr(response, 'id', 'unknown'),
                        "model": getattr(response, 'model', 'unknown'),
                        "choices_count": len(response.choices) if response.choices else 0,
                        "has_usage": bool(hasattr(response, 'usage') and response.usage)
                    }
                    print(f"{self.debug_logging_prefix} Batch response: {response_summary}")

                # Convert LiteLLM's ModelResponse to a dictionary
                response_json = response.model_dump()

                if self.debug:
                    usage_info = response_json.get("usage", {})
                    print(f"{self.debug_logging_prefix} Usage: input={usage_info.get('prompt_tokens', 'N/A')}, output={usage_info.get('completion_tokens', 'N/A')}")

                api_usage = response_json.get("usage", {})
                api_input_tokens = api_usage.get("prompt_tokens", input_tokens)
                api_completion_tokens = api_usage.get("completion_tokens", 0)
                
                # Get reasoning tokens from API usage stats if available
                api_reasoning_tokens = 0
                completion_tokens_details = api_usage.get("completion_tokens_details", {})
                if completion_tokens_details and completion_tokens_details.get("reasoning_tokens") is not None:
                    api_reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
                    if self.debug:
                        print(f"{self.debug_logging_prefix} API reasoning tokens: {api_reasoning_tokens}")
                else:
                    # Fallback: Extract reasoning_content from the first choice's message and count manually
                    api_reasoning_content = ""
                    if response_json.get("choices") and len(response_json["choices"]) > 0:
                        choice = response_json["choices"][0]
                        
                        # Check for reasoning_content at choice level
                        api_reasoning_content = choice.get("reasoning_content", "")
                        
                        # Check message level as well
                        message = choice.get("message", {})
                        if not api_reasoning_content:
                            api_reasoning_content = message.get("reasoning_content", "")
                        
                        # Also check thinking_blocks
                        thinking_blocks_text = ""
                        thinking_blocks = choice.get("thinking_blocks") or message.get("thinking_blocks")
                        if thinking_blocks:
                             for tb in thinking_blocks:
                                if isinstance(tb, dict) and tb.get('type') == 'thinking' and tb.get('thinking'):
                                    thinking_blocks_text += tb['thinking']
                        
                        if thinking_blocks_text:
                            api_reasoning_content = (api_reasoning_content + "\n" + thinking_blocks_text).strip()
                    
                    # Count tokens for API-provided reasoning content as fallback
                    api_reasoning_tokens = cost_tracking_manager.count_tokens(api_reasoning_content)
                    if self.debug:
                        print(f"{self.debug_logging_prefix} Calculated reasoning tokens: {api_reasoning_tokens}")
                
                # Extract reasoning content for thinking block injection (regardless of token counting method)
                api_reasoning_content = ""
                if response_json.get("choices") and len(response_json["choices"]) > 0:
                    choice = response_json["choices"][0]
                    
                    # Check for reasoning_content at choice level
                    api_reasoning_content = choice.get("reasoning_content", "")
                    
                    # Check message level as well
                    message = choice.get("message", {})
                    if not api_reasoning_content:
                        api_reasoning_content = message.get("reasoning_content", "")
                    
                    # Also check thinking_blocks
                    thinking_blocks_text = ""
                    thinking_blocks = choice.get("thinking_blocks") or message.get("thinking_blocks")
                    if thinking_blocks:
                         for tb in thinking_blocks:
                            if isinstance(tb, dict) and tb.get('type') == 'thinking' and tb.get('thinking'):
                                thinking_blocks_text += tb['thinking']
                    
                    if thinking_blocks_text:
                        api_reasoning_content = (api_reasoning_content + "\n" + thinking_blocks_text).strip()
                
                cost_tracking_manager.calculate_costs_update_status_and_persist(
                    input_tokens=api_input_tokens,
                    generated_tokens=api_completion_tokens,
                    reasoning_tokens=api_reasoning_tokens,
                    start_time=start_time,
                    __event_emitter__=__event_emitter__,
                    status="Completed",
                    persist_usage=True,
                    context_messages_count=len(body["messages"]),
                    web_search_requests_count=web_search_requests_count
                )

                # Inject thinking block if requested and reasoning content is available
                if generate_thinking_block_flag and api_reasoning_content and response_json.get("choices") and len(response_json["choices"]) > 0:
                    main_content = response_json["choices"][0]["message"].get("content", "")
                    response_json["choices"][0]["message"]["content"] = f"<think>{api_reasoning_content}</think>\n{main_content}"
                    if self.debug:
                        print(f"{self.debug_logging_prefix} Injected thinking block in batch response")

                return response_json

        except Exception as e:
            error_message = str(e)
            
            # Handle LiteLLM-specific exceptions
            if "AuthenticationError" in error_message or "authentication" in error_message.lower():
                model_name = body.get("model", "").lower()
                missing_key_provider = "the required provider (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY)"
                if "anthropic" in model_name or "claude" in model_name:
                    missing_key_provider = "Anthropic (ANTHROPIC_API_KEY)"
                elif "gpt-" in model_name or "openai" in model_name:
                    missing_key_provider = "OpenAI (OPENAI_API_KEY)"
                
                user_message = f"Authentication failed. Please check if the API key for {missing_key_provider} is correctly set in the OpenWebUI manifold settings. Details: {error_message}"
                
            elif "RateLimitError" in error_message or "rate limit" in error_message.lower():
                user_message = f"LiteLLM API rate limit exceeded. Please try again later. Details: {error_message}"
                
            elif "APIConnectionError" in error_message or "connection" in error_message.lower():
                user_message = f"LiteLLM could not connect to the API. Please check the API endpoint and network connectivity. Details: {error_message}"
                
            else:
                _, _, tb = sys.exc_info()
                line_number = tb.tb_lineno
                user_message = f"LiteLLM Pipe error on line {line_number}: {error_message}"
            
            print(user_message)
            
            # Update cost tracking for failed request
            cost_tracking_manager.calculate_costs_update_status_and_persist(
                input_tokens=input_tokens,
                generated_tokens=0,
                reasoning_tokens=0,
                start_time=start_time,
                __event_emitter__=__event_emitter__,
                status="Error",
                persist_usage=True,
                context_messages_count=len(body["messages"]),
                web_search_requests_count=web_search_requests_count
            )
            raise Exception(user_message) from e

# Minimal Pipe class for OpenWebUI to recognize this as a loadable module
class Pipe:
    def __init__(self):
        self.type = "manifold_helper"
        self.id = "litellm_pipe_module"
        self.name = "LiteLLM Pipe Shared Module"
        pass

    def pipes(self) -> list[dict]:
        return [] 