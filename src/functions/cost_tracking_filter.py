"""
title: Cost Tracking Filter
author: 
author_url: 
version: 0.1.0
required_open_webui_version: 0.5.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Callable, Awaitable, List, Dict

import sys
import time
import json

MODULE_USAGE_TRACKING = "function_module_usage_tracking"

# This filter can work on top of basic out-of-the-box integrations with OpenAI (not through manifolder).
# However, it only works with Web UI chat requests, and API streaming (which are rarely used)
# Batch API requests do not currently call outlet filters, so these costs would not be tracked.

class Filter:
    class Valves(BaseModel):
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging"
        )

    def __init__(self):
        self.type = "filter"
        self.id = "cost_tracking_filter"
        self.name = "Cost Tracking Filter"
        self.valves = self.Valves()
        self.start_time = None
        self.input_tokens = 0
        self.cost_tracking_manager = None
        self.streamed_content_buffer = ""
        self.generated_tokens = 0
        self.last_update_time = 0
        self.messages_count = 0

    def _initialize_cost_tracker(self, body: dict, __user__: dict, __metadata__: dict, __task__):
        """Initialize the cost tracking manager if not already done"""
        if self.cost_tracking_manager is None:
            # Get the model name from the request
            model = body.get("model", "unknown")
            
            # Initialize CostTrackingManager from "usage_tracking_util" function
            cost_tracker_module_name = MODULE_USAGE_TRACKING
            
            if cost_tracker_module_name not in sys.modules:
                if self.valves.DEBUG:
                    print(f"Cost Tracking Filter: Module {cost_tracker_module_name} is not loaded")
                return False
                
            cost_tracker_module = sys.modules[cost_tracker_module_name]
            
            self.cost_tracking_manager = cost_tracker_module.CostTrackingManager(
                model=model,
                __user__=__user__,
                __metadata__=__metadata__,
                task=__task__,
                debug=self.valves.DEBUG
            )
            return True
        return True

    def inlet(self, body: dict, __user__: Optional[dict] = None, __metadata__: Optional[dict] = None, 
              __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None, __task__=None) -> dict:
        """Track the input tokens and start time"""
        
        # Initialize cost tracker if needed
        if not self._initialize_cost_tracker(body, __user__, __metadata__, __task__):
            return body
            
        # Reset tracking variables
        self.start_time = time.time()
        self.streamed_content_buffer = ""
        self.generated_tokens = 0
        self.last_update_time = 0
        
        # Calculate input tokens
        from open_webui.utils.misc import get_messages_content
        input_content = get_messages_content(body.get("messages", []))
        self.messages_count = len(body.get("messages", []))
        self.input_tokens = self.cost_tracking_manager.count_tokens(input_content) + self.messages_count
        
        # Update status
        if __event_emitter__:
            self.cost_tracking_manager.calculate_costs_update_status_and_persist(
                input_tokens=self.input_tokens,
                generated_tokens=None,
                reasoning_tokens=None,
                start_time=self.start_time,
                __event_emitter__=__event_emitter__,
                status="Requested...",
                persist_usage=False,
                context_messages_count=self.messages_count
            )
        
        if self.valves.DEBUG:
            print(f"Cost Tracking Filter: Inlet processed, input tokens: {self.input_tokens}")
            
        return body

    def stream(self, event: dict, __user__: Optional[dict] = None, __metadata__: Optional[dict] = None, 
               __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None, __task__=None) -> dict:
        """Track the streaming tokens and update costs"""
        
        if self.cost_tracking_manager is None:
            return event
            
        # Extract content from the event
        content = ""
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta and delta["content"] is not None:
                content += delta["content"]
        
        # Buffer content for costs calculation
        self.streamed_content_buffer += content
        
        # Update status approximately every second
        current_time = time.time()
        if __event_emitter__ and (current_time - self.last_update_time >= 1):
            # Count new tokens and add to total
            new_tokens = self.cost_tracking_manager.count_tokens(self.streamed_content_buffer)
            self.generated_tokens += new_tokens
            
            # Update costs and status
            self.cost_tracking_manager.calculate_costs_update_status_and_persist(
                input_tokens=self.input_tokens,
                generated_tokens=self.generated_tokens,
                reasoning_tokens=None,
                start_time=self.start_time,
                __event_emitter__=__event_emitter__,
                status="Streaming...",
                persist_usage=False,
                context_messages_count=self.messages_count
            )
            
            # Reset buffer and update last update time
            self.streamed_content_buffer = ""
            self.last_update_time = current_time
            
            if self.valves.DEBUG:
                print(f"Cost Tracking Filter: Stream update, generated tokens: {self.generated_tokens}")
        
        return event

    def outlet(self, body: dict, __user__: Optional[dict] = None, __metadata__: Optional[dict] = None, 
               __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None, __task__=None) -> dict:
        """Finalize cost tracking at the end of a request"""
        
        if self.cost_tracking_manager is None or self.start_time is None:
            return body
            
        # For streaming requests, we've been tracking tokens incrementally
        # For batch requests, we need to extract token counts from response if available
        
        is_streaming = False
        completion_tokens = 0
        reasoning_tokens = 0
        
        # Check if this was a streaming response (we already counted tokens)
        if self.generated_tokens > 0:
            is_streaming = True
            completion_tokens = self.generated_tokens
            
            # Process any remaining content in the buffer
            if self.streamed_content_buffer:
                new_tokens = self.cost_tracking_manager.count_tokens(self.streamed_content_buffer)
                completion_tokens += new_tokens
                self.streamed_content_buffer = ""
        
        # For batch responses, try to get token counts from response
        elif "usage" in body:
            usage = body.get("usage", {})
            # Prefer the reported input tokens if available
            if "prompt_tokens" in usage:
                self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
            
            completion_tokens = usage.get("completion_tokens", 0)
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        
        # If no token count available from response, estimate from content
        else:
            # Find assistant messages in the conversation
            for message in body.get("messages", []):
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    completion_tokens += self.cost_tracking_manager.count_tokens(content)
        
        # Final cost calculation and persisting
        if __event_emitter__:
            self.cost_tracking_manager.calculate_costs_update_status_and_persist(
                input_tokens=self.input_tokens,
                generated_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                start_time=self.start_time,
                __event_emitter__=__event_emitter__,
                status="Completed",
                persist_usage=True,
                context_messages_count=self.messages_count
            )
        
        if self.valves.DEBUG:
            print(f"Cost Tracking Filter: Outlet finalized, input tokens: {self.input_tokens}, " +
                  f"generated tokens: {completion_tokens}, reasoning tokens: {reasoning_tokens}")
        
        # Reset tracking variables
        self.start_time = None
        self.input_tokens = 0
        self.cost_tracking_manager = None
        self.streamed_content_buffer = ""
        self.generated_tokens = 0
        self.last_update_time = 0
        self.messages_count = 0
        
        return body 