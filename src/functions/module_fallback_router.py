"""
title: Module: fallback router
author:
author_url:
version: 0.1.0
required_open_webui_version: 0.6.9
license: MIT
"""

from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from typing import Union, Any, Awaitable, Callable, AsyncIterator
import sys
import asyncio
from open_webui.models.functions import Functions


class StreamingResponseWrapper:
    """Simplified wrapper that tests the first chunk of a streaming response and implements fallback"""

    def __init__(
        self,
        response: StreamingResponse,
        fallback_func: Callable,
        debug_prefix: str = "",
        debug: bool = False,
        debug_logging_prefix: str = "",
    ):
        self.response = response
        self.fallback_func = fallback_func
        self.debug_prefix = debug_prefix
        self.debug = debug
        self.debug_logging_prefix = debug_logging_prefix

    async def generate_with_fallback(self):
        """Generate response with fallback logic - only fallback on first chunk failure"""

        iterator = self.response.body_iterator

        try:
            # Test the first chunk - this is where fallback can happen
            first_chunk = await iterator.__anext__()

            if self.debug:
                print(
                    f"{self.debug_logging_prefix} primary received first chunk, continue streaming without fallback"
                )

            # If we get here, the first chunk succeeded - yield it and continue
            yield first_chunk

        except Exception as e:

            print(
                f"{self.debug_logging_prefix} primary chunk failed, attempting fallback. Exception: {e}"
            )

            try:
                fallback_response = await self.fallback_func()

                if isinstance(fallback_response, StreamingResponse):
                    async for chunk in fallback_response.body_iterator:
                        yield chunk
                else:
                    yield fallback_response

            except Exception as fallback_error:
                print(
                    f"{self.debug_logging_prefix} Fallback also failed: {fallback_error}"
                )
                combined_error = Exception(
                    f"Primary error: {e}.\n Fallback error: {fallback_error}"
                )
                raise combined_error

        # Continue streaming the rest of the chunks without fallback protection
        try:
            async for chunk in iterator:
                yield chunk
        except Exception as e:
            # After first chunk, any exceptions are raised without fallback
            if self.debug:
                print(
                    f"{self.debug_logging_prefix} Streaming error after first chunk (no fallback): {e}"
                )
            raise e


class Pipe:
    class Valves(BaseModel):
        DEBUG: bool = Field(default=False, description="Display debugging messages")

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic_fallback"
        self.name = "anthropic_fallback/"

        self.valves = self.Valves()

        self.valves.DEBUG = Functions.get_function_valves_by_id(
            "module_fallback_router"
        ).get("DEBUG", False)

        print("self.valves", self.valves)

        self.debug_logging_prefix = "DEBUG:    " + __name__ + " - "

    def get_module(self, prefix: str):
        """Get module based on prefix by concatenating 'function_' with the prefix"""
        module_name = f"function_{prefix}"
        if module_name not in sys.modules:
            try:
                __import__(module_name)
            except ImportError as e:
                print(f"Failed to import {module_name}: {e}")
                raise Exception(
                    f"Module {module_name} is not loaded and could not be imported."
                )

        return sys.modules[module_name]

    def pipes(self) -> list[dict]:
        return []

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __task__,
    ) -> Union[str, StreamingResponse]:

        if self.valves.DEBUG:
            print(
                f"{self.debug_logging_prefix} Anthropic Manifold received request body: {body}"
            )

        # Get the original model ID and extract prefix
        original_model = body.get("model", "")
        model_prefix = original_model.split(".", 1)[0] if "." in original_model else ""
        model_id_without_prefix = (
            original_model.split(".", 1)[1] if "." in original_model else original_model
        )

        if not model_prefix:
            raise ValueError(
                f"Model {original_model} does not have a prefix (expected format: prefix.model_id)"
            )

        available_models_for_fallback = body.pop("available_models_for_fallback", [])

        # Find the model configuration
        model_config = next(
            (
                m
                for m in available_models_for_fallback
                if m["id"] == model_id_without_prefix
            ),
            None,
        )

        if not model_config:
            raise ValueError(
                f"Model {model_id_without_prefix} not found in available models"
            )

        # Check if this is a streaming request
        is_streaming = body.get("stream", False)

        if not is_streaming:
            # Non-streaming request - use original fallback logic
            return await self._handle_with_fallback(
                body,
                model_config,
                model_prefix,
                __user__,
                __metadata__,
                __event_emitter__,
                __task__,
            )
        else:
            # Streaming request - use wrapper logic
            return await self._handle_streaming_with_fallback(
                body,
                model_config,
                model_prefix,
                __user__,
                __metadata__,
                __event_emitter__,
                __task__,
            )

    async def _try_module(
        self,
        body: dict,
        model_id: str,
        module_type: str,
        __user__,
        __metadata__,
        __event_emitter__,
        __task__,
    ):
        """Try a module with the given model configuration"""
        if self.valves.DEBUG:
            print(
                f"{self.debug_logging_prefix} Trying {module_type} for model: {model_id}"
            )

        body_copy = body.copy()
        body_copy["model"] = model_id

        # Extract the module prefix from model_id
        module_prefix = model_id.split(".", 1)[0] if "." in model_id else model_id
        module = self.get_module(module_prefix)
        pipe = module.Pipe()

        # Load valves for pipe
        if hasattr(pipe, "valves") and hasattr(pipe, "Valves"):
            valves_data = Functions.get_function_valves_by_id(module_prefix)
            pipe.valves = pipe.Valves(**(valves_data if valves_data else {}))

        return await pipe.pipe(
            body=body_copy,
            __user__=__user__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __task__=__task__,
        )

    async def _try_primary(
        self,
        body: dict,
        model_config: dict,
        model_prefix: str,
        __user__,
        __metadata__,
        __event_emitter__,
        __task__,
    ):
        """Try primary with the given configuration"""
        return await self._try_module(
            body,
            model_config["primary_model_id"],
            "primary",
            __user__,
            __metadata__,
            __event_emitter__,
            __task__,
        )

    async def _try_secondary(
        self,
        body: dict,
        model_config: dict,
        model_prefix: str,
        __user__,
        __metadata__,
        __event_emitter__,
        __task__,
    ):
        """Try secondary with the given configuration"""
        secondary_model_id = model_config.get("secondary_model_id")
        if secondary_model_id is None:
            raise ValueError("No secondary model configured for fallback")

        return await self._try_module(
            body,
            secondary_model_id,
            "secondary",
            __user__,
            __metadata__,
            __event_emitter__,
            __task__,
        )

    def _has_secondary_model(self, model_config: dict) -> bool:
        """Check if a secondary model is configured"""
        return model_config.get("secondary_model_id") is not None

    async def _handle_with_fallback(
        self,
        body,
        model_config,
        model_prefix,
        __user__,
        __metadata__,
        __event_emitter__,
        __task__,
    ):
        """Handle requests with fallback logic (works for both streaming and non-streaming)"""
        # Try primary first
        try:
            return await self._try_primary(
                body,
                model_config,
                model_prefix,
                __user__,
                __metadata__,
                __event_emitter__,
                __task__,
            )

        except Exception as e:
            print(f"{self.debug_logging_prefix} primary failed with error: {e}")

            # Check if secondary model is available before attempting fallback
            if not self._has_secondary_model(model_config):
                if self.valves.DEBUG:
                    print(
                        f"{self.debug_logging_prefix} No secondary model configured, not attempting fallback"
                    )
                raise e  # Re-raise the original primary error

            print(f"{self.debug_logging_prefix} Falling back to secondary")

            try:
                return await self._try_secondary(
                    body,
                    model_config,
                    model_prefix,
                    __user__,
                    __metadata__,
                    __event_emitter__,
                    __task__,
                )

            except Exception as fallback_error:
                if self.valves.DEBUG:
                    print(
                        f"{self.debug_logging_prefix} secondary also failed with error: {fallback_error}"
                    )

                # If both fail, raise the original primary error with context
                raise Exception(
                    f"Both primary and secondary failed. primary error: {e}. secondary error: {fallback_error}"
                )

    async def _handle_streaming_with_fallback(
        self,
        body,
        model_config,
        model_prefix,
        __user__,
        __metadata__,
        __event_emitter__,
        __task__,
    ):
        """Handle streaming requests with first-chunk testing and fallback"""

        # Create closures that capture the context for the wrapper
        async def try_primary_closure():
            return await self._try_primary(
                body,
                model_config,
                model_prefix,
                __user__,
                __metadata__,
                __event_emitter__,
                __task__,
            )

        async def try_secondary_closure():
            return await self._try_secondary(
                body,
                model_config,
                model_prefix,
                __user__,
                __metadata__,
                __event_emitter__,
                __task__,
            )

        # Try primary first
        try:
            primary_response = await try_primary_closure()

            if isinstance(primary_response, StreamingResponse):
                # Only create wrapper with fallback if secondary model is available
                if self._has_secondary_model(model_config):
                    wrapper = StreamingResponseWrapper(
                        primary_response,
                        try_secondary_closure,
                        self.debug_logging_prefix,
                        self.valves.DEBUG,
                        self.debug_logging_prefix,
                    )

                    return StreamingResponse(
                        wrapper.generate_with_fallback(),
                        media_type=primary_response.media_type,
                        headers=primary_response.headers,
                    )
                else:
                    # No secondary model, return primary response without fallback wrapper
                    if self.valves.DEBUG:
                        print(
                            f"{self.debug_logging_prefix} No secondary model configured, streaming without fallback protection"
                        )
                    return primary_response
            else:
                # Non-streaming response from primary
                return primary_response

        except Exception as e:
            print(
                f"{self.debug_logging_prefix} primary failed immediately with error: {e}"
            )

            # Check if secondary model is available before attempting fallback
            if not self._has_secondary_model(model_config):
                if self.valves.DEBUG:
                    print(
                        f"{self.debug_logging_prefix} No secondary model configured, not attempting fallback"
                    )
                raise e  # Re-raise the original primary error

            print(f"{self.debug_logging_prefix} Falling back to secondary")

            # primary failed immediately, try secondary
            return await try_secondary_closure()
