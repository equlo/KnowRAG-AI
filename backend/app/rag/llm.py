"""
llm.py - Step 6 of RAG: GENERATE the answer with Claude (the generative AI part).

`ClaudeLLM` wraps Anthropic's official Python SDK and exposes two methods:
  * stream_answer() - streams the answer token-by-token so the UI can show it live;
  * complete()      - a short non-streaming call (used to rewrite follow-up questions).

Both convert SDK objects and SDK errors into plain dictionaries ("events") so
the rest of the app never has to know which LLM provider is behind it.
"""

# AsyncIterator is the type of an `async def` function that `yield`s values.
from collections.abc import AsyncIterator

# Protocol lets tests swap in a fake LLM with the same methods.
from typing import Any, Protocol

# Anthropic's official SDK for the Claude API.
import anthropic

# Our settings (model, effort, max tokens, fallback flag).
from app.config import Settings

# Beta flag that enables `fallbacks="default"`: if Claude's safety classifiers
# decline a request, the API re-runs it on Anthropic's recommended fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLM(Protocol):
    """Interface implemented by ClaudeLLM (and by the fake LLM in the tests)."""

    # Stream an answer as a series of event dictionaries.
    def stream_answer(self, system: str, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]: ...

    # Return a short complete answer as one string.
    async def complete(self, system: str, messages: list[dict[str, Any]]) -> str: ...


class ClaudeLLM:
    """Generates text with Claude through the Anthropic Messages API."""

    def __init__(self, settings: Settings):
        # Unwrap the secret key if one was configured (in the environment or .env).
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        # The async client lets one server process handle many chats concurrently.
        # With api_key=None the SDK looks up credentials itself.
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        # Keep the settings for later requests.
        self._settings = settings

    def _common_options(self, effort: str) -> dict[str, Any]:
        """Request parameters shared by both methods."""
        # The model to call and how much reasoning effort it should spend.
        # (Claude Opus 5.5 always uses adaptive thinking; `effort` is the dial.)
        options: dict[str, Any] = {
            "model": self._settings.claude_model,
            "output_config": {"effort": effort},
        }
        # Optionally opt in to server-side refusal fallbacks (Claude API feature).
        if self._settings.claude_refusal_fallback:
            # The beta header that unlocks the feature...
            options["betas"] = [FALLBACK_BETA]
            # ...and "default" lets Anthropic pick the right fallback model.
            options["fallbacks"] = "default"
        # Hand the options back to the caller.
        return options

    async def stream_answer(self, system: str, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Stream Claude's answer, yielding {"type": ...} event dictionaries."""
        # Any network/API problem is turned into an "error" event instead of a crash.
        try:
            # Open a streaming request. `client.beta.messages` is needed because
            # `betas`/`fallbacks` are beta parameters.
            async with self._client.beta.messages.stream(
                # Upper bound on generated tokens (thinking + answer).
                max_tokens=self._settings.claude_max_tokens,
                # Role and rules for the assistant.
                system=system,
                # Conversation so far, ending with the sources + question.
                messages=messages,
                # Model, effort and fallback options.
                **self._common_options(self._settings.claude_effort),
            ) as stream:
                # The SDK yields both raw API events and convenient derived events.
                async for event in stream:
                    # "text" events carry the next piece of answer text.
                    if event.type == "text":
                        # Forward it to the browser.
                        yield {"type": "token", "text": event.text}
                    # A "fallback" block marks the point where another model took over.
                    elif event.type == "content_block_start" and event.content_block.type == "fallback":
                        # Tell the user which model is continuing the answer.
                        yield {"type": "notice", "text": f"Answer continued by {event.content_block.to.model}."}
                # After the stream ends, get the fully assembled message.
                final = await stream.get_final_message()
        # No API key configured: the SDK raises TypeError before sending anything.
        except TypeError as exc:
            # Only translate the authentication case; other TypeErrors are real bugs.
            if "authentication" not in str(exc):
                raise
            # Tell the user how to fix it.
            yield {"type": "error", "message": "No Anthropic API key configured. Set ANTHROPIC_API_KEY in backend/.env."}
            return
        # The key exists but is wrong or revoked (HTTP 401).
        except anthropic.AuthenticationError:
            yield {"type": "error", "message": "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."}
            return
        # Too many requests (HTTP 429); the SDK already retried a couple of times.
        except anthropic.RateLimitError:
            yield {"type": "error", "message": "Rate limited by the Claude API. Please wait a moment and retry."}
            return
        # Any other HTTP error from the API (400 bad request, 5xx server error, ...).
        except anthropic.APIStatusError as exc:
            yield {"type": "error", "message": f"Claude API error {exc.status_code}: {exc.message}"}
            return
        # Network problems: DNS failure, timeout, connection reset...
        except anthropic.APIConnectionError:
            yield {"type": "error", "message": "Could not reach the Claude API. Check your internet connection."}
            return

        # Safety classifiers (or the model itself) declined, even after any fallback.
        if final.stop_reason == "refusal":
            yield {"type": "error", "message": "Claude declined to answer this request."}
            return
        # The answer hit the max_tokens ceiling and was cut off.
        if final.stop_reason == "max_tokens":
            yield {"type": "notice", "text": "The answer was truncated (KNOWRAG_CLAUDE_MAX_TOKENS reached)."}
        # Finish with usage statistics so the UI can show model and token counts.
        yield {
            "type": "done",
            "model": final.model,
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
        }

    async def complete(self, system: str, messages: list[dict[str, Any]]) -> str:
        """One short, non-streaming request. Returns "" on any failure."""
        # Failures here are non-fatal: the caller falls back to the original question.
        try:
            # A small request at low effort - rewriting a query is an easy task.
            message = await self._client.beta.messages.create(
                # Plenty for a one-line query plus brief thinking.
                max_tokens=2048,
                # Instructions for the task.
                system=system,
                # The input to work on.
                messages=messages,
                # Same model/fallback settings, but low effort for speed.
                **self._common_options("low"),
            )
        # Any SDK error (bad key, rate limit, network...) -> give up quietly.
        except anthropic.AnthropicError:
            return ""
        # Missing API key -> also give up quietly; other TypeErrors are real bugs.
        except TypeError as exc:
            if "authentication" not in str(exc):
                raise
            return ""
        # A declined request has no usable text.
        if message.stop_reason == "refusal":
            return ""
        # Join all text blocks (thinking blocks are skipped) and trim whitespace.
        return "".join(block.text for block in message.content if block.type == "text").strip()
