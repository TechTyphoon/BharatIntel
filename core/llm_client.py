"""
BharatIntel — Unified LLM Client

Wraps litellm to provide a single interface for all LLM calls system-wide.

Responsibilities:
  - Async completion calls with retry + backoff
  - **Automatic fallback chain**: if primary model fails, tries fallback models
  - JSON response parsing and schema validation
  - Token usage tracking per call
  - Maps litellm exceptions to BharatIntel exception hierarchy
  - Model selection via config (no hardcoded model names)

Dependencies: litellm

Usage:
    from core.llm_client import LLMClient

    client = LLMClient(
        model="gemini/gemini-2.0-flash",
        fallback_models=["openrouter/google/gemini-2.0-flash-001", "openrouter/meta-llama/llama-3.3-70b-instruct"],
    )
    result = await client.complete(prompt="...", system="...")
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

import litellm

from core.exceptions import LLMError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from core.logger import get_logger

log = get_logger("llm_client")

# Suppress litellm's verbose default logging
litellm.suppress_debug_info = True
litellm.set_verbose = False

import logging as _logging
_logging.getLogger("LiteLLM").setLevel(_logging.WARNING)


class LLMClient:
    """
    Unified async LLM interface with automatic fallback chain.

    If the primary model fails (rate limit, API error, timeout), the client
    automatically tries each fallback model in order before raising.

    Args:
        model:           litellm model string (e.g. "gemini/gemini-2.0-flash")
        fallback_models: Ordered list of fallback model strings to try on failure
        temperature:     Sampling temperature (0.0 = deterministic)
        max_tokens:      Max tokens in response
        max_retries:     litellm-level retries for transient errors (per model attempt)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        fallback_models: list[str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 2,
    ):
        self.model = model
        self.fallback_models = fallback_models or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        # Running totals for the lifetime of this client
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

    def _all_models(self) -> list[str]:
        """Return primary + fallback models in order."""
        return [self.model] + self.fallback_models

    async def _try_single_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[str],
    ) -> str:
        """Attempt a single model call. Raises on failure."""
        # For free-tier models: merge system into user prompt and skip response_format
        # as these features may not be supported by all providers
        is_free = ":free" in model
        call_messages = messages
        if is_free and len(messages) > 1 and messages[0]["role"] == "system":
            # Merge system message into user message
            system_text = messages[0]["content"]
            user_text = messages[1]["content"]
            call_messages = [{"role": "user", "content": f"{system_text}\n\n{user_text}"}]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": call_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_retries": self.max_retries,
        }
        if response_format == "json" and not is_free:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await litellm.acompletion(**kwargs)
        except litellm.RateLimitError as exc:
            raise LLMRateLimitError(
                f"Rate limit hit on {model}",
                context={"model": model, "error": str(exc)},
            ) from exc
        except litellm.Timeout as exc:
            raise LLMTimeoutError(
                f"Timeout calling {model}",
                context={"model": model, "error": str(exc)},
            ) from exc
        except litellm.APIError as exc:
            raise LLMError(
                f"API error from {model}",
                context={"model": model, "error": str(exc)},
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"Unexpected LLM error: {exc}",
                context={"model": model, "error": str(exc)},
            ) from exc

        if not response.choices:
            raise LLMResponseError(
                "LLM returned empty choices array",
                context={"model": model},
            )

        text = response.choices[0].message.content or ""
        text = text.strip()

        if not text:
            raise LLMResponseError(
                "LLM returned empty response",
                context={"model": model},
            )

        # Track token usage
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_calls += 1
            log.debug(
                "llm_response",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_len=len(text),
            )

        return text

    async def complete(
        self,
        prompt: str,
        system: str = "",
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a completion request with retry-on-rate-limit + model fallback.

        Strategy:
          1. Try each model in the chain (primary + fallbacks)
          2. On rate limit: retry the SAME model with exponential backoff (up to 3 times)
          3. On other errors: immediately try the next model
          4. Only raise if ALL models fail

        Args:
            prompt:          User message content
            system:          System message (optional)
            response_format: "json" to request JSON output, None for free text
            temperature:     Override instance default
            max_tokens:      Override instance default

        Returns:
            Raw response text from whichever model succeeded.

        Raises:
            LLMRateLimitError: All models hit rate limits
            LLMTimeoutError:   All models timed out
            LLMError:          All models failed
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        models = self._all_models()
        last_error: Exception | None = None

        for model_idx, model in enumerate(models):
            # Per-model retry loop with backoff for rate limits
            max_rate_limit_retries = 1
            for retry in range(max_rate_limit_retries + 1):
                try:
                    log.debug(
                        "llm_request",
                        model=model,
                        model_idx=model_idx + 1,
                        total_models=len(models),
                        retry=retry,
                        prompt_len=len(prompt),
                    )
                    text = await self._try_single_model(
                        model=model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                        response_format=response_format,
                    )
                    if model_idx > 0 or retry > 0:
                        log.info(
                            "llm_success_after_retry",
                            model=model,
                            model_idx=model_idx + 1,
                            retry=retry,
                        )
                    return text

                except LLMRateLimitError as exc:
                    last_error = exc
                    if retry < max_rate_limit_retries:
                        # Quick retry: 5s then move on
                        delay = 5
                        log.warning(
                            "llm_rate_limit_retry",
                            model=model,
                            retry=retry + 1,
                            max_retries=max_rate_limit_retries,
                            backoff_seconds=delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        log.warning(
                            "llm_rate_limit_exhausted",
                            model=model,
                            retries=max_rate_limit_retries,
                        )
                        break  # Move to next model

                except (LLMTimeoutError, LLMError) as exc:
                    last_error = exc
                    log.warning(
                        "llm_model_error",
                        model=model,
                        error=str(exc)[:200],
                    )
                    break  # Non-rate-limit error: skip to next model

            # Brief pause before trying next model
            if model_idx < len(models) - 1:
                await asyncio.sleep(1)

        log.error(
            "llm_all_models_failed",
            models_tried=models,
            final_error=str(last_error)[:200] if last_error else "unknown",
        )

        # All models failed — re-raise the last error
        raise last_error  # type: ignore[misc]

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Send a completion request and parse the response as JSON.

        Falls back to regex JSON extraction if the model wraps JSON in markdown.

        Returns:
            Parsed dict.

        Raises:
            LLMResponseError: If response is not valid JSON.
        """
        text = await self.complete(
            prompt=prompt,
            system=system,
            response_format="json",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from markdown code blocks (greedy to capture nested objects)
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback 2: extract the outermost JSON object from the raw text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        log.warning(
            "llm_json_parse_failed",
            model=self.model,
            response_length=len(text),
            response_preview=text[:300],
        )
        raise LLMResponseError(
            "LLM response is not valid JSON",
            context={"model": self.model, "raw_response": text[:500]},
        )

    @property
    def usage_summary(self) -> dict[str, int]:
        return {
            "total_calls": self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }


def build_fallback_chain(primary_model: str) -> list[str]:
    """
    Build an ordered fallback model list based on available API keys.

    Inspects environment variables to determine which providers are configured,
    then returns models from those providers (excluding the primary).

    Fallback priority:
      1. Gemini (generous free tier)
      2. Groq (fast inference, free tier)
      3. OpenRouter (multi-provider gateway)
      4. OpenAI
      5. Anthropic
      6. xAI Grok

    Returns:
        List of fallback model strings (may be empty if no other keys are set).
    """
    import os

    fallbacks: list[str] = []

    if os.environ.get("GEMINI_API_KEY"):
        fallbacks.append("gemini/gemini-2.5-flash")

    if os.environ.get("GROQ_API_KEY"):
        fallbacks.append("groq/llama-3.1-8b-instant")

    if os.environ.get("OPENROUTER_API_KEY"):
        fallbacks.append("openrouter/google/gemma-3-4b-it:free")

    if os.environ.get("OPENAI_API_KEY"):
        fallbacks.append("openai/gpt-4o-mini")

    if os.environ.get("ANTHROPIC_API_KEY"):
        fallbacks.append("anthropic/claude-3-haiku-20240307")

    if os.environ.get("XAI_API_KEY"):
        fallbacks.append("xai/grok-3-mini-fast")

    # Remove primary model from fallback list (no point retrying the same model)
    fallbacks = [m for m in fallbacks if m != primary_model]

    return fallbacks


def auto_select_model() -> str:
    """
    Auto-select the best available primary model based on which API keys are set.
    Called when LLM_SUMMARY_MODEL / LLM_RANK_MODEL are not explicitly configured.
    """
    import os

    # Priority order: Gemini (best free tier) > Groq > OpenRouter > OpenAI > Anthropic > xAI
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-flash"
    if os.environ.get("GROQ_API_KEY"):
        return "groq/llama-3.1-8b-instant"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter/google/gemma-3-4b-it:free"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-3-haiku-20240307"
    if os.environ.get("XAI_API_KEY"):
        return "xai/grok-3-mini-fast"

    log.warning("no_api_keys_configured", msg="No LLM API keys found. Configure one via Settings.")
    return "gemini/gemini-2.5-flash"  # Default — will fail without a key
