"""
LLM client wrapper for FixFirst AI.

Provider-agnostic: routes to Anthropic, OpenAI, or Groq based on
settings.llm_provider, so the same silver-labeling and inference-fallback
code works regardless of which provider's key is configured in .env.

Two things this module guarantees that a naive per-thread retry loop does
NOT:

1. RATE LIMITING IS GLOBAL, NOT PER-WORKER. A single shared RateLimiter
   instance is used by every call_llm() invocation regardless of which
   thread calls it. This matters a lot under ThreadPoolExecutor concurrency:
   without it, N worker threads each retry independently against the same
   API key, and raising max_workers can make 429s WORSE, not better,
   because you're just adding more threads that will all get rate-limited
   at once. With a shared limiter, max_workers controls parallelism while
   LLM_MAX_REQUESTS_PER_MINUTE controls actual request rate — they're
   decoupled, which is what you want.

2. BACKOFF HONORS THE SERVER'S Retry-After HEADER WHEN PRESENT, instead of
   always waiting a fixed 60s. Groq/OpenAI/Anthropic all attach a
   Retry-After header (or an equivalent in the response body) to 429s
   telling you exactly how long to wait — ignoring it and guessing 60s
   flat means you're either waiting longer than necessary or, worse,
   retrying too early and eating another 429. Falls back to exponential
   backoff with jitter only when no such header is available.

This module is intentionally thin — a single call_llm() function returning
raw text — so parser.py (already tested independently) handles all output
validation. Network calls happen ONLY here, keeping everything else in the
labeling package unit-testable without hitting a real API.
"""

import random
import sys
import threading
import time
from typing import Optional

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0  # deterministic labeling, not creative generation
MAX_RETRIES = 7
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_CAP_SECONDS = 60.0


class RateLimiter:
    """
    Thread-safe minimum-interval limiter. Every acquire() call blocks (if
    needed) so that calls across ALL threads are spaced at least
    (60 / max_calls_per_minute) seconds apart, enforcing a hard ceiling on
    global request rate regardless of how many worker threads are calling
    it concurrently.

    Holding the lock for the duration of the sleep is deliberate: it's
    what makes this a GLOBAL limiter rather than a per-thread one — only
    one thread can be "in the gate" at a time, which is exactly what
    caps the aggregate rate.
    """

    def __init__(self, max_calls_per_minute: int):
        if max_calls_per_minute <= 0:
            raise ValueError("max_calls_per_minute must be positive")
        self._interval = 60.0 / max_calls_per_minute
        self._lock = threading.Lock()
        self._last_call_time = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_call_time + self._interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_call_time = now


_rate_limiter = RateLimiter(settings.llm_max_requests_per_minute)


def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.active_llm_api_key())
    response = client.messages.create(
        model=settings.llm_model_name,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.active_llm_api_key())
    response = client.chat.completions.create(
        model=settings.llm_model_name,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=settings.active_llm_api_key())
    response = client.chat.completions.create(
        model=settings.llm_model_name,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


_PROVIDER_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "groq": _call_groq,
}


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Detects a 429 across anthropic/openai/groq SDKs without importing all
    three at module load time. Each SDK exposes a status_code somewhere
    (either directly or on a nested .response), and all of them also
    surface "429" in the string representation as a fallback signal.
    """
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True
    return "429" in str(error) or "rate limit" in str(error).lower()


def _extract_retry_after_seconds(error: Exception) -> Optional[float]:
    """
    Reads a Retry-After header off the error's underlying HTTP response,
    if the SDK exposes one. Returns None if unavailable — caller falls
    back to exponential backoff in that case.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _compute_backoff_seconds(attempt: int, error: Exception) -> float:
    retry_after = _extract_retry_after_seconds(error)
    if retry_after is not None:
        return retry_after

    # Exponential backoff with jitter, capped, when the server didn't tell
    # us exactly how long to wait.
    exponential = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0, BACKOFF_BASE_SECONDS)
    return min(exponential + jitter, BACKOFF_CAP_SECONDS)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Sends (system_prompt, user_prompt) to the configured LLM_PROVIDER and
    returns the raw text response. Rate-limited globally via the shared
    RateLimiter (safe to call from any number of worker threads). Retries
    on 429s with server-provided Retry-After when available, exponential
    backoff otherwise. Raises FixFirstException on any non-retryable
    failure, or after exhausting MAX_RETRIES on a persistent 429.
    """
    provider = settings.llm_provider
    handler = _PROVIDER_DISPATCH.get(provider)

    if handler is None:
        raise FixFirstException(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            f"Supported: {list(_PROVIDER_DISPATCH.keys())}",
            sys,
        )

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limiter.acquire()
        try:
            return handler(system_prompt, user_prompt)
        except Exception as e:
            last_error = e

            if not _is_rate_limit_error(e):
                logging.error(f"call_llm: {provider} request failed (non-retryable): {e}")
                raise FixFirstException(e, sys)

            if attempt == MAX_RETRIES:
                break

            wait_seconds = _compute_backoff_seconds(attempt, e)
            logging.warning(
                f"call_llm: rate limited by {provider}, retrying in {wait_seconds:.1f}s "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(wait_seconds)

    raise FixFirstException(
        f"call_llm: exhausted {MAX_RETRIES} attempts against {provider}, still rate-limited. "
        f"Last error: {last_error}. Consider lowering LLM_MAX_REQUESTS_PER_MINUTE or max_workers.",
        sys,
    )