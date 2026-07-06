"""Map raw upstream failures (OpenAI, LightRAG) to user-safe HTTP responses.

Previously any exception inside /query became a 500 whose detail leaked the
raw error ("RetryError[<Future ... APIConnectionError>]") straight into the
chat UI. Classify instead: rate limits -> 429, connectivity/auth -> 503,
everything else -> 500, all with messages fit for end users. The raw error
is logged server-side, never returned.
"""

from __future__ import annotations

RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "quota",
    "insufficient",
)
UNAVAILABLE_MARKERS = (
    "connection",
    "timeout",
    "timed out",
    "unavailable",
    "retryerror",
    "api key",
    "apikey",
    "authentication",
    "unauthorized",
    "401",
    "temporarily",
)

RATE_LIMIT_MESSAGE = "The AI service is receiving too many requests right now. Please try again in a minute."
UNAVAILABLE_MESSAGE = (
    "The AI service is temporarily unavailable. Please try again shortly."
)
GENERIC_MESSAGE = (
    "Something went wrong while answering this question. Please try again."
)


def classify_upstream_error(error: BaseException) -> tuple[int, str]:
    """Return (http_status, user_safe_message) for an upstream failure."""
    text = f"{type(error).__name__} {error}".lower()
    if any(marker in text for marker in RATE_LIMIT_MARKERS):
        return 429, RATE_LIMIT_MESSAGE
    if any(marker in text for marker in UNAVAILABLE_MARKERS):
        return 503, UNAVAILABLE_MESSAGE
    return 500, GENERIC_MESSAGE
