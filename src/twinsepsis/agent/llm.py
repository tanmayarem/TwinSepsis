"""Provider-replaceable LLM client for the TwinSepsis agent.

The client speaks the OpenAI-compatible ``/chat/completions`` protocol, which
Groq exposes.  It is deliberately thin and injectable:

* configuration comes only from environment variables;
* the HTTP client can be replaced (tests use ``httpx.MockTransport``);
* the agent depends on the small :class:`ChatClient` protocol, not on Groq.

No credentials are ever logged or embedded in error messages.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_PROVIDER = "TWINSEPSIS_LLM_PROVIDER"
ENV_MODEL = "TWINSEPSIS_LLM_MODEL"
ENV_BASE_URL = "TWINSEPSIS_LLM_BASE_URL"
ENV_API_KEY = "TWINSEPSIS_LLM_API_KEY"
ENV_TIMEOUT = "TWINSEPSIS_LLM_TIMEOUT_S"

SUPPORTED_PROVIDERS = ("groq", "openai", "openai_compatible")


class LLMError(Exception):
    """Base class for every LLM-layer failure."""


class LLMConfigurationError(LLMError):
    """Raised when environment configuration is missing or invalid."""


class LLMNetworkError(LLMError):
    """Raised for transport failures and non-2xx HTTP responses."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class LLMRateLimitError(LLMNetworkError):
    """Raised when the provider rejects a request because of rate limiting."""


class LLMResponseError(LLMError):
    """Raised when the provider returns a payload we cannot interpret."""


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header into non-negative seconds.

    The header may be a number of seconds or an HTTP date.  Invalid values are
    ignored rather than turned into a retry loop; the client never retries.
    """

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()

    if seconds < 0:
        return 0.0
    return round(seconds, 3)


@dataclass(frozen=True)
class LLMConfig:
    """Validated LLM configuration."""

    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise LLMConfigurationError("provider must not be empty")
        if not self.model.strip():
            raise LLMConfigurationError("model must not be empty")
        if not self.api_key.strip():
            raise LLMConfigurationError("api_key must not be empty")

        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LLMConfigurationError(
                f"{ENV_BASE_URL} must be an absolute http(s) URL, got {self.base_url!r}"
            )
        if self.timeout_s <= 0:
            raise LLMConfigurationError("timeout_s must be > 0")

    @property
    def endpoint(self) -> str:
        """The chat-completions endpoint for an OpenAI-compatible base URL."""

        return self.base_url.rstrip("/") + "/chat/completions"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "LLMConfig":
        """Build configuration from environment variables.

        When ``env`` is omitted, ``.env`` is loaded first (without overriding
        already-exported variables) and then ``os.environ`` is read.
        """

        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(PROJECT_ROOT / ".env", override=False)
            except ImportError:  # pragma: no cover - python-dotenv is pinned
                pass
            env = os.environ

        def read(name: str) -> str:
            value = env.get(name, "")
            if value is None:
                return ""
            return str(value).strip()

        missing = [
            name
            for name in (ENV_PROVIDER, ENV_MODEL, ENV_BASE_URL, ENV_API_KEY)
            if not read(name)
        ]
        if missing:
            raise LLMConfigurationError(
                "Missing LLM configuration: "
                + ", ".join(missing)
                + ". Copy the names from .env.example into .env and set them there."
            )

        raw_timeout = read(ENV_TIMEOUT) or "60"
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise LLMConfigurationError(
                f"{ENV_TIMEOUT} must be a number, got {raw_timeout!r}"
            ) from exc

        return cls(
            provider=read(ENV_PROVIDER).lower(),
            model=read(ENV_MODEL),
            base_url=read(ENV_BASE_URL),
            api_key=read(ENV_API_KEY),
            timeout_s=timeout,
        )


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantTurn:
    """A normalised assistant response."""

    content: str | None
    tool_calls: list[ToolCallRequest]


class ChatClient(Protocol):
    """Minimal interface the agent depends on."""

    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AssistantTurn: ...


class OpenAICompatibleClient:
    """Groq/openai-compatible chat-completions client."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=config.timeout_s)

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- request -------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
            "tool_choice": "auto",
        }
        if tools:
            payload["tools"] = list(tools)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = self._client.post(
                self.config.endpoint,
                json=payload,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise LLMNetworkError(
                f"LLM request to {self.provider!r} timed out after "
                f"{self.config.timeout_s:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMNetworkError(
                f"LLM request to {self.provider!r} failed: {type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            retry_after_s = _parse_retry_after(response.headers.get("retry-after"))
            if response.status_code == 429:
                if retry_after_s is None:
                    message = (
                        "The LLM provider rate limit was reached (HTTP 429). "
                        "Please try again later."
                    )
                else:
                    message = (
                        "The LLM provider rate limit was reached (HTTP 429). "
                        f"Please retry after {retry_after_s:g} seconds."
                    )
                raise LLMRateLimitError(
                    message,
                    status_code=response.status_code,
                    retry_after_s=retry_after_s,
                )

            if response.status_code in {401, 403}:
                message = (
                    f"LLM authentication failed (HTTP {response.status_code}); "
                    "check the API key in the environment."
                )
            elif response.status_code >= 500:
                message = (
                    "The LLM provider is temporarily unavailable "
                    f"(HTTP {response.status_code}). Please try again later."
                )
            else:
                message = (
                    "The LLM provider rejected the request "
                    f"(HTTP {response.status_code})."
                )
            raise LLMNetworkError(
                message,
                status_code=response.status_code,
                retry_after_s=retry_after_s,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "LLM endpoint returned a response that is not valid JSON"
            ) from exc

        return self._parse_response(data)

    # -- parsing -------------------------------------------------------

    @staticmethod
    def _parse_response(data: Any) -> AssistantTurn:
        if not isinstance(data, Mapping):
            raise LLMResponseError("LLM response must be a JSON object")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("LLM response has no choices")

        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMResponseError("LLM choice must be a JSON object")

        message = first.get("message")
        if not isinstance(message, Mapping):
            raise LLMResponseError("LLM choice has no message object")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMResponseError("LLM message content must be a string or null")

        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise LLMResponseError("LLM message tool_calls must be a list")

        calls: list[ToolCallRequest] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, Mapping):
                raise LLMResponseError(f"tool_calls[{index}] must be an object")

            call_id = raw.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise LLMResponseError(f"tool_calls[{index}] has no id")

            function = raw.get("function")
            if not isinstance(function, Mapping):
                raise LLMResponseError(f"tool_calls[{index}] has no function object")

            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise LLMResponseError(f"tool_calls[{index}] has no function name")

            arguments = function.get("arguments")
            # The OpenAI schema uses a JSON string.  A few compatible servers
            # return an object instead; normalise that rather than losing it.
            if isinstance(arguments, Mapping):
                arguments = json.dumps(arguments)
            if not isinstance(arguments, str):
                raise LLMResponseError(
                    f"tool_calls[{index}] function.arguments must be a JSON string"
                )

            calls.append(
                ToolCallRequest(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )

        return AssistantTurn(content=content, tool_calls=calls)


def create_client(
    config: LLMConfig,
    *,
    client: httpx.Client | None = None,
) -> OpenAICompatibleClient:
    """Create a client for a supported OpenAI-compatible provider."""

    if config.provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            f"Unsupported provider {config.provider!r}. Supported values: "
            + ", ".join(SUPPORTED_PROVIDERS)
        )
    return OpenAICompatibleClient(config, client=client)


def load_client_from_env(
    *,
    client: httpx.Client | None = None,
) -> OpenAICompatibleClient:
    """Load configuration from the environment and build a client."""

    return create_client(LLMConfig.from_env(), client=client)


__all__ = [
    "LLMError",
    "LLMConfigurationError",
    "LLMNetworkError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMConfig",
    "ToolCallRequest",
    "AssistantTurn",
    "ChatClient",
    "OpenAICompatibleClient",
    "create_client",
    "load_client_from_env",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "ENV_TIMEOUT",
    "SUPPORTED_PROVIDERS",
]
