from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.llm import (  # noqa: E402
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    ENV_TIMEOUT,
    LLMConfig,
    LLMConfigurationError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseError,
    OpenAICompatibleClient,
    create_client,
)

VALID_ENV = {
    ENV_PROVIDER: "groq",
    ENV_MODEL: "test-model",
    ENV_BASE_URL: "https://api.groq.com/openai/v1",
    ENV_API_KEY: "test-key-not-a-real-secret",
}


def make_client(handler) -> tuple[OpenAICompatibleClient, httpx.Client]:
    config = LLMConfig.from_env(VALID_ENV)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleClient(config, client=http_client), http_client


class ConfigTests(unittest.TestCase):
    def test_from_env_reads_all_four_variables(self) -> None:
        config = LLMConfig.from_env(VALID_ENV)
        self.assertEqual(config.provider, "groq")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(config.api_key, "test-key-not-a-real-secret")
        self.assertEqual(config.timeout_s, 60.0)
        self.assertEqual(
            config.endpoint,
            "https://api.groq.com/openai/v1/chat/completions",
        )

    def test_missing_configuration_is_named_not_leaked(self) -> None:
        for name in (ENV_PROVIDER, ENV_MODEL, ENV_BASE_URL, ENV_API_KEY):
            env = dict(VALID_ENV)
            env.pop(name)
            with self.subTest(missing=name):
                with self.assertRaises(LLMConfigurationError) as context:
                    LLMConfig.from_env(env)
                self.assertIn(name, str(context.exception))
                self.assertNotIn("test-key-not-a-real-secret", str(context.exception))

    def test_invalid_base_url_is_rejected(self) -> None:
        env = dict(VALID_ENV)
        env[ENV_BASE_URL] = "api.groq.com/openai/v1"
        with self.assertRaises(LLMConfigurationError):
            LLMConfig.from_env(env)

    def test_invalid_timeout_is_rejected(self) -> None:
        env = dict(VALID_ENV)
        env[ENV_TIMEOUT] = "soon"
        with self.assertRaises(LLMConfigurationError):
            LLMConfig.from_env(env)

    def test_api_key_is_not_in_repr(self) -> None:
        config = LLMConfig.from_env(VALID_ENV)
        self.assertNotIn("test-key-not-a-real-secret", repr(config))

    def test_unsupported_provider_is_rejected(self) -> None:
        config = LLMConfig.from_env({**VALID_ENV, ENV_PROVIDER: "unknown-provider"})
        with self.assertRaises(LLMConfigurationError):
            create_client(config)

    def test_provider_is_replaceable(self) -> None:
        config = LLMConfig.from_env({**VALID_ENV, ENV_PROVIDER: "openai_compatible"})
        client = create_client(config, client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": []})
        )))
        self.assertEqual(client.provider, "openai_compatible")
        client.close()


class ClientTests(unittest.TestCase):
    def test_parses_tool_call_and_sends_expected_payload(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "calling a tool",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "predict_risk",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        client, http_client = make_client(handler)
        try:
            turn = client.complete(
                [{"role": "user", "content": "hello"}],
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "predict_risk",
                            "description": "Risk",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        finally:
            http_client.close()

        self.assertEqual(captured["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer test-key-not-a-real-secret")
        self.assertEqual(captured["body"]["model"], "test-model")
        self.assertEqual(captured["body"]["temperature"], 0)
        self.assertEqual(captured["body"]["tool_choice"], "auto")
        self.assertEqual(captured["body"]["tools"][0]["function"]["name"], "predict_risk")

        self.assertEqual(turn.content, "calling a tool")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "call-1")
        self.assertEqual(turn.tool_calls[0].name, "predict_risk")
        self.assertEqual(turn.tool_calls[0].arguments, "{}")

    def test_plain_content_without_tools(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "done"}}]}
            )

        client, http_client = make_client(handler)
        try:
            turn = client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()
        self.assertEqual(turn.content, "done")
        self.assertEqual(turn.tool_calls, [])

    def test_http_error_becomes_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="upstream exploded")

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMNetworkError) as context:
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()
        self.assertIn("500", str(context.exception))
        self.assertNotIn("upstream exploded", str(context.exception))
        self.assertNotIn("test-key-not-a-real-secret", str(context.exception))

    def test_rate_limit_is_sanitised_and_retry_after_is_parsed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"Retry-After": "12.5"},
                text='{"error":{"message":"raw provider detail with test-key-not-a-real-secret"}}',
            )

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMRateLimitError) as context:
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()

        error = context.exception
        self.assertIsInstance(error, LLMNetworkError)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_after_s, 12.5)
        self.assertIn("rate limit", str(error))
        self.assertIn("12.5 seconds", str(error))
        self.assertNotIn("raw provider detail", str(error))
        self.assertNotIn("test-key-not-a-real-secret", str(error))

    def test_rate_limit_without_retry_after_still_has_a_safe_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="quota exceeded for this key")

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMRateLimitError) as context:
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()

        error = context.exception
        self.assertIsNone(error.retry_after_s)
        self.assertIn("HTTP 429", str(error))
        self.assertNotIn("quota exceeded", str(error))

    def test_rate_limit_accepts_an_http_date_retry_after(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"},
            )

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMRateLimitError) as context:
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()

        # The date is in the past, so it must be clamped rather than negative.
        self.assertEqual(context.exception.retry_after_s, 0.0)

    def test_auth_error_is_clear(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="no")

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMNetworkError) as context:
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()
        self.assertIn("authentication", str(context.exception))

    def test_transport_failure_becomes_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("synthetic network failure")

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMNetworkError):
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()

    def test_non_json_response_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        client, http_client = make_client(handler)
        try:
            with self.assertRaises(LLMResponseError):
                client.complete([{"role": "user", "content": "hi"}], [])
        finally:
            http_client.close()

    def test_malformed_payloads_are_rejected(self) -> None:
        invalid_payloads = [
            [],
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": "not-an-object"}]},
            {"choices": [{"message": {"tool_calls": "nope"}}]},
            {"choices": [{"message": {"tool_calls": [{}]}}]},
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "x", "function": {"name": "t", "arguments": 3}}
                            ]
                        }
                    }
                ]
            },
        ]
        for index, payload in enumerate(invalid_payloads):
            with self.subTest(index=index):
                with self.assertRaises(LLMResponseError):
                    OpenAICompatibleClient._parse_response(payload)

    def test_empty_message_and_object_arguments_are_tolerated(self) -> None:
        empty = OpenAICompatibleClient._parse_response(
            {"choices": [{"message": {}}]}
        )
        self.assertIsNone(empty.content)
        self.assertEqual(empty.tool_calls, [])

        parsed = OpenAICompatibleClient._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "x", "function": {"name": "t", "arguments": {}}}
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual(parsed.tool_calls[0].name, "t")
        self.assertEqual(parsed.tool_calls[0].arguments, "{}")


if __name__ == "__main__":
    unittest.main()
