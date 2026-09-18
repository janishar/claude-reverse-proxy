#!/usr/bin/env python3
"""Configuration, routing and request-handling tests for the Python proxy."""

import json
import os
import unittest
from urllib.error import HTTPError

import main
from test_claude_route import FakeUpstream, ProxyUnderTest, make_config

CONFIG_ENV_KEYS = [
    "SERVER_PORT", "TARGET_BASE_URL", "TARGET_MODELS_PATH", "TARGET_CHAT_PATH",
    "TARGET_RESPONSES_PATH", "TARGET_EMBEDDINGS_PATH", "TARGET_MODERATIONS_PATH",
    "TARGET_COMPLETIONS_PATH", "TARGET_MESSAGES_PATH", "PROXY_MODEL_NAME",
    "TARGET_MODEL_NAME", "PROXY_MODEL_ID", "PROXY_MODEL_OBJECT", "PROXY_MODEL_CREATED",
    "PROXY_MODEL_OWNED_BY", "PROVIDER_API_KEY", "PROXY_API_KEY",
    "MESSAGES_UPSTREAM_MODE", "ANTHROPIC_MODEL_MARKERS", "MAX_BODY_MB",
]


class EnvIsolated(unittest.TestCase):
    """Starts every test from documented defaults, whatever the shell exports."""

    def setUp(self):
        self._env_backup = dict(os.environ)
        for key in CONFIG_ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)


class TestConfig(EnvIsolated):
    def test_defaults(self):
        config = main.Config()
        self.assertEqual(config.server_port, 8080)
        self.assertEqual(config.target_base_url, "http://localhost:8000")
        self.assertEqual(config.target_models_path, "/v1/models")
        self.assertEqual(config.target_chat_path, "/v1/chat/completions")
        self.assertEqual(config.target_responses_path, "/v1/responses")
        self.assertEqual(config.target_embeddings_path, "/v1/embeddings")
        self.assertEqual(config.target_moderations_path, "/v1/moderations")
        self.assertEqual(config.target_completions_path, "/v1/completions")
        self.assertEqual(config.target_messages_path, "/v1/messages")
        self.assertEqual(config.proxy_model_name, "claude-opus-5")
        self.assertEqual(config.target_model_name, "nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(config.proxy_model_id, "claude-opus-5")
        self.assertEqual(config.proxy_model_object, "model")
        self.assertEqual(config.proxy_model_owned_by, "cloud-proxy")
        self.assertEqual(config.provider_api_key, "")
        self.assertEqual(config.proxy_api_key, "")
        self.assertEqual(config.messages_upstream_mode, main.MODE_AUTO)
        self.assertEqual(config.max_body_bytes, 50 * 1024 * 1024)
        self.assertEqual(config.anthropic_model_markers, main.DEFAULT_ANTHROPIC_MODEL_MARKERS)
        self.assertTrue(config.proxy_model_created > 0)

    def test_from_env(self):
        os.environ.update({
            "SERVER_PORT": "9000",
            "TARGET_BASE_URL": "https://custom.api.com",
            "TARGET_CHAT_PATH": "/custom/chat",
            "TARGET_MESSAGES_PATH": "/custom/messages",
            "PROXY_MODEL_NAME": "custom-proxy",
            "TARGET_MODEL_NAME": "custom-target",
            "PROXY_MODEL_ID": "custom-id",
            "PROXY_MODEL_OBJECT": "custom-object",
            "PROXY_MODEL_CREATED": "9999999999",
            "PROXY_MODEL_OWNED_BY": "custom-owner",
            "PROVIDER_API_KEY": "upstream",
            "PROXY_API_KEY": "downstream",
            "MESSAGES_UPSTREAM_MODE": "PASSTHROUGH",
            "MAX_BODY_MB": "7",
        })
        config = main.Config()
        self.assertEqual(config.server_port, 9000)
        self.assertEqual(config.target_base_url, "https://custom.api.com")
        self.assertEqual(config.target_chat_path, "/custom/chat")
        self.assertEqual(config.target_messages_path, "/custom/messages")
        self.assertEqual(config.proxy_model_name, "custom-proxy")
        self.assertEqual(config.target_model_name, "custom-target")
        self.assertEqual(config.proxy_model_id, "custom-id")
        self.assertEqual(config.proxy_model_object, "custom-object")
        self.assertEqual(config.proxy_model_created, 9999999999)
        self.assertEqual(config.proxy_model_owned_by, "custom-owner")
        self.assertEqual(config.provider_api_key, "upstream")
        self.assertEqual(config.proxy_api_key, "downstream")
        self.assertEqual(config.max_body_bytes, 7 * 1024 * 1024)
        # The mode is lowercased, so the env var is not case sensitive.
        self.assertEqual(config.messages_upstream_mode, main.MODE_PASSTHROUGH)

    def test_trims_trailing_slash_from_base_url(self):
        os.environ["TARGET_BASE_URL"] = "https://api.example.com///"
        self.assertEqual(main.Config().target_base_url, "https://api.example.com")

    def test_custom_markers_replace_the_defaults(self):
        os.environ["ANTHROPIC_MODEL_MARKERS"] = " NEMOTRON , mistral ,, "
        config = main.Config()
        self.assertEqual(config.anthropic_model_markers, ["nemotron", "mistral"])
        self.assertTrue(main.is_anthropic_model(config, "nvidia/nemotron-3"))
        self.assertFalse(main.is_anthropic_model(config, "claude-opus-4-5"))

    def test_get_env_int(self):
        os.environ["PROXY_TEST_INT"] = "42"
        self.assertEqual(main.get_env_int("PROXY_TEST_INT", 7), 42)
        os.environ["PROXY_TEST_INT"] = "not-a-number"
        self.assertEqual(main.get_env_int("PROXY_TEST_INT", 7), 7)
        os.environ.pop("PROXY_TEST_INT")
        self.assertEqual(main.get_env_int("PROXY_TEST_INT", 7), 7)

    def test_parse_model_markers(self):
        self.assertEqual(main.parse_model_markers("opus,sonnet"), ["opus", "sonnet"])
        self.assertEqual(main.parse_model_markers(" Opus , SONNET "), ["opus", "sonnet"])
        self.assertEqual(main.parse_model_markers("opus,,sonnet,"), ["opus", "sonnet"])
        self.assertEqual(main.parse_model_markers(""), [])
        self.assertEqual(main.parse_model_markers(None), [])


class TestHelpers(unittest.TestCase):
    def test_extract_bearer_token(self):
        cases = {
            "Bearer abc": "abc",
            "bearer abc": "abc",
            "BEARER abc": "abc",
            "Bearer a b c": "a b c",
            "Basic abc": "",
            "abc": "",
            "": "",
            "Bearer": "",
        }
        for header, want in cases.items():
            self.assertEqual(main.extract_bearer_token(header), want, header)

    def test_to_int(self):
        self.assertEqual(main.to_int(7), 7)
        self.assertEqual(main.to_int(7.9), 7)
        self.assertEqual(main.to_int("7"), 7)
        self.assertEqual(main.to_int("x"), 0)
        self.assertEqual(main.to_int(None), 0)

    def test_content_to_text(self):
        self.assertEqual(main.content_to_text("hello"), "hello")
        self.assertEqual(
            main.content_to_text(
                [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
            ),
            "a\nb",
        )
        self.assertEqual(
            main.content_to_text([{"type": "image"}, {"type": "text", "text": "only me"}]),
            "only me",
        )
        self.assertEqual(main.content_to_text([]), "")
        self.assertEqual(main.content_to_text(None), "")
        self.assertEqual(main.content_to_text(42), "")

    def test_anthropic_error_type(self):
        cases = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            413: "request_too_large",
            429: "rate_limit_error",
            500: "api_error",
            502: "api_error",
            529: "overloaded_error",
        }
        for status, want in cases.items():
            self.assertEqual(main.anthropic_error_type(status), want, status)

    def test_openai_error_to_anthropic(self):
        cases = [
            (400, b'{"error":{"message":"bad model"}}', "bad model"),
            (500, b'{"message":"boom"}', "boom"),
            (502, b"upstream exploded", "upstream exploded"),
            (500, b"", "Upstream returned 500"),
        ]
        for status, body, want in cases:
            out = json.loads(main.openai_error_to_anthropic(status, body))
            self.assertEqual(out["type"], "error")
            self.assertEqual(out["error"]["message"], want)
            self.assertEqual(out["error"]["type"], main.anthropic_error_type(status))

    def test_replace_model_in_response_passes_non_json_through(self):
        config = make_config()
        for raw in (b"not json", b"", b"[1,2,3]"):
            self.assertEqual(main.replace_model_in_response(config, raw), raw)

    def test_is_event_stream(self):
        self.assertTrue(main.is_event_stream("text/event-stream"))
        self.assertTrue(main.is_event_stream("text/event-stream; charset=utf-8"))
        self.assertTrue(main.is_event_stream("TEXT/EVENT-STREAM"))
        self.assertFalse(main.is_event_stream("application/json"))
        self.assertFalse(main.is_event_stream(None))

    def test_drain_stops_at_eof(self):
        import io

        stream = io.BytesIO(b"abc")
        main.drain(stream, 1000)  # must not hang or raise when the peer stops early
        self.assertEqual(stream.read(), b"")


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)

    def start_proxy(self, **overrides):
        settings = {"TARGET_BASE_URL": self.upstream.url, "PROVIDER_API_KEY": "upstream-key"}
        settings.update(overrides)
        config = make_config(**settings)
        proxy = ProxyUnderTest(config)
        self.addCleanup(proxy.close)
        return proxy

    def test_models_endpoint_is_served_locally(self):
        proxy = self.start_proxy(PROXY_MODEL_ID="test-id", PROXY_MODEL_OWNED_BY="test-owner")

        from urllib.request import urlopen

        with urlopen(proxy.url + "/v1/models") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get("Content-Type"), "application/json")
            data = json.loads(response.read().decode())

        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], "test-id")
        self.assertEqual(data["data"][0]["owned_by"], "test-owner")
        self.assertIsNone(self.upstream.received_path, "upstream must not be called")

    def test_models_endpoint_forwards_non_get(self):
        self.upstream.respond("/v1/models", json.dumps({"object": "list", "data": []}))
        proxy = self.start_proxy()

        with proxy.post("/v1/models", {"model": "claude-opus-5"}) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.upstream.received_path, "/v1/models")

    def test_each_endpoint_reaches_its_target_path(self):
        proxy = self.start_proxy(
            TARGET_CHAT_PATH="/custom/chat",
            TARGET_RESPONSES_PATH="/custom/responses",
            TARGET_EMBEDDINGS_PATH="/custom/embeddings",
            TARGET_MODERATIONS_PATH="/custom/moderations",
            TARGET_COMPLETIONS_PATH="/custom/completions",
        )
        cases = {
            "/v1/chat/completions": "/custom/chat",
            "/v1/responses": "/custom/responses",
            "/v1/embeddings": "/custom/embeddings",
            "/v1/moderations": "/custom/moderations",
            "/v1/completions": "/custom/completions",
        }
        for inbound, target in cases.items():
            self.upstream.respond(target, "{}")
            with proxy.post(inbound, {"model": "claude-opus-5"}):
                pass
            self.assertEqual(self.upstream.received_path, target, inbound)

    def test_wildcard_keeps_the_inbound_path_and_query(self):
        self.upstream.respond("/v1/files", json.dumps({"object": "list"}))
        proxy = self.start_proxy()

        from urllib.request import urlopen

        with urlopen(proxy.url + "/v1/files?limit=2") as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.upstream.received_path, "/v1/files?limit=2")

    def test_unknown_path_is_404(self):
        proxy = self.start_proxy()
        with self.assertRaises(HTTPError) as cm:
            proxy.post("/not-an-api-path", {})
        self.assertEqual(cm.exception.code, 404)

    def test_invalid_json_is_rejected_before_the_upstream(self):
        proxy = self.start_proxy()

        from urllib.request import Request, urlopen

        request = Request(
            proxy.url + "/v1/chat/completions",
            data=b"invalid json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as cm:
            urlopen(request)

        self.assertEqual(cm.exception.code, 400)
        body = json.loads(cm.exception.read().decode())
        self.assertEqual(body["error"]["message"], "Invalid JSON")
        self.assertIsNone(self.upstream.received_path)

    def test_oversized_body_is_rejected_with_413(self):
        proxy = self.start_proxy(MAX_BODY_MB="1")

        with self.assertRaises(HTTPError) as cm:
            proxy.post("/v1/chat/completions", {"model": "claude-opus-5", "pad": "x" * (2 * 1024 * 1024)})

        self.assertEqual(cm.exception.code, 413)
        body = json.loads(cm.exception.read().decode())
        self.assertEqual(body["error"]["message"], "Request body too large")
        self.assertIsNone(self.upstream.received_path)

    def test_body_under_the_limit_is_forwarded(self):
        self.upstream.respond("/v1/chat/completions", "{}")
        proxy = self.start_proxy(MAX_BODY_MB="1")

        with proxy.post("/v1/chat/completions", {"model": "claude-opus-5", "pad": "x" * 1024}):
            pass
        self.assertEqual(self.upstream.received_path, "/v1/chat/completions")

    def test_unreachable_upstream_returns_502(self):
        proxy = self.start_proxy(TARGET_BASE_URL="http://127.0.0.1:1")

        with self.assertRaises(HTTPError) as cm:
            proxy.post("/v1/chat/completions", {"model": "claude-opus-5"})
        self.assertEqual(cm.exception.code, 502)
        self.assertIn("error", json.loads(cm.exception.read().decode()))

        with self.assertRaises(HTTPError) as cm:
            proxy.post("/v1/messages", {"model": "claude-opus-5"})
        self.assertEqual(cm.exception.code, 502)
        # Anthropic clients need the error envelope, not the OpenAI one.
        self.assertEqual(json.loads(cm.exception.read().decode())["type"], "error")

    def test_upstream_error_status_is_preserved_on_openai_routes(self):
        self.upstream.respond(
            "/v1/chat/completions", json.dumps({"error": {"message": "nope"}}), status=400
        )
        proxy = self.start_proxy()

        with self.assertRaises(HTTPError) as cm:
            proxy.post("/v1/chat/completions", {"model": "claude-opus-5"})
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("nope", cm.exception.read().decode())


class TestProxyAuthentication(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)
        self.upstream.respond("/v1/chat/completions", "{}")
        config = make_config(
            TARGET_BASE_URL=self.upstream.url,
            PROVIDER_API_KEY="upstream-key",
            PROXY_API_KEY="secret",
        )
        self.proxy = ProxyUnderTest(config)
        self.addCleanup(self.proxy.close)

    def test_accepts_a_bearer_token(self):
        with self.proxy.post(
            "/v1/chat/completions", {"model": "claude-opus-5"}, {"Authorization": "Bearer secret"}
        ) as response:
            self.assertEqual(response.status, 200)

    def test_accepts_an_anthropic_api_key_header(self):
        with self.proxy.post(
            "/v1/chat/completions", {"model": "claude-opus-5"}, {"x-api-key": "secret"}
        ) as response:
            self.assertEqual(response.status, 200)

    def test_rejects_a_wrong_or_missing_key(self):
        for headers in ({"Authorization": "Bearer wrong"}, {"x-api-key": "wrong"}, {}):
            with self.assertRaises(HTTPError) as cm:
                self.proxy.post("/v1/chat/completions", {"model": "claude-opus-5"}, headers)
            self.assertEqual(cm.exception.code, 401, headers)
            body = json.loads(cm.exception.read().decode())
            self.assertEqual(body["error"]["message"], "Unauthorized")

    def test_open_proxy_when_no_key_is_configured(self):
        config = make_config(TARGET_BASE_URL=self.upstream.url, PROVIDER_API_KEY="upstream-key")
        open_proxy = ProxyUnderTest(config)
        self.addCleanup(open_proxy.close)

        with open_proxy.post("/v1/chat/completions", {"model": "claude-opus-5"}) as response:
            self.assertEqual(response.status, 200)


class TestUpstreamCredentials(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)
        self.upstream.respond("/v1/chat/completions", "{}")

    def test_no_provider_key_means_no_authorization_header(self):
        # A keyless upstream (Ollama, vLLM, llama.cpp) is a supported setup.
        config = make_config(TARGET_BASE_URL=self.upstream.url)
        proxy = ProxyUnderTest(config)
        self.addCleanup(proxy.close)

        with proxy.post("/v1/chat/completions", {"model": "claude-opus-5"}):
            pass
        self.assertIsNone(self.upstream.received_headers.get("Authorization"))

    def test_provider_key_replaces_the_caller_credential(self):
        config = make_config(TARGET_BASE_URL=self.upstream.url, PROVIDER_API_KEY="upstream-key")
        proxy = ProxyUnderTest(config)
        self.addCleanup(proxy.close)

        with proxy.post(
            "/v1/chat/completions",
            {"model": "claude-opus-5"},
            {"Authorization": "Bearer client-key", "x-api-key": "client-key"},
        ):
            pass
        self.assertEqual(self.upstream.received_headers.get("Authorization"), "Bearer upstream-key")
        self.assertIsNone(self.upstream.received_headers.get("x-api-key"))


if __name__ == "__main__":
    unittest.main()
