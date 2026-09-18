#!/usr/bin/env python3
"""Tests for the Anthropic <-> OpenAI relay and the model masking it wraps."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import main


def make_config(**overrides):
    env_backup = dict(os.environ)
    for key in (
        "PROXY_MODEL_NAME",
        "TARGET_MODEL_NAME",
        "TARGET_BASE_URL",
        "PROVIDER_API_KEY",
        "PROXY_API_KEY",
        "MESSAGES_UPSTREAM_MODE",
        "ANTHROPIC_MODEL_MARKERS",
    ):
        os.environ.pop(key, None)
    os.environ["PROXY_MODEL_NAME"] = "claude-opus-5"
    os.environ["TARGET_MODEL_NAME"] = "test-target"
    for key, value in overrides.items():
        os.environ[key] = value
    try:
        return main.Config()
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


class FakeUpstream:
    """Records the last request and replays a scripted response."""

    def __init__(self):
        self.received_path = None
        self.received_headers = None
        self.received_body = None
        self.routes = {}
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                self.handle_any()

            def do_POST(self):
                self.handle_any()

            def handle_any(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                upstream.received_path = self.path
                upstream.received_headers = dict(self.headers)
                try:
                    upstream.received_body = json.loads(raw.decode())
                except json.JSONDecodeError:
                    upstream.received_body = None

                route = upstream.routes.get(self.path.split("?", 1)[0])
                if route is None:
                    body = b'{"error":{"message":"no such route"}}'
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                status, content_type, payload = route
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def respond(self, path, payload, status=200, content_type="application/json"):
        self.routes[path] = (status, content_type, payload.encode("utf-8"))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ProxyUnderTest:
    """Runs the real proxy handler against a configured upstream."""

    def __init__(self, config):
        handler_config = config

        class Handler(main.ProxyHandler):
            config = handler_config

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def post(self, path, payload, headers=None):
        request = Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        return urlopen(request)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def parse_sse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if event_type and data is not None:
            data["__event"] = event_type
            events.append(data)
    return events


class TestModelDetection(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_detects_anthropic_model_names(self):
        for model in (
            "claude-opus-4-5",
            "CLAUDE-SONNET-4-20250514",
            "claude-haiku-3-5",
            "claude-haiki-1",
            "anthropic.claude-fable-1",
        ):
            self.assertTrue(main.is_anthropic_model(self.config, model), model)

        for model in ("gpt-4o", "nvidia/nemotron-3-super-120b-a12b", "claude", "", None):
            self.assertFalse(main.is_anthropic_model(self.config, model), model)

    def test_markers_are_configurable(self):
        config = make_config(ANTHROPIC_MODEL_MARKERS="nemotron, mistral")
        self.assertTrue(main.is_anthropic_model(config, "nvidia/nemotron-3"))
        self.assertFalse(main.is_anthropic_model(config, "claude-opus-4-5"))

    def test_routing_decision(self):
        self.assertTrue(main.should_route_claude_to_chat(self.config, "claude-opus-4-5"))
        self.assertTrue(main.should_route_claude_to_chat(self.config, "claude-opus-5"))
        self.assertFalse(main.should_route_claude_to_chat(self.config, "gpt-4o"))
        self.assertFalse(main.should_route_claude_to_chat(self.config, ""))

        # The proxy's own model name always relays, whatever it is called.
        custom = make_config(PROXY_MODEL_NAME="my-model")
        self.assertTrue(main.should_route_claude_to_chat(custom, "my-model"))

        translate = make_config(MESSAGES_UPSTREAM_MODE="translate")
        self.assertTrue(main.should_route_claude_to_chat(translate, "gpt-4o"))

        passthrough = make_config(MESSAGES_UPSTREAM_MODE="passthrough")
        self.assertFalse(main.should_route_claude_to_chat(passthrough, "claude-opus-4-5"))


class TestModelMasking(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_masks_proxy_and_anthropic_model_names(self):
        for model in ("claude-opus-5", "claude-sonnet-4-5", "CLAUDE-HAIKU-3"):
            masked = main.mask_request_model(self.config, {"model": model, "n": 1})
            self.assertEqual(masked["model"], "test-target")
            self.assertEqual(masked["n"], 1, "other fields must survive masking")

    def test_leaves_foreign_model_alone(self):
        for model in ("gpt-4o", "nvidia/nemotron-3", ""):
            body = {"model": model}
            self.assertEqual(main.mask_request_model(self.config, body)["model"], model)

    def test_masking_does_not_mutate_the_caller_s_dict(self):
        original = {"model": "claude-opus-5"}
        main.mask_request_model(self.config, original)
        self.assertEqual(original["model"], "claude-opus-5")

    def test_body_without_a_model_is_untouched(self):
        self.assertEqual(main.mask_request_model(self.config, {"input": "hi"}), {"input": "hi"})

    def test_replaces_model_in_response(self):
        raw = json.dumps(
            {
                "model": "test-target",
                "choices": [{"message": {"model": "test-target", "content": "hi"}}],
                "data": [{"model": "test-target"}],
            }
        ).encode()
        resp = json.loads(main.replace_model_in_response(self.config, raw))
        self.assertEqual(resp["model"], "claude-opus-5")
        self.assertEqual(resp["choices"][0]["message"]["model"], "claude-opus-5")
        self.assertEqual(resp["data"][0]["model"], "claude-opus-5")

    def test_serialised_json_is_compact(self):
        # Go and Node emit compact JSON; Python must match byte for byte so the
        # upstream sees the same request from every implementation.
        self.assertEqual(main.dumps({"a": 1, "b": [1, 2]}), '{"a":1,"b":[1,2]}')
        openai_req = main.anthropic_to_openai_request(
            self.config,
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "t1", "name": "f", "input": {"city": "Paris"}}
                        ],
                    }
                ]
            },
        )
        arguments = openai_req["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, '{"city":"Paris"}')

    def test_masks_model_inside_sse_frames_only(self):
        line = b'data: {"model":"test-target","choices":[]}\n'
        self.assertIn(b"claude-opus-5", main.mask_model_in_sse_line(self.config, line))
        for untouched in (b"data: [DONE]\n", b"\n", b"event: ping\n"):
            self.assertEqual(main.mask_model_in_sse_line(self.config, untouched), untouched)


class TestAnthropicToOpenAIRequest(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_translates_a_full_request(self):
        out = main.anthropic_to_openai_request(
            self.config,
            {
                "model": "claude-opus-4-5",
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 5,
                "stop_sequences": ["\n\n", "HALT"],
                "system": "You are a helpful assistant.",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Let me check that."},
                            {
                                "type": "tool_use",
                                "id": "toolu_01",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_01",
                                "content": "Sunny, 25C",
                            }
                        ],
                    },
                    {"role": "user", "content": "Thanks"},
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "input_schema": {"type": "object"},
                    }
                ],
                "tool_choice": {"type": "auto"},
            },
        )

        self.assertEqual(out["model"], "test-target")
        self.assertEqual(out["max_tokens"], 1024)
        self.assertEqual(out["temperature"], 0.7)
        self.assertEqual(out["top_p"], 0.9)
        self.assertEqual(out["stop"], ["\n\n", "HALT"])
        self.assertNotIn("top_k", out)

        msgs = out["messages"]
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[0], {"role": "system", "content": "You are a helpful assistant."})
        self.assertEqual(msgs[1], {"role": "user", "content": "Hello"})
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["content"], "Let me check that.")
        tool_call = msgs[2]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "toolu_01")
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(
            msgs[3],
            {"role": "tool", "tool_call_id": "toolu_01", "content": "Sunny, 25C"},
        )
        self.assertEqual(msgs[4], {"role": "user", "content": "Thanks"})

        self.assertEqual(out["tools"][0]["type"], "function")
        self.assertEqual(out["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(out["tools"][0]["function"]["parameters"], {"type": "object"})
        self.assertEqual(out["tool_choice"], "auto")

    def test_translates_tool_choice_variants(self):
        def choice(tc):
            return main.anthropic_to_openai_request(self.config, {"tool_choice": tc}).get("tool_choice")

        self.assertEqual(choice({"type": "auto"}), "auto")
        self.assertEqual(choice({"type": "any"}), "required")
        self.assertEqual(choice({"type": "none"}), "none")
        self.assertEqual(
            choice({"type": "tool", "name": "get_weather"}),
            {"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(choice("auto"), "auto")

    def test_translates_image_blocks(self):
        out = main.anthropic_to_openai_request(
            self.config,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ]
            },
        )
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/png;base64,AAAA")
        self.assertEqual(parts[1], {"type": "text", "text": "What is this?"})

    def test_system_blocks_are_flattened(self):
        out = main.anthropic_to_openai_request(
            self.config,
            {"system": [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "Be kind."}]},
        )
        self.assertEqual(out["messages"][0], {"role": "system", "content": "Be terse.\nBe kind."})


class TestOpenAIToAnthropicResponse(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_translates_text_response(self):
        out = main.openai_to_anthropic_response(
            self.config,
            {
                "id": "chatcmpl-abc123",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
        self.assertEqual(out["id"], "msg_abc123")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["model"], "claude-opus-5")
        self.assertEqual(out["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"], {"input_tokens": 10, "output_tokens": 5})

    def test_translates_tool_calls(self):
        out = main.openai_to_anthropic_response(
            self.config,
            {
                "id": "chatcmpl-tool",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Checking",
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        self.assertEqual(out["content"][0], {"type": "text", "text": "Checking"})
        self.assertEqual(
            out["content"][1],
            {"type": "tool_use", "id": "call_9", "name": "get_weather", "input": {"city": "Paris"}},
        )
        self.assertEqual(out["stop_reason"], "tool_use")

    def test_maps_finish_reasons(self):
        self.assertEqual(main.map_finish_reason("stop"), "end_turn")
        self.assertEqual(main.map_finish_reason("length"), "max_tokens")
        self.assertEqual(main.map_finish_reason("tool_calls"), "tool_use")
        self.assertEqual(main.map_finish_reason("content_filter"), "end_turn")
        self.assertEqual(main.map_finish_reason(None), "end_turn")

    def test_shapes_upstream_errors_as_anthropic_errors(self):
        out = json.loads(
            main.openai_error_to_anthropic(429, b'{"error":{"message":"slow down"}}')
        )
        self.assertEqual(out["type"], "error")
        self.assertEqual(out["error"]["type"], "rate_limit_error")
        self.assertEqual(out["error"]["message"], "slow down")


class TestStreamTranslation(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def translate(self, lines):
        events = []
        translator = main.AnthropicStreamTranslator(
            self.config, lambda event_type, data: events.append((event_type, data))
        )
        for line in lines:
            translator.feed_line(line)
            if translator.done:
                break
        translator.finish()
        return events

    def test_translates_text_stream(self):
        events = self.translate(
            [
                'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}',
                'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" there"}}]}',
                'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
                "data: [DONE]",
            ]
        )
        self.assertEqual(
            [e[0] for e in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        self.assertEqual(events[0][1]["message"]["id"], "msg_s1")
        self.assertEqual(events[0][1]["message"]["model"], "claude-opus-5")
        text = "".join(
            e[1]["delta"]["text"] for e in events if e[0] == "content_block_delta"
        )
        self.assertEqual(text, "Hi there")
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[-2][1]["usage"], {"input_tokens": 3, "output_tokens": 2})

    def test_translates_tool_call_stream(self):
        events = self.translate(
            [
                'data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"get_weather","arguments":""}}]}}]}',
                'data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ci"}}]}}]}',
                'data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\":\\"Paris\\"}"}}]}}]}',
                'data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
        types = [e[0] for e in events]
        # A tool-only stream still has to open with message_start.
        self.assertEqual(types[0], "message_start")
        self.assertEqual(
            types,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        block = events[1][1]["content_block"]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_9")
        self.assertEqual(block["name"], "get_weather")
        partial = "".join(
            e[1]["delta"]["partial_json"] for e in events if e[0] == "content_block_delta"
        )
        self.assertEqual(json.loads(partial), {"city": "Paris"})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")

    def test_empty_stream_still_produces_a_valid_message(self):
        events = self.translate(["data: [DONE]"])
        self.assertEqual([e[0] for e in events], ["message_start", "message_delta", "message_stop"])


class TestMessagesRouting(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)

    def start_proxy(self, **overrides):
        config = make_config(TARGET_BASE_URL=self.upstream.url, PROVIDER_API_KEY="upstream-key", **overrides)
        proxy = ProxyUnderTest(config)
        self.addCleanup(proxy.close)
        return proxy

    def test_claude_model_relays_through_chat_completions(self):
        self.upstream.respond(
            "/v1/chat/completions",
            json.dumps(
                {
                    "id": "chatcmpl-abc123",
                    "model": "test-target",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi from upstream"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ),
        )
        proxy = self.start_proxy()

        with proxy.post(
            "/v1/messages",
            {
                "model": "claude-opus-4-5",
                "max_tokens": 100,
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            self.assertEqual(response.status, 200)
            body = json.loads(response.read().decode())

        self.assertEqual(self.upstream.received_path, "/v1/chat/completions")
        self.assertEqual(self.upstream.received_body["model"], "test-target")
        self.assertEqual(self.upstream.received_body["max_tokens"], 100)
        self.assertEqual(
            self.upstream.received_body["messages"][0],
            {"role": "system", "content": "You are helpful."},
        )
        self.assertEqual(self.upstream.received_headers.get("Authorization"), "Bearer upstream-key")

        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertEqual(body["id"], "msg_abc123")
        self.assertEqual(body["content"], [{"type": "text", "text": "Hi from upstream"}])
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertEqual(body["usage"], {"input_tokens": 10, "output_tokens": 5})

    def test_claude_streaming_relays_with_sse_translation(self):
        self.upstream.respond(
            "/v1/chat/completions",
            "\n\n".join(
                [
                    'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}',
                    'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" there"}}]}',
                    'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}',
                    "data: [DONE]",
                ]
            )
            + "\n\n",
            content_type="text/event-stream",
        )
        proxy = self.start_proxy()

        with proxy.post(
            "/v1/messages",
            {
                "model": "claude-sonnet-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get("Content-Type"), "text/event-stream")
            raw = response.read().decode()

        self.assertEqual(self.upstream.received_path, "/v1/chat/completions")
        self.assertIs(self.upstream.received_body["stream"], True)
        self.assertEqual(self.upstream.received_headers.get("Accept"), "text/event-stream")

        events = parse_sse_events(raw)
        self.assertEqual(
            [e["__event"] for e in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        text = "".join(
            e["delta"]["text"] for e in events if e["__event"] == "content_block_delta"
        )
        self.assertEqual(text, "Hi there")

    def test_non_anthropic_model_uses_upstream_messages_api(self):
        self.upstream.respond(
            "/v1/messages",
            json.dumps({"type": "message", "content": [{"type": "text", "text": "direct"}]}),
        )
        proxy = self.start_proxy()

        with proxy.post(
            "/v1/messages",
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        ) as response:
            body = response.read().decode()

        self.assertEqual(self.upstream.received_path, "/v1/messages")
        self.assertIn("direct", body)

    def test_falls_back_to_chat_when_upstream_has_no_messages_api(self):
        self.upstream.respond(
            "/v1/chat/completions",
            json.dumps(
                {
                    "id": "chatcmpl-fallback",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "translated"}, "finish_reason": "stop"}
                    ],
                }
            ),
        )
        proxy = self.start_proxy()

        with proxy.post(
            "/v1/messages",
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        ) as response:
            body = json.loads(response.read().decode())

        self.assertEqual(self.upstream.received_path, "/v1/chat/completions")
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertEqual(body["content"], [{"type": "text", "text": "translated"}])

    def test_passthrough_mode_never_translates(self):
        self.upstream.respond(
            "/v1/messages",
            json.dumps({"type": "message", "content": [{"type": "text", "text": "direct"}]}),
        )
        proxy = self.start_proxy(MESSAGES_UPSTREAM_MODE="passthrough")

        with proxy.post(
            "/v1/messages",
            {"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "Hello"}]},
        ) as response:
            body = response.read().decode()

        self.assertEqual(self.upstream.received_path, "/v1/messages")
        self.assertIn("direct", body)

    def test_upstream_error_becomes_anthropic_error(self):
        self.upstream.respond(
            "/v1/chat/completions",
            json.dumps({"error": {"message": "slow down", "type": "rate_limit_exceeded"}}),
            status=429,
        )
        proxy = self.start_proxy()

        from urllib.error import HTTPError

        with self.assertRaises(HTTPError) as cm:
            proxy.post(
                "/v1/messages",
                {"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "Hello"}]},
            )
        self.assertEqual(cm.exception.code, 429)
        body = json.loads(cm.exception.read().decode())
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "rate_limit_error")
        self.assertEqual(body["error"]["message"], "slow down")


class TestOpenAIRoutesAreNotTranslated(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)
        config = make_config(TARGET_BASE_URL=self.upstream.url, PROVIDER_API_KEY="upstream-key")
        self.proxy = ProxyUnderTest(config)
        self.addCleanup(self.proxy.close)

    def test_chat_completions_only_masks_the_model(self):
        self.upstream.respond(
            "/v1/chat/completions",
            json.dumps(
                {
                    "id": "chatcmpl-1",
                    "model": "test-target",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                    ],
                }
            ),
        )

        with self.proxy.post(
            "/v1/chat/completions",
            {
                "model": "claude-opus-5",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.5,
            },
        ) as response:
            body = json.loads(response.read().decode())

        # Request reaches the upstream in OpenAI shape, model swapped.
        self.assertEqual(self.upstream.received_body["model"], "test-target")
        self.assertEqual(self.upstream.received_body["temperature"], 0.5)
        self.assertEqual(
            self.upstream.received_body["messages"], [{"role": "user", "content": "hello"}]
        )
        self.assertNotIn("system", self.upstream.received_body)

        # Response stays in OpenAI shape, model swapped back.
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertIn("choices", body)
        self.assertNotIn("content", body)

    def test_chat_completions_stream_passes_through_masked(self):
        self.upstream.respond(
            "/v1/chat/completions",
            'data: {"id":"chatcmpl-1","model":"test-target","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
            content_type="text/event-stream",
        )

        with self.proxy.post(
            "/v1/chat/completions", {"model": "claude-opus-5", "stream": True}
        ) as response:
            self.assertEqual(response.headers.get("Content-Type"), "text/event-stream")
            raw = response.read().decode()

        self.assertNotIn("test-target", raw)
        self.assertIn('"model":"claude-opus-5"', raw)
        # OpenAI clients on this route keep receiving OpenAI chunks.
        self.assertIn('"choices"', raw)
        self.assertNotIn("content_block_delta", raw)
        self.assertIn("data: [DONE]", raw)

    def test_client_credentials_are_not_leaked_upstream(self):
        self.upstream.respond("/v1/chat/completions", "{}")

        with self.proxy.post(
            "/v1/chat/completions",
            {"model": "claude-opus-5"},
            headers={"Authorization": "Bearer client-key", "x-api-key": "client-key"},
        ):
            pass

        self.assertEqual(self.upstream.received_headers.get("Authorization"), "Bearer upstream-key")
        self.assertIsNone(self.upstream.received_headers.get("x-api-key"))


if __name__ == "__main__":
    unittest.main()
