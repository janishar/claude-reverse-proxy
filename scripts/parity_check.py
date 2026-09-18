#!/usr/bin/env python3
"""
Cross-implementation parity check.

Boots the Go, Python and Node proxies against one shared fake upstream, sends
each of them the same requests, and diffs the answers. The three
implementations are meant to be interchangeable; this is what proves it.

    python3 scripts/parity_check.py

Exits 0 when all three agree, 1 otherwise.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAT_JSON = json.dumps({
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "upstream-model-x",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hi from upstream",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }],
        },
        "finish_reason": "tool_calls",
    }],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}).encode()

CHAT_SSE = ("\n\n".join([
    'data: {"id":"chatcmpl-s1","model":"upstream-model-x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}',
    'data: {"id":"chatcmpl-s1","model":"upstream-model-x","choices":[{"index":0,"delta":{"content":" there"}}]}',
    'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}',
    "data: [DONE]",
]) + "\n\n").encode()

seen = {}


class Upstream(BaseHTTPRequestHandler):
    """An OpenAI-compatible upstream with no Anthropic Messages API."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode())
        except json.JSONDecodeError:
            body = None
        seen[self.path] = {
            "body": body,
            "auth": self.headers.get("Authorization"),
            "x-api-key": self.headers.get("x-api-key"),
            "accept": self.headers.get("Accept"),
        }

        if self.path != "/v1/chat/completions":
            self.reply(404, "application/json", b'{"error":{"message":"no messages api"}}')
            return
        if not isinstance(body, dict):
            self.reply(400, "application/json", b'{"error":{"message":"body must be an object"}}')
            return
        if body.get("stream"):
            self.reply(200, "text/event-stream", CHAT_SSE)
        else:
            self.reply(200, "application/json", CHAT_JSON)

    def reply(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def call(port, path, payload=None, method="POST", headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request_headers = {"Content-Type": "application/json", "Authorization": "Bearer proxy-secret"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode()


def sse_events(raw):
    events = []
    for block in raw.strip().split("\n\n"):
        event_type, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if data and data != "[DONE]":
            try:
                events.append((event_type, json.loads(data)))
            except json.JSONDecodeError:
                pass
    return events


def probe(port):
    """Run the shared request set and collect everything worth comparing."""
    results = {}

    seen.clear()
    _, _, body = call(port, "/v1/models", method="GET")
    results["models"] = json.loads(body)

    seen.clear()
    status, _, body = call(port, "/v1/messages", {
        "model": "claude-opus-5",
        "max_tokens": 64,
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"},
            ]},
        ],
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
        "stop_sequences": ["HALT"],
    })
    results["messages"] = (status, json.loads(body))
    results["messages_upstream"] = seen.get("/v1/chat/completions")

    seen.clear()
    status, content_type, body = call(port, "/v1/messages", {
        "model": "claude-opus-5", "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    events = sse_events(body)
    results["messages_stream"] = (
        status,
        "text/event-stream" in content_type,
        [name for name, _ in events],
        "".join(payload["delta"]["text"] for name, payload in events
                if name == "content_block_delta" and payload["delta"].get("type") == "text_delta"),
        [payload for name, payload in events if name == "message_delta"],
    )

    seen.clear()
    status, _, body = call(port, "/v1/chat/completions", {
        "model": "claude-opus-5", "messages": [{"role": "user", "content": "Hello"}],
    })
    results["chat"] = (status, json.loads(body))
    results["chat_upstream"] = seen.get("/v1/chat/completions")

    seen.clear()
    status, content_type, body = call(port, "/v1/chat/completions", {
        "model": "claude-opus-5", "stream": True,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    models_in_stream = sorted({
        chunk.get("model")
        for chunk in (json.loads(line[len("data:"):].strip())
                      for line in body.splitlines()
                      if line.startswith("data:") and line[len("data:"):].strip() not in ("", "[DONE]"))
    }, key=str)
    results["chat_stream"] = (
        status,
        "text/event-stream" in content_type,
        models_in_stream,
        "content_block_delta" in body,
        body.strip().endswith("data: [DONE]"),
    )

    results["auth_rejects_wrong_key"] = call(
        port, "/v1/chat/completions", {}, headers={"Authorization": "Bearer wrong"})[0]
    results["auth_accepts_x_api_key"] = call(
        port, "/v1/messages", {"model": "claude-opus-5", "messages": []},
        headers={"Authorization": "", "x-api-key": "proxy-secret"})[0]
    results["invalid_json_is_rejected"] = call(
        port, "/v1/chat/completions", "not-json-at-all")[0]
    return results


def normalise(value):
    """Blank out ids that embed a timestamp, which legitimately differ per run."""
    if isinstance(value, dict):
        return {
            key: "<generated-id>"
            if key == "id" and isinstance(item, str) and re.fullmatch(r"msg_\d{10,}", item)
            else normalise(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalise(item) for item in value]
    return value


def main():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"

    env = dict(
        os.environ,
        TARGET_BASE_URL=upstream_url,
        PROXY_MODEL_NAME="claude-opus-5",
        TARGET_MODEL_NAME="upstream-model-x",
        PROXY_MODEL_ID="claude-opus-5",
        PROXY_MODEL_OWNED_BY="cloud-proxy",
        PROXY_MODEL_CREATED="1700000000",
        PROVIDER_API_KEY="upstream-key",
        PROXY_API_KEY="proxy-secret",
    )
    env.pop("MESSAGES_UPSTREAM_MODE", None)
    env.pop("ANTHROPIC_MODEL_MARKERS", None)

    implementations = [
        ("go", ["./goproxy"], "goproxy"),
        ("python", [sys.executable, "main.py"], "pyproxy"),
        ("node", ["node", "main.js"], "nodeproxy"),
    ]

    binary = os.path.join(ROOT, "goproxy", "goproxy")
    if not os.path.exists(binary):
        print("building goproxy...")
        subprocess.run(["go", "build", "-o", "goproxy", "."], cwd=os.path.join(ROOT, "goproxy"), check=True)

    results, processes = {}, []
    try:
        for name, command, directory in implementations:
            port = free_port()
            process = subprocess.Popen(
                command, cwd=os.path.join(ROOT, directory),
                env=dict(env, SERVER_PORT=str(port)),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            processes.append(process)
            if not wait_for_port(port):
                print(f"{name}: failed to start")
                if process.stdout:
                    print(process.stdout.read())
                return 1
            results[name] = probe(port)
            print(f"  probed {name}")
    finally:
        for process in processes:
            process.terminate()
        upstream.shutdown()

    print()
    ok = True
    for key in results["go"]:
        rendered = {
            name: json.dumps(normalise(results[name][key]), sort_keys=True) for name in results
        }
        if len(set(rendered.values())) == 1:
            print(f"  ok       {key}")
        else:
            ok = False
            print(f"  MISMATCH {key}")
            for name, value in rendered.items():
                print(f"             {name:7} {value}")

    print("\nAll three implementations agree." if ok else "\nImplementations disagree.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
