#!/usr/bin/env python3
"""
Python reverse proxy for OpenAI-compatible APIs.

Downstream clients see a Claude-shaped surface: they call /v1/messages (or the
OpenAI routes) with the model name this proxy advertises, e.g. claude-opus-5.
Upstream is any OpenAI-compatible API.

Two things happen on the way through:

1. Model masking. The downstream model name is swapped for TARGET_MODEL_NAME on
   the way out, and the upstream name is swapped back on the way in, so callers
   only ever see the model they asked for.
2. Messages relay. When /v1/messages carries an Anthropic model, the request is
   translated into an OpenAI chat-completions call and the answer -- streamed or
   not -- is translated back into Anthropic events. Requests that arrive on the
   OpenAI routes are forwarded unchanged apart from the model name.
"""

import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Messages upstream modes, selected with MESSAGES_UPSTREAM_MODE.
#   auto        translate /v1/messages when the model is an Anthropic one,
#               otherwise try the upstream Messages API and fall back on 404
#   translate   always translate /v1/messages to /v1/chat/completions
#   passthrough never translate: /v1/messages always goes upstream as-is
MODE_AUTO = "auto"
MODE_TRANSLATE = "translate"
MODE_PASSTHROUGH = "passthrough"

# Substrings that identify an Anthropic model name.
DEFAULT_ANTHROPIC_MODEL_MARKERS = [
    "opus",
    "sonnet",
    "haik",
    "fable",
    "anthropic",
    "claude-",
]

# Headers that belong to a single hop and must not be relayed.
HOP_BY_HOP_HEADERS = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)


def dumps(value: Any) -> str:
    """Compact JSON, byte-for-byte identical to what Go and Node emit."""
    return json.dumps(value, separators=(",", ":"))


def get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def parse_model_markers(raw: str) -> List[str]:
    return [part.strip().lower() for part in (raw or "").split(",") if part.strip()]


class Config:
    def __init__(self):
        self.server_port = int(os.getenv("SERVER_PORT", "8080"))
        self.target_base_url = os.getenv("TARGET_BASE_URL", "http://localhost:8000").rstrip("/")
        self.target_models_path = os.getenv("TARGET_MODELS_PATH", "/v1/models")
        self.target_chat_path = os.getenv("TARGET_CHAT_PATH", "/v1/chat/completions")
        self.target_responses_path = os.getenv("TARGET_RESPONSES_PATH", "/v1/responses")
        self.target_embeddings_path = os.getenv("TARGET_EMBEDDINGS_PATH", "/v1/embeddings")
        self.target_moderations_path = os.getenv("TARGET_MODERATIONS_PATH", "/v1/moderations")
        self.target_completions_path = os.getenv("TARGET_COMPLETIONS_PATH", "/v1/completions")
        self.target_messages_path = os.getenv("TARGET_MESSAGES_PATH", "/v1/messages")
        self.proxy_model_name = os.getenv("PROXY_MODEL_NAME", "claude-opus-5")
        self.target_model_name = os.getenv("TARGET_MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")
        self.proxy_model_id = os.getenv("PROXY_MODEL_ID", "claude-opus-5")
        self.proxy_model_object = os.getenv("PROXY_MODEL_OBJECT", "model")
        self.proxy_model_created = get_env_int("PROXY_MODEL_CREATED", int(time.time()))
        self.proxy_model_owned_by = os.getenv("PROXY_MODEL_OWNED_BY", "cloud-proxy")
        self.provider_api_key = os.getenv("PROVIDER_API_KEY", "")
        self.proxy_api_key = os.getenv("PROXY_API_KEY", "")
        self.messages_upstream_mode = os.getenv("MESSAGES_UPSTREAM_MODE", MODE_AUTO).lower()
        self.max_body_bytes = get_env_int("MAX_BODY_MB", 50) * 1024 * 1024
        self.anthropic_model_markers = (
            parse_model_markers(os.getenv("ANTHROPIC_MODEL_MARKERS", ""))
            or list(DEFAULT_ANTHROPIC_MODEL_MARKERS)
        )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def extract_bearer_token(auth_header: str) -> str:
    if not auth_header:
        return ""
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1]


def check_auth(handler: BaseHTTPRequestHandler, config: Config) -> bool:
    """Accepts an OpenAI-style bearer token or an Anthropic-style x-api-key."""
    if not config.proxy_api_key:
        return True
    token = extract_bearer_token(handler.headers.get("Authorization", ""))
    if not token:
        token = handler.headers.get("x-api-key", "")
    if token != config.proxy_api_key:
        send_json_error(handler, 401, "Unauthorized")
        return False
    return True


# ---------------------------------------------------------------------------
# Model masking
# ---------------------------------------------------------------------------


def is_anthropic_model(config: Config, model: Any) -> bool:
    if not isinstance(model, str) or not model:
        return False
    lower = model.lower()
    return any(marker in lower for marker in config.anthropic_model_markers)


def should_mask_model(config: Config, model: Any) -> bool:
    """True when an inbound model name has to be swapped for the upstream one."""
    if not isinstance(model, str) or not model:
        return False
    return model == config.proxy_model_name or is_anthropic_model(config, model)


def mask_request_model(config: Config, req_body: Dict[str, Any]) -> Dict[str, Any]:
    if should_mask_model(config, req_body.get("model")):
        req_body = dict(req_body)
        req_body["model"] = config.target_model_name
    return req_body


def replace_model_in_response(config: Config, body: bytes) -> bytes:
    """Hide the upstream model name so callers only see the one they asked for."""
    try:
        resp = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return body
    if not isinstance(resp, dict):
        return body

    changed = False

    def mask(container: Any) -> None:
        nonlocal changed
        if isinstance(container, dict) and container.get("model") == config.target_model_name:
            container["model"] = config.proxy_model_name
            changed = True

    mask(resp)

    if isinstance(resp.get("choices"), list):
        for choice in resp["choices"]:
            if isinstance(choice, dict):
                mask(choice.get("message"))

    if isinstance(resp.get("data"), list):
        for item in resp["data"]:
            mask(item)

    # Streamed Anthropic events carry the model inside message_start.
    mask(resp.get("message"))

    if not changed:
        return body
    return dumps(resp).encode("utf-8")


def should_route_claude_to_chat(config: Config, model: Any) -> bool:
    """Whether /v1/messages has to be relayed through /v1/chat/completions."""
    if config.messages_upstream_mode == MODE_PASSTHROUGH:
        return False
    if config.messages_upstream_mode == MODE_TRANSLATE:
        return True
    if is_anthropic_model(config, model):
        return True
    return bool(model) and model == config.proxy_model_name


# ---------------------------------------------------------------------------
# Anthropic Messages -> OpenAI Chat Completions
# ---------------------------------------------------------------------------


def to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


def image_to_openai_part(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and source.get("media_type") and source.get("data"):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"},
        }
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def anthropic_to_openai_request(config: Config, req: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model": config.target_model_name}
    messages: List[Dict[str, Any]] = []

    if "system" in req:
        text = content_to_text(req["system"])
        if text:
            messages.append({"role": "system", "content": text})

    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
                continue
            if isinstance(content, list):
                text_parts: List[str] = []
                image_parts: List[Dict[str, Any]] = []
                tool_msgs: List[Dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
                    elif block_type == "image":
                        part = image_to_openai_part(block)
                        if part:
                            image_parts.append(part)
                    elif block_type == "tool_result":
                        tool_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": content_to_text(block.get("content")),
                            }
                        )
                # Tool results come first so the upstream sees every tool_call
                # answered before the turn's own input.
                messages.extend(tool_msgs)
                if image_parts:
                    parts: List[Dict[str, Any]] = list(image_parts)
                    if text_parts:
                        parts.append({"type": "text", "text": "\n".join(text_parts)})
                    messages.append({"role": "user", "content": parts})
                elif text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        # assistant (or any other role): string content passes through, block
        # content expands text blocks and turns tool_use into tool_calls.
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            text_parts = []
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": dumps(block.get("input") or {}),
                            },
                        }
                    )
            converted: Dict[str, Any] = {
                "role": role,
                "content": "\n".join(text_parts) if text_parts else "",
            }
            if tool_calls:
                converted["tool_calls"] = tool_calls
            messages.append(converted)

    out["messages"] = messages

    for key in (
        "max_tokens",
        "temperature",
        "top_p",
        "stream",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "n",
    ):
        if key in req:
            out[key] = req[key]
    if "stop_sequences" in req:
        out["stop"] = req["stop_sequences"]
    if "response_format" in req:
        out["response_format"] = req["response_format"]
    metadata = req.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        out["user"] = metadata["user_id"]

    tools = req.get("tools")
    if isinstance(tools, list) and tools:
        oai_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn: Dict[str, Any] = {}
            if isinstance(tool.get("name"), str):
                fn["name"] = tool["name"]
            if isinstance(tool.get("description"), str):
                fn["description"] = tool["description"]
            if "input_schema" in tool:
                fn["parameters"] = tool["input_schema"]
            oai_tools.append({"type": "function", "function": fn})
        if oai_tools:
            out["tools"] = oai_tools

    if "tool_choice" in req:
        tc = req["tool_choice"]
        if isinstance(tc, dict):
            tc_type = tc.get("type")
            if tc_type == "auto":
                out["tool_choice"] = "auto"
            elif tc_type == "any":
                out["tool_choice"] = "required"
            elif tc_type == "none":
                out["tool_choice"] = "none"
            elif tc_type == "tool" and isinstance(tc.get("name"), str):
                out["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }
        elif isinstance(tc, str):
            out["tool_choice"] = tc

    return out


# ---------------------------------------------------------------------------
# OpenAI Chat Completions -> Anthropic Messages
# ---------------------------------------------------------------------------


def map_finish_reason(finish_reason: Any) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, "end_turn")


def anthropic_error_type(status: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        529: "overloaded_error",
    }.get(status, "api_error")


def anthropic_error(status: int, message: str) -> bytes:
    return dumps(
        {"type": "error", "error": {"type": anthropic_error_type(status), "message": message}}
    ).encode("utf-8")


def openai_error_to_anthropic(status: int, body: bytes) -> bytes:
    """Reshape an upstream error into the envelope Anthropic clients expect."""
    try:
        message = body.decode("utf-8").strip()
    except (UnicodeDecodeError, AttributeError):
        message = ""
    try:
        parsed = json.loads(message)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict) and error.get("message"):
                message = str(error["message"])
            elif parsed.get("message"):
                message = str(parsed["message"])
    except json.JSONDecodeError:
        pass
    return anthropic_error(status, message or f"Upstream returned {status}")


def openai_to_anthropic_response(config: Config, resp: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "model": config.proxy_model_name,
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

    resp_id = resp.get("id")
    if isinstance(resp_id, str) and resp_id:
        out["id"] = "msg_" + resp_id[len("chatcmpl-"):] if resp_id.startswith("chatcmpl-") else f"msg_{resp_id}"

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                out["content"].append({"type": "text", "text": content})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    arguments = fn.get("arguments")
                    tool_input: Any = {}
                    if isinstance(arguments, str) and arguments:
                        try:
                            tool_input = json.loads(arguments)
                        except json.JSONDecodeError:
                            tool_input = arguments
                    out["content"].append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": fn.get("name"),
                            "input": tool_input,
                        }
                    )
        if choice.get("finish_reason"):
            out["stop_reason"] = map_finish_reason(choice["finish_reason"])

    usage = resp.get("usage")
    if isinstance(usage, dict):
        out["usage"] = {
            "input_tokens": to_int(usage.get("prompt_tokens")),
            "output_tokens": to_int(usage.get("completion_tokens")),
        }

    return out


class AnthropicStreamTranslator:
    """Turns OpenAI chat-completion SSE chunks into Anthropic message events.

    Feed it one SSE line at a time with ``feed_line``; call ``finish`` when the
    upstream stream ends. ``emit(event_type, data)`` receives each event.
    """

    def __init__(self, config: Config, emit: Callable[[str, Dict[str, Any]], None]):
        self.config = config
        self.emit = emit
        self.msg_id = f"msg_{int(time.time() * 1000)}"
        self.started = False
        self.text_open = False
        self.block_index = 0
        self.open_tool_blocks: Dict[int, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self.stop_sent = False
        self.done = False

    def _emit_start(self) -> None:
        if self.started:
            return
        self.started = True
        self.emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.config.proxy_model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            },
        )

    def _close_text(self) -> None:
        if self.text_open:
            self.emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": self.block_index},
            )
            self.text_open = False
            self.block_index += 1

    def _close_all(self) -> None:
        self._close_text()
        for index in self.open_tool_blocks.values():
            self.emit("content_block_stop", {"type": "content_block_stop", "index": index})
        self.open_tool_blocks = {}

    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("data:"):
            return
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return
        if isinstance(chunk, dict):
            self.feed_chunk(chunk)

    def feed_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id and not self.started:
            self.msg_id = (
                "msg_" + chunk_id[len("chatcmpl-"):]
                if chunk_id.startswith("chatcmpl-")
                else f"msg_{chunk_id}"
            )

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = to_int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                self.output_tokens = to_int(usage["completion_tokens"])

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta")

        if isinstance(delta, dict):
            text = delta.get("content")
            if isinstance(text, str) and text:
                if not self.text_open:
                    self._emit_start()
                    self.emit(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": self.block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    self.text_open = True
                self.emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    tc_index = to_int(tc.get("index"))
                    if tc_index not in self.open_tool_blocks:
                        self._emit_start()
                        self._close_text()
                        self.open_tool_blocks[tc_index] = self.block_index
                        self.emit(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": self.block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.get("id"),
                                    "name": fn.get("name"),
                                    "input": {},
                                },
                            },
                        )
                        self.block_index += 1
                    arguments = fn.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        self.emit(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self.open_tool_blocks[tc_index],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments,
                                },
                            },
                        )

        if choice.get("finish_reason"):
            self.stop_reason = map_finish_reason(choice["finish_reason"])

    def finish(self) -> None:
        if self.stop_sent:
            return
        self.stop_sent = True
        self._emit_start()
        self._close_all()
        self.emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
            },
        )
        self.emit("message_stop", {"type": "message_stop"})


def translate_openai_stream_to_anthropic(
    config: Config, upstream: Any, emit: Callable[[str, Dict[str, Any]], None]
) -> None:
    translator = AnthropicStreamTranslator(config, emit)
    for raw_line in iter(upstream.readline, b""):
        translator.feed_line(raw_line.decode("utf-8", errors="replace"))
        if translator.done:
            break
    translator.finish()


# ---------------------------------------------------------------------------
# Upstream forwarding
# ---------------------------------------------------------------------------


def is_event_stream(content_type: Any) -> bool:
    return "text/event-stream" in str(content_type or "").lower()


def build_upstream_headers(
    handler: BaseHTTPRequestHandler, config: Config, want_stream: bool
) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    for key, value in handler.headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS or lower in ("content-length", "host", "accept-encoding"):
            continue
        headers[key] = value
    headers["Content-Type"] = "application/json"
    if want_stream:
        headers["Accept"] = "text/event-stream"
    elif not any(key.lower() == "accept" for key in headers):
        headers["Accept"] = "application/json"
    if config.provider_api_key:
        # Never leak the caller's own credential to the upstream.
        headers["Authorization"] = f"Bearer {config.provider_api_key}"
        headers.pop("x-api-key", None)
        headers.pop("X-Api-Key", None)
        headers.pop("X-api-key", None)
    return headers


def open_upstream(
    handler: BaseHTTPRequestHandler,
    config: Config,
    target_url: str,
    body: Optional[bytes],
    want_stream: bool = False,
):
    request = urllib.request.Request(
        target_url,
        data=body,
        headers=build_upstream_headers(handler, config, want_stream),
        method=handler.command,
    )
    return urllib.request.urlopen(request)


def drain(stream: Any, remaining: int, chunk_size: int = 65536) -> None:
    """Read and discard `remaining` bytes, stopping early if the peer goes away."""
    while remaining > 0:
        chunk = stream.read(min(chunk_size, remaining))
        if not chunk:
            return
        remaining -= len(chunk)


def send_json_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    payload = dumps({"error": {"message": message, "type": "proxy_error"}}).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def send_anthropic_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    payload = anthropic_error(status, message)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def copy_response_headers(handler: BaseHTTPRequestHandler, headers: Any) -> None:
    items = headers.items() if hasattr(headers, "items") else []
    for key, value in items:
        lower = key.lower()
        # Content-Length/Encoding are dropped: the body may be rewritten here
        # and urllib has already decoded any upstream compression.
        if lower in HOP_BY_HOP_HEADERS or lower in ("content-length", "content-encoding"):
            continue
        handler.send_header(key, value)


def mask_model_in_sse_line(config: Config, line: bytes) -> bytes:
    """Rewrite the model name inside one `data:` frame, leaving others alone."""
    stripped = line.rstrip(b"\r\n")
    if not stripped.startswith(b"data:"):
        return line
    payload = stripped[len(b"data:"):].strip()
    if not payload or payload == b"[DONE]":
        return line
    masked = replace_model_in_response(config, payload)
    if masked == payload:
        return line
    return b"data: " + masked + b"\n"


def stream_upstream_response(handler: BaseHTTPRequestHandler, config: Config, response: Any) -> None:
    handler.send_response(getattr(response, "status", 200))
    copy_response_headers(handler, getattr(response, "headers", None))
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.close_connection = True
    handler.end_headers()
    for line in iter(response.readline, b""):
        handler.wfile.write(mask_model_in_sse_line(config, line))
        handler.wfile.flush()


def write_upstream_response(handler: BaseHTTPRequestHandler, config: Config, response: Any) -> None:
    headers = getattr(response, "headers", None)
    content_type = headers.get("Content-Type", "") if hasattr(headers, "get") else ""
    if is_event_stream(content_type):
        stream_upstream_response(handler, config, response)
        return

    body = replace_model_in_response(config, response.read())
    handler.send_response(getattr(response, "status", 200))
    copy_response_headers(handler, headers)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def handle_claude_via_chat_completions(
    handler: BaseHTTPRequestHandler, config: Config, parsed: Dict[str, Any]
) -> None:
    """Relay one Anthropic Messages request through chat completions."""
    openai_req = anthropic_to_openai_request(config, parsed)
    want_stream = bool(openai_req.get("stream"))
    body = dumps(openai_req).encode("utf-8")
    target_url = config.target_base_url + config.target_chat_path

    try:
        response = open_upstream(handler, config, target_url, body, want_stream)
    except urllib.error.HTTPError as exc:
        payload = openai_error_to_anthropic(exc.code, exc.read())
        handler.send_response(exc.code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        return
    except Exception as exc:  # noqa: BLE001 - upstream failures become 502s
        logger.error("Forward to chat completions failed: %s", exc)
        send_anthropic_error(handler, 502, "Failed to forward request to upstream")
        return

    with response:
        if want_stream:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Connection", "close")
            handler.close_connection = True
            handler.end_headers()

            def emit(event_type: str, data: Dict[str, Any]) -> None:
                handler.wfile.write(
                    f"event: {event_type}\ndata: {dumps(data)}\n\n".encode("utf-8")
                )
                handler.wfile.flush()

            translate_openai_stream_to_anthropic(config, response, emit)
            return

        raw = response.read()
        try:
            converted = dumps(
                openai_to_anthropic_response(config, json.loads(raw.decode("utf-8")))
            ).encode("utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            converted = replace_model_in_response(config, raw)

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(converted)))
        handler.end_headers()
        handler.wfile.write(converted)


def handle_messages(
    handler: BaseHTTPRequestHandler,
    config: Config,
    body: Optional[bytes],
    parsed: Optional[Dict[str, Any]],
) -> None:
    """Serve /v1/messages.

    Anthropic models are relayed through chat completions; anything else is
    offered to the upstream Messages API first and only translated if the
    upstream turns out not to implement it.
    """
    if parsed is not None and should_route_claude_to_chat(config, parsed.get("model")):
        handle_claude_via_chat_completions(handler, config, parsed)
        return

    out_body = body
    want_stream = False
    if parsed is not None:
        out_body = dumps(mask_request_model(config, parsed)).encode("utf-8")
        want_stream = bool(parsed.get("stream"))

    target_url = config.target_base_url + config.target_messages_path
    try:
        with open_upstream(handler, config, target_url, out_body, want_stream) as response:
            write_upstream_response(handler, config, response)
        return
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and parsed is not None:
            # Upstream has no Messages API: translate and retry against chat.
            exc.close()
            handle_claude_via_chat_completions(handler, config, parsed)
            return
        write_upstream_response(handler, config, exc)
        return
    except Exception as exc:  # noqa: BLE001 - upstream failures become 502s
        logger.error("Forward request failed: %s", exc)
        send_anthropic_error(handler, 502, "Failed to forward request to upstream")


def forward_request(handler: BaseHTTPRequestHandler, config: Config, target_path: str) -> None:
    """Forward one request upstream, masking the model name on the way."""
    content_length = int(handler.headers.get("Content-Length", 0) or 0)
    if config.max_body_bytes > 0 and content_length > config.max_body_bytes:
        # Drain what the client is still sending: replying before the request is
        # complete makes the client see a reset instead of the 413.
        drain(handler.rfile, content_length)
        send_json_error(handler, 413, "Request body too large")
        handler.close_connection = True
        return
    body = handler.rfile.read(content_length) if content_length > 0 else None

    parsed: Optional[Dict[str, Any]] = None
    if body and handler.command in ("POST", "PUT", "PATCH"):
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            send_json_error(handler, 400, "Invalid JSON")
            return
        # These APIs only ever take a JSON object; a bare string, list or null
        # would be forwarded as something the upstream cannot use.
        if not isinstance(decoded, dict):
            send_json_error(handler, 400, "Invalid JSON")
            return
        parsed = decoded

    if target_path == config.target_messages_path:
        handle_messages(handler, config, body, parsed)
        return

    want_stream = False
    if parsed is not None:
        body = dumps(mask_request_model(config, parsed)).encode("utf-8")
        want_stream = bool(parsed.get("stream"))

    target_url = config.target_base_url + target_path
    try:
        with open_upstream(handler, config, target_url, body, want_stream) as response:
            write_upstream_response(handler, config, response)
    except urllib.error.HTTPError as exc:
        write_upstream_response(handler, config, exc)
    except Exception as exc:  # noqa: BLE001 - upstream failures become 502s
        logger.error("Forward request failed: %s", exc)
        send_json_error(handler, 502, "Failed to forward request")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class ProxyHandler(BaseHTTPRequestHandler):
    config = Config()
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def do_PUT(self):
        self._dispatch()

    def do_DELETE(self):
        self._dispatch()

    def do_PATCH(self):
        self._dispatch()

    def _dispatch(self):
        if not check_auth(self, self.config):
            return
        self.handle_request()

    def handle_request(self):
        path = self.path.split("?", 1)[0]

        if path == "/v1/models" and self.command == "GET":
            self.handle_models()
        elif path == "/v1/chat/completions" and self.command == "POST":
            forward_request(self, self.config, self.config.target_chat_path)
        elif path == "/v1/responses" and self.command == "POST":
            forward_request(self, self.config, self.config.target_responses_path)
        elif path == "/v1/embeddings" and self.command == "POST":
            forward_request(self, self.config, self.config.target_embeddings_path)
        elif path == "/v1/moderations" and self.command == "POST":
            forward_request(self, self.config, self.config.target_moderations_path)
        elif path == "/v1/completions" and self.command == "POST":
            forward_request(self, self.config, self.config.target_completions_path)
        elif path == "/v1/messages" and self.command == "POST":
            forward_request(self, self.config, self.config.target_messages_path)
        elif self.path.startswith("/v1/"):
            forward_request(self, self.config, self.path)
        else:
            send_json_error(self, 404, "Not found")

    def handle_models(self):
        response = {
            "object": "list",
            "data": [
                {
                    "id": self.config.proxy_model_id,
                    "object": self.config.proxy_model_object,
                    "created": self.config.proxy_model_created,
                    "owned_by": self.config.proxy_model_owned_by,
                }
            ],
        }
        self.send_json_response(200, response)

    def send_json_response(self, status: int, data: Dict[Any, Any]):
        payload = dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def main():
    config = Config()
    ProxyHandler.config = config
    server = ThreadingHTTPServer(("", config.server_port), ProxyHandler)
    logger.info("Server starting on port %s", config.server_port)
    logger.info("Target API: %s", config.target_base_url)
    logger.info(
        "Model mapping: %s -> %s", config.proxy_model_name, config.target_model_name
    )
    logger.info(
        "Messages relay mode: %s (markers: %s)",
        config.messages_upstream_mode,
        ",".join(config.anthropic_model_markers),
    )
    if config.proxy_api_key:
        logger.info("Proxy API key authentication: ENABLED")
    server.serve_forever()


if __name__ == "__main__":
    main()
