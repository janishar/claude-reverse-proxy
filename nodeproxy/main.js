/**
 * Node.js reverse proxy for OpenAI-compatible APIs.
 *
 * Downstream clients see a Claude-shaped surface: they call /v1/messages (or
 * the OpenAI routes) with the model name this proxy advertises, e.g.
 * claude-opus-5. Upstream is any OpenAI-compatible API.
 *
 * Two things happen on the way through:
 *
 * 1. Model masking. The downstream model name is swapped for TARGET_MODEL_NAME
 *    on the way out and swapped back on the way in, so callers only ever see
 *    the model they asked for.
 * 2. Messages relay. When /v1/messages carries an Anthropic model the request
 *    is translated into an OpenAI chat-completions call and the answer --
 *    streamed or not -- is translated back into Anthropic events. Requests that
 *    arrive on the OpenAI routes are forwarded unchanged apart from the model.
 */

const express = require("express");
const fetch = require("node-fetch");

// Messages upstream modes, selected with MESSAGES_UPSTREAM_MODE.
//   auto        translate /v1/messages when the model is an Anthropic one,
//               otherwise try the upstream Messages API and fall back on 404
//   translate   always translate /v1/messages to /v1/chat/completions
//   passthrough never translate: /v1/messages always goes upstream as-is
const MODE_AUTO = "auto";
const MODE_TRANSLATE = "translate";
const MODE_PASSTHROUGH = "passthrough";

// Substrings that identify an Anthropic model name.
const DEFAULT_ANTHROPIC_MODEL_MARKERS = [
  "opus",
  "sonnet",
  "haik",
  "fable",
  "anthropic",
  "claude-",
];

// Headers that belong to a single hop and must not be relayed.
const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function parseModelMarkers(raw) {
  return String(raw || "")
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);
}

const config = {
  serverPort: parseInt(process.env.SERVER_PORT || "8080", 10),
  targetBaseUrl: (process.env.TARGET_BASE_URL || "http://localhost:8000").replace(/\/+$/, ""),
  targetModelsPath: process.env.TARGET_MODELS_PATH || "/v1/models",
  targetChatPath: process.env.TARGET_CHAT_PATH || "/v1/chat/completions",
  targetResponsesPath: process.env.TARGET_RESPONSES_PATH || "/v1/responses",
  targetEmbeddingsPath: process.env.TARGET_EMBEDDINGS_PATH || "/v1/embeddings",
  targetModerationsPath: process.env.TARGET_MODERATIONS_PATH || "/v1/moderations",
  targetCompletionsPath: process.env.TARGET_COMPLETIONS_PATH || "/v1/completions",
  targetMessagesPath: process.env.TARGET_MESSAGES_PATH || "/v1/messages",
  proxyModelName: process.env.PROXY_MODEL_NAME || "claude-opus-5",
  targetModelName: process.env.TARGET_MODEL_NAME || "nvidia/nemotron-3-super-120b-a12b",
  proxyModelId: process.env.PROXY_MODEL_ID || "claude-opus-5",
  proxyModelObject: process.env.PROXY_MODEL_OBJECT || "model",
  proxyModelCreated: parseInt(
    process.env.PROXY_MODEL_CREATED || String(Math.floor(Date.now() / 1000)),
    10,
  ),
  proxyModelOwnedBy: process.env.PROXY_MODEL_OWNED_BY || "cloud-proxy",
  providerApiKey: process.env.PROVIDER_API_KEY || "",
  proxyApiKey: process.env.PROXY_API_KEY || "",
  messagesUpstreamMode: (process.env.MESSAGES_UPSTREAM_MODE || MODE_AUTO).toLowerCase(),
  anthropicModelMarkers:
    parseModelMarkers(process.env.ANTHROPIC_MODEL_MARKERS).length > 0
      ? parseModelMarkers(process.env.ANTHROPIC_MODEL_MARKERS)
      : [...DEFAULT_ANTHROPIC_MODEL_MARKERS],
};

const app = express();
// Raw bodies: the proxy parses JSON itself so it can forward exact bytes and
// reject malformed payloads with an API-shaped error rather than Express HTML.
app.use(express.raw({ type: () => true, limit: `${parseInt(process.env.MAX_BODY_MB || "50", 10)}mb` }));

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

function extractBearerToken(authHeader) {
  if (!authHeader) return "";
  const parts = String(authHeader).split(" ");
  if (parts.length !== 2 || parts[0].toLowerCase() !== "bearer") return "";
  return parts[1];
}

/** Accepts an OpenAI-style bearer token or an Anthropic-style x-api-key. */
function authMiddleware(req, res, next) {
  if (!config.proxyApiKey) return next();
  const headers = req.headers || {};
  const token = extractBearerToken(headers.authorization) || headers["x-api-key"] || "";
  if (token !== config.proxyApiKey) {
    return sendProxyError(res, 401, "Unauthorized");
  }
  next();
}

// ---------------------------------------------------------------------------
// Model masking
// ---------------------------------------------------------------------------

function isClaudeModelName(model) {
  if (typeof model !== "string" || !model) return false;
  const lower = model.toLowerCase();
  return config.anthropicModelMarkers.some((marker) => lower.includes(marker));
}

/** True when an inbound model name has to be swapped for the upstream one. */
function shouldMaskModel(model) {
  if (typeof model !== "string" || !model) return false;
  return model === config.proxyModelName || isClaudeModelName(model);
}

function modifyBodyForProxy(body) {
  if (body && typeof body === "object" && shouldMaskModel(body.model)) {
    return { ...body, model: config.targetModelName };
  }
  return body;
}

/** Hide the upstream model name so callers only see the one they asked for. */
function replaceModelInResponse(body) {
  let resp;
  try {
    resp = JSON.parse(body);
  } catch (e) {
    return body;
  }
  if (!resp || typeof resp !== "object") return body;

  let changed = false;
  const mask = (container) => {
    if (container && typeof container === "object" && container.model === config.targetModelName) {
      container.model = config.proxyModelName;
      changed = true;
    }
  };

  mask(resp);

  if (Array.isArray(resp.choices)) {
    for (const choice of resp.choices) {
      if (choice && typeof choice === "object") mask(choice.message);
    }
  }

  if (Array.isArray(resp.data)) {
    for (const item of resp.data) mask(item);
  }

  // Streamed Anthropic events carry the model inside message_start.
  mask(resp.message);

  return changed ? JSON.stringify(resp) : body;
}

/** Whether /v1/messages has to be relayed through /v1/chat/completions. */
function shouldRouteClaudeToChat(cfg, model) {
  const mode = (cfg && cfg.messagesUpstreamMode) || MODE_AUTO;
  if (mode === MODE_PASSTHROUGH) return false;
  if (mode === MODE_TRANSLATE) return true;
  if (isClaudeModelName(model)) return true;
  return Boolean(model) && model === (cfg && cfg.proxyModelName);
}

// ---------------------------------------------------------------------------
// Anthropic Messages -> OpenAI Chat Completions
// ---------------------------------------------------------------------------

function contentToText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((block) => block && block.type === "text" && typeof block.text === "string")
      .map((block) => block.text)
      .join("\n");
  }
  return "";
}

function imageToOpenAIPart(block) {
  const source = block.source;
  if (!source) return null;
  if (source.type === "base64" && source.media_type && source.data) {
    return {
      type: "image_url",
      image_url: { url: `data:${source.media_type};base64,${source.data}` },
    };
  }
  if (source.type === "url" && source.url) {
    return { type: "image_url", image_url: { url: source.url } };
  }
  return null;
}

function anthropicToOpenAIRequest(config, req) {
  const out = { model: config.targetModelName };
  const messages = [];

  if (req.system !== undefined) {
    const text = contentToText(req.system);
    if (text) messages.push({ role: "system", content: text });
  }

  if (Array.isArray(req.messages)) {
    for (const msg of req.messages) {
      if (!msg || typeof msg !== "object") continue;
      const role = msg.role;
      const content = msg.content;

      if (role === "user") {
        if (typeof content === "string") {
          messages.push({ role: "user", content });
          continue;
        }
        if (Array.isArray(content)) {
          const textParts = [];
          const imageParts = [];
          const toolMsgs = [];
          for (const block of content) {
            if (!block || typeof block !== "object") continue;
            if (block.type === "text" && typeof block.text === "string") {
              textParts.push(block.text);
            } else if (block.type === "image") {
              const part = imageToOpenAIPart(block);
              if (part) imageParts.push(part);
            } else if (block.type === "tool_result") {
              toolMsgs.push({
                role: "tool",
                tool_call_id: block.tool_use_id,
                content: contentToText(block.content),
              });
            }
          }
          // Tool results come first so the upstream sees every tool_call
          // answered before the turn's own input.
          messages.push(...toolMsgs);
          if (imageParts.length > 0) {
            const parts = [...imageParts];
            if (textParts.length > 0) {
              parts.push({ type: "text", text: textParts.join("\n") });
            }
            messages.push({ role: "user", content: parts });
          } else if (textParts.length > 0) {
            messages.push({ role: "user", content: textParts.join("\n") });
          }
        }
        continue;
      }

      if (typeof content === "string") {
        messages.push({ role, content });
        continue;
      }
      if (Array.isArray(content)) {
        const textParts = [];
        const toolCalls = [];
        for (const block of content) {
          if (!block || typeof block !== "object") continue;
          if (block.type === "text" && typeof block.text === "string") {
            textParts.push(block.text);
          } else if (block.type === "tool_use") {
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: block.name,
                arguments: JSON.stringify(block.input || {}),
              },
            });
          }
        }
        const converted = {
          role,
          content: textParts.length > 0 ? textParts.join("\n") : "",
        };
        if (toolCalls.length > 0) converted.tool_calls = toolCalls;
        messages.push(converted);
      }
    }
  }
  out.messages = messages;

  for (const key of [
    "max_tokens",
    "temperature",
    "top_p",
    "stream",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "n",
  ]) {
    if (req[key] !== undefined) out[key] = req[key];
  }
  if (req.stop_sequences !== undefined) out.stop = req.stop_sequences;
  if (req.response_format !== undefined) out.response_format = req.response_format;
  if (req.metadata && typeof req.metadata === "object" && req.metadata.user_id) {
    out.user = req.metadata.user_id;
  }

  if (Array.isArray(req.tools) && req.tools.length > 0) {
    const oaiTools = req.tools
      .filter((tool) => tool && typeof tool === "object")
      .map((tool) => {
        const fn = {};
        if (typeof tool.name === "string") fn.name = tool.name;
        if (typeof tool.description === "string") fn.description = tool.description;
        if (tool.input_schema !== undefined) fn.parameters = tool.input_schema;
        return { type: "function", function: fn };
      });
    if (oaiTools.length > 0) out.tools = oaiTools;
  }

  if (req.tool_choice !== undefined) {
    const tc = req.tool_choice;
    if (typeof tc === "object" && tc !== null) {
      switch (tc.type) {
        case "auto":
          out.tool_choice = "auto";
          break;
        case "any":
          out.tool_choice = "required";
          break;
        case "none":
          out.tool_choice = "none";
          break;
        case "tool":
          if (typeof tc.name === "string") {
            out.tool_choice = { type: "function", function: { name: tc.name } };
          }
          break;
      }
    } else if (typeof tc === "string") {
      out.tool_choice = tc;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions -> Anthropic Messages
// ---------------------------------------------------------------------------

function mapFinishReason(finishReason) {
  switch (finishReason) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool_calls":
      return "tool_use";
    default:
      return "end_turn";
  }
}

function anthropicErrorType(status) {
  switch (status) {
    case 400:
      return "invalid_request_error";
    case 401:
      return "authentication_error";
    case 403:
      return "permission_error";
    case 404:
      return "not_found_error";
    case 413:
      return "request_too_large";
    case 429:
      return "rate_limit_error";
    case 529:
      return "overloaded_error";
    default:
      return "api_error";
  }
}

function anthropicError(status, message) {
  return JSON.stringify({
    type: "error",
    error: { type: anthropicErrorType(status), message },
  });
}

/** Reshape an upstream error into the envelope Anthropic clients expect. */
function openAIErrorToAnthropic(status, body) {
  let message = String(body || "").trim();
  try {
    const parsed = JSON.parse(message);
    if (parsed && typeof parsed === "object") {
      if (parsed.error && parsed.error.message) message = String(parsed.error.message);
      else if (parsed.message) message = String(parsed.message);
    }
  } catch (e) {
    // Not JSON: keep the raw text as the message.
  }
  return anthropicError(status, message || `Upstream returned ${status}`);
}

function openAIToAnthropicResponse(config, resp) {
  const out = {
    id: `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    model: config.proxyModelName,
    content: [],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 },
  };

  if (typeof resp.id === "string" && resp.id) {
    out.id = `msg_${resp.id.replace(/^chatcmpl-/, "")}`;
  }

  if (Array.isArray(resp.choices) && resp.choices.length > 0) {
    const choice = resp.choices[0];
    if (choice && choice.message) {
      if (typeof choice.message.content === "string" && choice.message.content !== "") {
        out.content.push({ type: "text", text: choice.message.content });
      }
      if (Array.isArray(choice.message.tool_calls)) {
        for (const tc of choice.message.tool_calls) {
          if (!tc || !tc.function) continue;
          let input = {};
          if (typeof tc.function.arguments === "string" && tc.function.arguments !== "") {
            try {
              input = JSON.parse(tc.function.arguments);
            } catch (e) {
              input = tc.function.arguments;
            }
          }
          out.content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input,
          });
        }
      }
    }
    if (choice && typeof choice.finish_reason === "string" && choice.finish_reason) {
      out.stop_reason = mapFinishReason(choice.finish_reason);
    }
  }

  if (resp.usage && typeof resp.usage === "object") {
    out.usage = {
      input_tokens: Number(resp.usage.prompt_tokens) || 0,
      output_tokens: Number(resp.usage.completion_tokens) || 0,
    };
  }

  return out;
}

async function translateOpenAIStreamToAnthropic(config, upstream, res) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  if (typeof res.flushHeaders === "function") res.flushHeaders();

  let msgId = `msg_${Date.now()}`;
  let started = false;
  let textOpen = false;
  let blockIndex = 0;
  let openToolBlocks = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = "end_turn";
  let stopSent = false;

  const send = (eventType, data) => {
    res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const emitStart = () => {
    if (started) return;
    started = true;
    send("message_start", {
      type: "message_start",
      message: {
        id: msgId,
        type: "message",
        role: "assistant",
        model: config.proxyModelName,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    });
  };

  const closeText = () => {
    if (textOpen) {
      send("content_block_stop", { type: "content_block_stop", index: blockIndex });
      textOpen = false;
      blockIndex++;
    }
  };

  const closeAll = () => {
    closeText();
    if (openToolBlocks) {
      for (const idx of Object.values(openToolBlocks)) {
        send("content_block_stop", { type: "content_block_stop", index: idx });
      }
      openToolBlocks = null;
    }
  };

  const emitStop = () => {
    if (stopSent) return;
    stopSent = true;
    emitStart();
    closeAll();
    send("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
    });
    send("message_stop", { type: "message_stop" });
    if (typeof res.end === "function") res.end();
  };

  const processChunk = (chunk) => {
    if (typeof chunk.id === "string" && chunk.id && !started) {
      msgId = `msg_${chunk.id.replace(/^chatcmpl-/, "")}`;
    }

    if (chunk.usage && typeof chunk.usage === "object") {
      if (chunk.usage.prompt_tokens !== undefined) {
        inputTokens = Number(chunk.usage.prompt_tokens) || 0;
      }
      if (chunk.usage.completion_tokens !== undefined) {
        outputTokens = Number(chunk.usage.completion_tokens) || 0;
      }
    }

    if (Array.isArray(chunk.choices) && chunk.choices.length > 0) {
      const choice = chunk.choices[0];
      const delta = choice && choice.delta;

      if (delta) {
        if (typeof delta.content === "string" && delta.content !== "") {
          if (!textOpen) {
            emitStart();
            send("content_block_start", {
              type: "content_block_start",
              index: blockIndex,
              content_block: { type: "text", text: "" },
            });
            textOpen = true;
          }
          send("content_block_delta", {
            type: "content_block_delta",
            index: blockIndex,
            delta: { type: "text_delta", text: delta.content },
          });
        }
        if (Array.isArray(delta.tool_calls)) {
          if (!openToolBlocks) openToolBlocks = {};
          for (const tc of delta.tool_calls) {
            if (!tc) continue;
            const fn = tc.function || {};
            const tcIndex = Number(tc.index) || 0;
            if (!(tcIndex in openToolBlocks)) {
              emitStart();
              closeText();
              openToolBlocks[tcIndex] = blockIndex;
              send("content_block_start", {
                type: "content_block_start",
                index: blockIndex,
                content_block: {
                  type: "tool_use",
                  id: tc.id,
                  name: fn.name,
                  input: {},
                },
              });
              blockIndex++;
            }
            if (typeof fn.arguments === "string" && fn.arguments !== "") {
              send("content_block_delta", {
                type: "content_block_delta",
                index: openToolBlocks[tcIndex],
                delta: { type: "input_json_delta", partial_json: fn.arguments },
              });
            }
          }
        }
      }

      if (typeof choice.finish_reason === "string" && choice.finish_reason) {
        stopReason = mapFinishReason(choice.finish_reason);
      }
    }
  };

  return new Promise((resolve, reject) => {
    let buffer = "";
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      emitStop();
      resolve();
    };

    upstream.on("data", (chunk) => {
      buffer += chunk.toString();
      let idx;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          finish();
          return;
        }
        try {
          processChunk(JSON.parse(data));
        } catch (e) {
          // Ignore malformed chunk
        }
      }
    });
    upstream.on("end", finish);
    upstream.on("error", (err) => {
      if (finished) return;
      finished = true;
      reject(err);
    });
  });
}

// ---------------------------------------------------------------------------
// Upstream forwarding
// ---------------------------------------------------------------------------

function isEventStream(contentType) {
  return String(contentType || "").toLowerCase().includes("text/event-stream");
}

function headerValue(headers, name) {
  if (!headers) return "";
  if (typeof headers.get === "function") return headers.get(name) || "";
  return headers[name] || headers[name.toLowerCase()] || "";
}

/**
 * Carry over the client's headers minus the ones that belong to this hop, and
 * swap in the provider credential so the caller's own key never goes upstream.
 */
function buildUpstreamHeaders(req, wantStream) {
  const headers = {};
  for (const [key, value] of Object.entries((req && req.headers) || {})) {
    const lower = key.toLowerCase();
    if (
      HOP_BY_HOP_HEADERS.has(lower) ||
      lower === "content-length" ||
      lower === "host" ||
      lower === "accept-encoding"
    ) {
      continue;
    }
    headers[lower] = value;
  }
  headers["content-type"] = "application/json";
  // Always explicit: node-fetch would otherwise default to */*.
  if (wantStream) headers["accept"] = "text/event-stream";
  else if (!headers["accept"]) headers["accept"] = "application/json";
  if (config.providerApiKey) {
    headers["authorization"] = `Bearer ${config.providerApiKey}`;
    delete headers["x-api-key"];
  }
  return headers;
}

function copyResponseHeaders(res, headers) {
  if (!headers || typeof headers.forEach !== "function") return;
  headers.forEach((value, key) => {
    const lower = String(key).toLowerCase();
    // Content-Length/Encoding are dropped: the body may be rewritten here and
    // node-fetch has already decoded any upstream compression.
    if (
      HOP_BY_HOP_HEADERS.has(lower) ||
      lower === "content-length" ||
      lower === "content-encoding"
    ) {
      return;
    }
    res.setHeader(key, value);
  });
}

/** Rewrite the model name inside one `data:` frame, leaving others alone. */
function maskModelInSSELine(line) {
  const stripped = line.replace(/[\r\n]+$/, "");
  if (!stripped.startsWith("data:")) return line;
  const payload = stripped.slice(5).trim();
  if (!payload || payload === "[DONE]") return line;
  const masked = replaceModelInResponse(payload);
  if (masked === payload) return line;
  return `data: ${masked}\n`;
}

function pipeSSEWithModelMasking(upstream, res) {
  return new Promise((resolve, reject) => {
    let buffer = "";
    upstream.on("data", (chunk) => {
      buffer += chunk.toString();
      let idx;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx + 1);
        buffer = buffer.slice(idx + 1);
        res.write(maskModelInSSELine(line));
      }
    });
    upstream.on("end", () => {
      if (buffer) res.write(maskModelInSSELine(buffer));
      if (typeof res.end === "function") res.end();
      resolve();
    });
    upstream.on("error", reject);
  });
}

async function sendUpstreamResponse(res, response, requestedStream) {
  const contentType = headerValue(response.headers, "content-type");
  const upstream = response.body;
  const streamable = upstream && typeof upstream.on === "function";

  if (streamable && (isEventStream(contentType) || requestedStream)) {
    copyResponseHeaders(res, response.headers);
    res.status(response.status);
    if (typeof res.flushHeaders === "function") res.flushHeaders();
    await pipeSSEWithModelMasking(upstream, res);
    return;
  }

  const data = replaceModelInResponse(await response.text());
  copyResponseHeaders(res, response.headers);
  res.status(response.status).send(data);
}

/** Relay one Anthropic Messages request through chat completions. */
async function forwardClaudeToChatCompletions(req, res, anthropicBody) {
  const openAIReq = anthropicToOpenAIRequest(config, anthropicBody);
  const wantStream = Boolean(openAIReq.stream);
  const chatTargetUrl = `${config.targetBaseUrl}${config.targetChatPath}`;

  try {
    const response = await fetch(chatTargetUrl, {
      method: req.method,
      headers: buildUpstreamHeaders(req, wantStream),
      body: JSON.stringify(openAIReq),
    });

    if (response.status >= 400) {
      const data = await response.text();
      res
        .status(response.status)
        .type("application/json")
        .send(openAIErrorToAnthropic(response.status, data));
      return;
    }

    if (wantStream) {
      await translateOpenAIStreamToAnthropic(config, response.body || response, res);
      return;
    }

    const text = await response.text();
    let converted;
    try {
      converted = JSON.stringify(openAIToAnthropicResponse(config, JSON.parse(text)));
    } catch (e) {
      // Pass non-JSON responses through unchanged apart from the model name.
      converted = replaceModelInResponse(text);
    }
    res.setHeader("Content-Type", "application/json");
    res.status(response.status).send(converted);
  } catch (error) {
    console.error("Forward to chat completions failed:", error);
    sendAnthropicError(res, 502, "Failed to forward request to upstream");
  }
}

/** Last-resort error writer: never throws, even on a half-written response. */
function sendAnthropicError(res, status, message) {
  try {
    if (res.headersSent) {
      if (typeof res.end === "function") res.end();
      return;
    }
    res.status(status).type("application/json").send(anthropicError(status, message));
  } catch (e) {
    console.error("Failed to send error response:", e);
  }
}

function sendProxyError(res, status, message) {
  try {
    if (res.headersSent) {
      if (typeof res.end === "function") res.end();
      return;
    }
    res.status(status).json({ error: { message, type: "proxy_error" } });
  } catch (e) {
    console.error("Failed to send error response:", e);
  }
}

/**
 * Serve /v1/messages. Anthropic models are relayed through chat completions;
 * anything else is offered to the upstream Messages API first and only
 * translated if the upstream turns out not to implement it.
 */
async function forwardMessages(req, res, body) {
  if (body && typeof body === "object" && shouldRouteClaudeToChat(config, body.model)) {
    return forwardClaudeToChatCompletions(req, res, body);
  }

  const wantStream = Boolean(body && body.stream);
  const targetUrl = `${config.targetBaseUrl}${config.targetMessagesPath}`;

  try {
    const response = await fetch(targetUrl, {
      method: req.method,
      headers: buildUpstreamHeaders(req, wantStream),
      body: upstreamBody(req, modifyBodyForProxy(body)),
    });

    if (response.status === 404 && body && typeof body === "object") {
      // Upstream has no Messages API: translate and retry against chat completions.
      return forwardClaudeToChatCompletions(req, res, body);
    }

    await sendUpstreamResponse(res, response, wantStream);
  } catch (error) {
    console.error("Forward request failed:", error);
    sendAnthropicError(res, 502, "Failed to forward request to upstream");
  }
}

/** Forward one request upstream, masking the model name on the way. */
async function forwardRequest(req, res, targetPath) {
  const parsed = parseRequestBody(req);
  if (parsed.invalid) {
    return sendProxyError(res, 400, "Invalid JSON");
  }

  if (targetPath === config.targetMessagesPath) {
    return forwardMessages(req, res, parsed.body);
  }

  const body = modifyBodyForProxy(parsed.body);
  const wantStream = Boolean(body && body.stream);

  try {
    const response = await fetch(`${config.targetBaseUrl}${targetPath}`, {
      method: req.method,
      headers: buildUpstreamHeaders(req, wantStream),
      body: upstreamBody(req, body),
    });
    await sendUpstreamResponse(res, response, wantStream);
  } catch (error) {
    console.error("Forward request failed:", error);
    sendProxyError(res, 502, "Failed to forward request");
  }
}

const METHODS_WITH_BODY = new Set(["POST", "PUT", "PATCH"]);

/**
 * express.raw gives a Buffer; a mounted JSON parser (or a test) may hand us an
 * already-parsed object instead. Both are accepted. body-parser leaves a plain
 * `{}` behind when there was nothing to parse, which is not a body.
 */
function parseRequestBody(req) {
  const raw = req && req.body;
  if (Buffer.isBuffer(raw)) {
    if (raw.length === 0) return { body: null };
    let parsed;
    try {
      parsed = JSON.parse(raw.toString("utf8"));
    } catch (e) {
      return { invalid: true };
    }
    // These APIs only ever take a JSON object; a bare string, list or null
    // would be forwarded as something the upstream cannot use.
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { invalid: true };
    return { body: parsed };
  }
  if (raw && typeof raw === "object" && Object.keys(raw).length > 0) return { body: raw };
  return { body: null };
}

/** node-fetch rejects a body on GET/HEAD, so only serialise one where it belongs. */
function upstreamBody(req, body) {
  if (body === null || body === undefined) return undefined;
  if (!METHODS_WITH_BODY.has(String((req && req.method) || "POST").toUpperCase())) return undefined;
  return JSON.stringify(body);
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/v1/models", authMiddleware, (req, res) => {
  res.json({
    object: "list",
    data: [
      {
        id: config.proxyModelId,
        object: config.proxyModelObject,
        created: config.proxyModelCreated,
        owned_by: config.proxyModelOwnedBy,
      },
    ],
  });
});

app.post("/v1/chat/completions", authMiddleware, (req, res) => forwardRequest(req, res, config.targetChatPath));
app.post("/v1/responses", authMiddleware, (req, res) => forwardRequest(req, res, config.targetResponsesPath));
app.post("/v1/embeddings", authMiddleware, (req, res) => forwardRequest(req, res, config.targetEmbeddingsPath));
app.post("/v1/moderations", authMiddleware, (req, res) => forwardRequest(req, res, config.targetModerationsPath));
app.post("/v1/completions", authMiddleware, (req, res) => forwardRequest(req, res, config.targetCompletionsPath));
app.post("/v1/messages", authMiddleware, (req, res) => forwardRequest(req, res, config.targetMessagesPath));

// Catch-all: anything else under /v1/ keeps its inbound path and query.
app.all("/v1/*", authMiddleware, async (req, res) => {
  await forwardRequest(req, res, req.originalUrl);
});

// body-parser rejects oversized payloads before any route runs; answer those in
// the same JSON envelope as every other proxy error instead of Express's HTML.
// eslint-disable-next-line no-unused-vars -- Express needs the 4-arg signature.
app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  const status = (err && err.status) || 500;
  const message =
    err && err.type === "entity.too.large"
      ? "Request body too large"
      : (err && err.message) || "Proxy error";
  sendProxyError(res, status, message);
});

function startServer() {
  return app.listen(config.serverPort, () => {
    console.log(`Server starting on port ${config.serverPort}`);
    console.log(`Target API: ${config.targetBaseUrl}`);
    console.log(`Model mapping: ${config.proxyModelName} -> ${config.targetModelName}`);
    console.log(
      `Messages relay mode: ${config.messagesUpstreamMode} (markers: ${config.anthropicModelMarkers.join(",")})`,
    );
    if (config.proxyApiKey) {
      console.log("Proxy API key authentication: ENABLED");
    }
    if (!config.providerApiKey && !/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])/i.test(config.targetBaseUrl)) {
      console.warn("PROVIDER_API_KEY is not set: upstream requests will carry no credential");
    }
  });
}

if (require.main === module) {
  startServer();
}

module.exports = {
  app,
  config,
  startServer,
  isClaudeModelName,
  shouldMaskModel,
  shouldRouteClaudeToChat,
  modifyBodyForProxy,
  replaceModelInResponse,
  maskModelInSSELine,
  contentToText,
  anthropicToOpenAIRequest,
  mapFinishReason,
  anthropicErrorType,
  openAIErrorToAnthropic,
  openAIToAnthropicResponse,
  translateOpenAIStreamToAnthropic,
  buildUpstreamHeaders,
  parseRequestBody,
};
