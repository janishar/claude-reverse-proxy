/**
 * End-to-end tests: a real upstream HTTP server behind the real proxy, driven
 * over the network. node-fetch is deliberately NOT mocked here, so streaming,
 * headers and status codes are exercised for real.
 */

const http = require("http");

const originalEnv = process.env;

/** Records the last request and replays a scripted response. */
function startUpstream() {
  const state = { path: null, headers: null, body: null, routes: {} };
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString();
      state.path = req.url;
      state.headers = req.headers;
      try {
        state.body = raw ? JSON.parse(raw) : null;
      } catch (e) {
        state.body = null;
      }

      const route = state.routes[req.url.split("?")[0]];
      if (!route) {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: "no such route" } }));
        return;
      }
      res.writeHead(route.status, { "Content-Type": route.contentType });
      res.end(route.payload);
    });
  });
  state.server = server;
  state.respond = (path, payload, { status = 200, contentType = "application/json" } = {}) => {
    state.routes[path] = { status, contentType, payload };
  };
  return state;
}

function listen(server) {
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server.address().port)));
}

function close(server) {
  return new Promise((resolve) => (server ? server.close(resolve) : resolve()));
}

function parseSSEEvents(raw) {
  const events = [];
  for (const block of raw.split("\n\n")) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    let eventType = "";
    let data = null;
    for (const line of trimmed.split("\n")) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) data = line.slice(5).trim();
    }
    if (!data || data === "[DONE]") continue;
    const parsed = JSON.parse(data);
    parsed.__event = eventType;
    events.push(parsed);
  }
  return events;
}

describe("proxy end to end", () => {
  let upstream;
  let proxy;
  let proxyUrl;

  async function startProxy(env = {}) {
    jest.resetModules();
    process.env = {
      ...originalEnv,
      SERVER_PORT: "0",
      TARGET_BASE_URL: `http://127.0.0.1:${upstream.server.address().port}`,
      PROXY_MODEL_NAME: "claude-opus-5",
      TARGET_MODEL_NAME: "test-target",
      PROVIDER_API_KEY: "upstream-key",
      ...env,
    };
    delete process.env.PROXY_API_KEY;
    const { startServer } = require("./main");
    proxy = startServer();
    await new Promise((resolve) => proxy.on("listening", resolve));
    proxyUrl = `http://127.0.0.1:${proxy.address().port}`;
  }

  function post(path, payload, headers = {}) {
    return fetch(proxyUrl + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: typeof payload === "string" ? payload : JSON.stringify(payload),
    });
  }

  beforeEach(async () => {
    upstream = startUpstream();
    await listen(upstream.server);
  });

  afterEach(async () => {
    await close(proxy);
    await close(upstream.server);
    proxy = null;
    process.env = originalEnv;
  });

  describe("OpenAI routes are forwarded, not translated", () => {
    test("only the model name is rewritten, in both directions", async () => {
      upstream.respond(
        "/v1/chat/completions",
        JSON.stringify({
          id: "chatcmpl-1",
          model: "test-target",
          choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
        }),
      );
      await startProxy();

      const res = await post("/v1/chat/completions", {
        model: "claude-opus-5",
        messages: [{ role: "user", content: "hello" }],
        temperature: 0.5,
      });
      const body = await res.json();

      expect(res.status).toBe(200);
      expect(upstream.path).toBe("/v1/chat/completions");
      expect(upstream.body.model).toBe("test-target");
      expect(upstream.body.temperature).toBe(0.5);
      expect(upstream.body.messages).toEqual([{ role: "user", content: "hello" }]);
      expect(upstream.body.system).toBeUndefined();

      expect(body.model).toBe("claude-opus-5");
      expect(body.choices).toBeDefined();
      expect(body.content).toBeUndefined();
    });

    test("any anthropic model name is masked, foreign ones are left alone", async () => {
      upstream.respond("/v1/chat/completions", JSON.stringify({ model: "test-target" }));
      await startProxy();

      await post("/v1/chat/completions", { model: "claude-sonnet-4-5", messages: [] });
      expect(upstream.body.model).toBe("test-target");

      await post("/v1/chat/completions", { model: "gpt-4o", messages: [] });
      expect(upstream.body.model).toBe("gpt-4o");
    });

    test("streamed chunks stay in OpenAI shape with the model masked", async () => {
      upstream.respond(
        "/v1/chat/completions",
        'data: {"id":"chatcmpl-1","model":"test-target","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
        { contentType: "text/event-stream" },
      );
      await startProxy();

      const res = await post("/v1/chat/completions", { model: "claude-opus-5", stream: true });
      const raw = await res.text();

      expect(res.headers.get("content-type")).toContain("text/event-stream");
      expect(upstream.headers.accept).toBe("text/event-stream");
      expect(raw).not.toContain("test-target");
      expect(raw).toContain('"model":"claude-opus-5"');
      expect(raw).toContain('"choices"');
      expect(raw).not.toContain("content_block_delta");
      expect(raw.trim().endsWith("data: [DONE]")).toBe(true);
    });

    test("the caller's credential is replaced, never relayed", async () => {
      upstream.respond("/v1/chat/completions", "{}");
      await startProxy();

      await post(
        "/v1/chat/completions",
        { model: "claude-opus-5" },
        { Authorization: "Bearer client-key", "x-api-key": "client-key" },
      );

      expect(upstream.headers.authorization).toBe("Bearer upstream-key");
      expect(upstream.headers["x-api-key"]).toBeUndefined();
    });

    test("malformed JSON is rejected before the upstream is called", async () => {
      upstream.respond("/v1/chat/completions", "{}");
      await startProxy();

      const res = await post("/v1/chat/completions", "not json");

      expect(res.status).toBe(400);
      expect(upstream.path).toBeNull();
    });

    test("each endpoint reaches its configured target path", async () => {
      await startProxy({
        TARGET_CHAT_PATH: "/custom/chat",
        TARGET_RESPONSES_PATH: "/custom/responses",
        TARGET_EMBEDDINGS_PATH: "/custom/embeddings",
        TARGET_MODERATIONS_PATH: "/custom/moderations",
        TARGET_COMPLETIONS_PATH: "/custom/completions",
      });

      const cases = {
        "/v1/chat/completions": "/custom/chat",
        "/v1/responses": "/custom/responses",
        "/v1/embeddings": "/custom/embeddings",
        "/v1/moderations": "/custom/moderations",
        "/v1/completions": "/custom/completions",
      };
      for (const [inbound, target] of Object.entries(cases)) {
        upstream.respond(target, "{}");
        upstream.path = null;
        await post(inbound, { model: "claude-opus-5" });
        expect(upstream.path).toBe(target);
      }
    });

    test("an oversized body is rejected with a JSON 413", async () => {
      upstream.respond("/v1/chat/completions", "{}");
      await startProxy({ MAX_BODY_MB: "1" });

      const res = await post("/v1/chat/completions", {
        model: "claude-opus-5",
        pad: "x".repeat(2 * 1024 * 1024),
      });

      expect(res.status).toBe(413);
      expect(res.headers.get("content-type")).toContain("application/json");
      expect((await res.json()).error.message).toBe("Request body too large");
      expect(upstream.path).toBeNull();
    });

    test("a body under the limit is forwarded", async () => {
      upstream.respond("/v1/chat/completions", "{}");
      await startProxy({ MAX_BODY_MB: "1" });

      await post("/v1/chat/completions", { model: "claude-opus-5", pad: "x".repeat(1024) });

      expect(upstream.path).toBe("/v1/chat/completions");
    });

    test("an unreachable upstream becomes a 502, not a crash", async () => {
      await startProxy({ TARGET_BASE_URL: "http://127.0.0.1:1" });

      const res = await post("/v1/chat/completions", { model: "claude-opus-5" });

      expect(res.status).toBe(502);
      expect((await res.json()).error).toBeDefined();
    });

    test("an upstream error status is preserved", async () => {
      upstream.respond("/v1/chat/completions", JSON.stringify({ error: { message: "nope" } }), {
        status: 400,
      });
      await startProxy();

      const res = await post("/v1/chat/completions", { model: "claude-opus-5" });

      expect(res.status).toBe(400);
      expect(await res.text()).toContain("nope");
    });

    test("the wildcard route keeps the inbound path and query", async () => {
      upstream.respond("/v1/files", JSON.stringify({ object: "list" }));
      await startProxy();

      const res = await fetch(`${proxyUrl}/v1/files?limit=2`);

      expect(res.status).toBe(200);
      expect(upstream.path).toBe("/v1/files?limit=2");
    });
  });

  describe("/v1/models", () => {
    test("is served locally without calling the upstream", async () => {
      await startProxy({
        PROXY_MODEL_ID: "test-id",
        PROXY_MODEL_OWNED_BY: "test-owner",
        PROXY_MODEL_CREATED: "1789100219",
      });

      const res = await fetch(`${proxyUrl}/v1/models`);
      const body = await res.json();

      expect(res.status).toBe(200);
      expect(body.object).toBe("list");
      expect(body.data).toEqual([
        { id: "test-id", object: "model", created: 1789100219, owned_by: "test-owner" },
      ]);
      expect(upstream.path).toBeNull();
    });

    test("a non-GET is forwarded upstream", async () => {
      upstream.respond("/v1/models", JSON.stringify({ object: "list", data: [] }));
      await startProxy();

      const res = await post("/v1/models", { model: "claude-opus-5" });

      expect(res.status).toBe(200);
      expect(upstream.path).toBe("/v1/models");
    });
  });

  describe("/v1/messages relay", () => {
    test("an anthropic model is relayed through chat completions", async () => {
      upstream.respond(
        "/v1/chat/completions",
        JSON.stringify({
          id: "chatcmpl-abc123",
          model: "test-target",
          choices: [
            { index: 0, message: { role: "assistant", content: "Hi from upstream" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 5 },
        }),
      );
      await startProxy();

      const res = await post("/v1/messages", {
        model: "claude-opus-4-5",
        max_tokens: 100,
        system: "You are helpful.",
        messages: [{ role: "user", content: "Hello" }],
      });
      const body = await res.json();

      expect(upstream.path).toBe("/v1/chat/completions");
      expect(upstream.body.model).toBe("test-target");
      expect(upstream.body.max_tokens).toBe(100);
      expect(upstream.body.messages[0]).toEqual({ role: "system", content: "You are helpful." });

      expect(body.type).toBe("message");
      expect(body.role).toBe("assistant");
      expect(body.model).toBe("claude-opus-5");
      expect(body.id).toBe("msg_abc123");
      expect(body.content).toEqual([{ type: "text", text: "Hi from upstream" }]);
      expect(body.stop_reason).toBe("end_turn");
      expect(body.usage).toEqual({ input_tokens: 10, output_tokens: 5 });
    });

    test("a streamed relay produces anthropic events and closes the response", async () => {
      upstream.respond(
        "/v1/chat/completions",
        [
          'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}',
          'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" there"}}]}',
          'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}',
          "data: [DONE]",
        ].join("\n\n") + "\n\n",
        { contentType: "text/event-stream" },
      );
      await startProxy();

      const res = await post("/v1/messages", {
        model: "claude-sonnet-4-5",
        stream: true,
        messages: [{ role: "user", content: "Hello" }],
      });
      // This read only returns because the proxy ends the response.
      const raw = await res.text();

      expect(res.headers.get("content-type")).toContain("text/event-stream");
      expect(upstream.body.stream).toBe(true);

      const events = parseSSEEvents(raw);
      expect(events.map((e) => e.__event)).toEqual([
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
      ]);
      const text = events
        .filter((e) => e.__event === "content_block_delta")
        .map((e) => e.delta.text)
        .join("");
      expect(text).toBe("Hi there");
      expect(events[0].message.model).toBe("claude-opus-5");
    });

    test("a non-anthropic model uses the upstream Messages API", async () => {
      upstream.respond(
        "/v1/messages",
        JSON.stringify({ type: "message", content: [{ type: "text", text: "direct" }] }),
      );
      await startProxy();

      const res = await post("/v1/messages", {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(upstream.path).toBe("/v1/messages");
      expect(await res.text()).toContain("direct");
    });

    test("a 404 from the upstream Messages API falls back to translation", async () => {
      upstream.respond(
        "/v1/chat/completions",
        JSON.stringify({
          id: "chatcmpl-fallback",
          choices: [
            { index: 0, message: { role: "assistant", content: "translated" }, finish_reason: "stop" },
          ],
        }),
      );
      await startProxy();

      const res = await post("/v1/messages", {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
      });
      const body = await res.json();

      expect(upstream.path).toBe("/v1/chat/completions");
      expect(body.type).toBe("message");
      expect(body.model).toBe("claude-opus-5");
      expect(body.content).toEqual([{ type: "text", text: "translated" }]);
    });

    test("passthrough mode never translates", async () => {
      upstream.respond(
        "/v1/messages",
        JSON.stringify({ type: "message", content: [{ type: "text", text: "direct" }] }),
      );
      await startProxy({ MESSAGES_UPSTREAM_MODE: "passthrough" });

      const res = await post("/v1/messages", {
        model: "claude-opus-4-5",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(upstream.path).toBe("/v1/messages");
      expect(await res.text()).toContain("direct");
    });

    test("an upstream error comes back in the anthropic envelope", async () => {
      upstream.respond(
        "/v1/chat/completions",
        JSON.stringify({ error: { message: "slow down", type: "rate_limit_exceeded" } }),
        { status: 429 },
      );
      await startProxy();

      const res = await post("/v1/messages", {
        model: "claude-opus-4-5",
        messages: [{ role: "user", content: "Hello" }],
      });
      const body = await res.json();

      expect(res.status).toBe(429);
      expect(body.type).toBe("error");
      expect(body.error.type).toBe("rate_limit_error");
      expect(body.error.message).toBe("slow down");
    });

    test("an unreachable upstream becomes a 502, not a crash", async () => {
      await startProxy({ TARGET_BASE_URL: "http://127.0.0.1:1" });

      const res = await post("/v1/messages", {
        model: "claude-opus-4-5",
        messages: [{ role: "user", content: "Hello" }],
      });
      const body = await res.json();

      expect(res.status).toBe(502);
      expect(body.type).toBe("error");
    });
  });

  describe("proxy authentication", () => {
    test("accepts a bearer token or an x-api-key, rejects anything else", async () => {
      upstream.respond("/v1/chat/completions", "{}");
      await startProxy();
      // startProxy deletes PROXY_API_KEY, so re-start with it set.
      await close(proxy);
      jest.resetModules();
      process.env = {
        ...process.env,
        SERVER_PORT: "0",
        TARGET_BASE_URL: `http://127.0.0.1:${upstream.server.address().port}`,
        PROXY_API_KEY: "secret",
        PROVIDER_API_KEY: "upstream-key",
      };
      const { startServer } = require("./main");
      proxy = startServer();
      await new Promise((resolve) => proxy.on("listening", resolve));
      proxyUrl = `http://127.0.0.1:${proxy.address().port}`;

      expect((await post("/v1/chat/completions", {}, { Authorization: "Bearer secret" })).status).toBe(200);
      expect((await post("/v1/chat/completions", {}, { "x-api-key": "secret" })).status).toBe(200);
      expect((await post("/v1/chat/completions", {}, { Authorization: "Bearer wrong" })).status).toBe(401);
      expect((await post("/v1/chat/completions", {})).status).toBe(401);
    });
  });
});
