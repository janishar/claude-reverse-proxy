const { PassThrough } = require("stream");

jest.mock("node-fetch", () => jest.fn());
const fetch = require("node-fetch");

const originalEnv = process.env;

describe("Claude model detection", () => {
  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("detects claude model names in any case and form", () => {
    const { isClaudeModelName } = require("./main");
    expect(isClaudeModelName("claude-opus-4-5")).toBe(true);
    expect(isClaudeModelName("CLAUDE-SONNET-4-20250514")).toBe(true);
    expect(isClaudeModelName("Claude-Sonnet-4")).toBe(true);
    expect(isClaudeModelName("claude-haiku-3-5")).toBe(true);
    expect(isClaudeModelName("claude-haiki-1")).toBe(true);
    expect(isClaudeModelName("anthropic.claude-fable-1")).toBe(true);
    expect(isClaudeModelName("gpt-4o")).toBe(false);
    expect(isClaudeModelName("nvidia/nemotron-3-super-120b-a12b")).toBe(false);
    expect(isClaudeModelName("claude")).toBe(false);
    expect(isClaudeModelName(undefined)).toBe(false);
  });

  test("markers are configurable via ANTHROPIC_MODEL_MARKERS", () => {
    process.env.ANTHROPIC_MODEL_MARKERS = "nemotron, mistral";
    jest.resetModules();
    const { isClaudeModelName } = require("./main");
    expect(isClaudeModelName("nvidia/nemotron-3")).toBe(true);
    expect(isClaudeModelName("claude-opus-4-5")).toBe(false);
    delete process.env.ANTHROPIC_MODEL_MARKERS;
  });

  test("routes /v1/messages on the model, not on the upstream vendor", () => {
    const { shouldRouteClaudeToChat } = require("./main");
    const autoCfg = { proxyModelName: "claude-opus-5" };

    expect(shouldRouteClaudeToChat(autoCfg, "claude-opus-4-5")).toBe(true);
    expect(shouldRouteClaudeToChat(autoCfg, "CLAUDE-SONNET-4")).toBe(true);
    expect(shouldRouteClaudeToChat(autoCfg, "gpt-4o")).toBe(false);
    expect(shouldRouteClaudeToChat(autoCfg, undefined)).toBe(false);
    expect(shouldRouteClaudeToChat(autoCfg, "nvidia/nemotron-3-super-120b-a12b")).toBe(false);

    // The proxy's own model name always relays, whatever it is called.
    expect(shouldRouteClaudeToChat({ proxyModelName: "my-model" }, "my-model")).toBe(true);

    const translateCfg = { proxyModelName: "claude-opus-5", messagesUpstreamMode: "translate" };
    expect(shouldRouteClaudeToChat(translateCfg, "gpt-4o")).toBe(true);

    const passthroughCfg = { proxyModelName: "claude-opus-5", messagesUpstreamMode: "passthrough" };
    expect(shouldRouteClaudeToChat(passthroughCfg, "claude-opus-4-5")).toBe(false);
  });
});

describe("anthropicToOpenAIRequest", () => {
  let anthropicToOpenAIRequest;
  const config = { targetModelName: "target-model", proxyModelName: "claude-opus-5" };

  beforeEach(() => {
    jest.resetModules();
    ({ anthropicToOpenAIRequest } = require("./main"));
  });

  test("translates a full claude request", () => {
    const out = anthropicToOpenAIRequest(config, {
      model: "claude-opus-4-5",
      max_tokens: 1024,
      temperature: 0.7,
      top_p: 0.9,
      stop_sequences: ["\n\n", "HALT"],
      system: "You are a helpful assistant.",
      messages: [
        { role: "user", content: "Hello" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check that." },
            { type: "tool_use", id: "toolu_01", name: "get_weather", input: { city: "Paris" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_01", content: "Sunny, 25C" },
          ],
        },
        { role: "user", content: "Thanks" },
      ],
      tools: [
        {
          name: "get_weather",
          description: "Get the weather",
          input_schema: { type: "object", properties: { city: { type: "string" } } },
        },
      ],
      tool_choice: { type: "auto" },
    });

    expect(out.model).toBe("target-model");
    expect(out.max_tokens).toBe(1024);
    expect(out.temperature).toBe(0.7);
    expect(out.top_p).toBe(0.9);
    expect(out.stop).toEqual(["\n\n", "HALT"]);
    expect(out.top_k).toBeUndefined();

    expect(out.messages).toHaveLength(5);
    expect(out.messages[0]).toEqual({ role: "system", content: "You are a helpful assistant." });
    expect(out.messages[1]).toEqual({ role: "user", content: "Hello" });

    const assistant = out.messages[2];
    expect(assistant.role).toBe("assistant");
    expect(assistant.content).toBe("Let me check that.");
    expect(assistant.tool_calls).toHaveLength(1);
    expect(assistant.tool_calls[0]).toEqual({
      id: "toolu_01",
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Paris"}' },
    });

    expect(out.messages[3]).toEqual({
      role: "tool",
      tool_call_id: "toolu_01",
      content: "Sunny, 25C",
    });
    expect(out.messages[4]).toEqual({ role: "user", content: "Thanks" });

    expect(out.tools).toEqual([
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ]);
    expect(out.tool_choice).toBe("auto");
  });

  test("translates tool_choice variants", () => {
    const base = { messages: [] };
    expect(anthropicToOpenAIRequest(config, { ...base, tool_choice: { type: "any" } }).tool_choice).toBe("required");
    expect(anthropicToOpenAIRequest(config, { ...base, tool_choice: { type: "none" } }).tool_choice).toBe("none");
    expect(
      anthropicToOpenAIRequest(config, { ...base, tool_choice: { type: "tool", name: "get_weather" } }).tool_choice
    ).toEqual({ type: "function", function: { name: "get_weather" } });
  });

  test("translates image blocks", () => {
    const out = anthropicToOpenAIRequest(config, {
      model: "claude-opus-4-5",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is in this image?" },
            { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
            { type: "image", source: { type: "url", url: "https://example.com/cat.png" } },
          ],
        },
      ],
    });

    const parts = out.messages[0].content;
    expect(parts).toHaveLength(3);
    expect(parts[0]).toEqual({ type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } });
    expect(parts[1]).toEqual({ type: "image_url", image_url: { url: "https://example.com/cat.png" } });
    expect(parts[2]).toEqual({ type: "text", text: "What is in this image?" });
  });
});

describe("openAIToAnthropicResponse", () => {
  let openAIToAnthropicResponse;
  const config = { targetModelName: "target-model", proxyModelName: "claude-opus-5" };

  beforeEach(() => {
    jest.resetModules();
    ({ openAIToAnthropicResponse } = require("./main"));
  });

  test("translates text response", () => {
    const out = openAIToAnthropicResponse(config, {
      id: "chatcmpl-abc123",
      choices: [
        { index: 0, message: { role: "assistant", content: "Hello there!" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 7, total_tokens: 17 },
    });

    expect(out.id).toBe("msg_abc123");
    expect(out.type).toBe("message");
    expect(out.role).toBe("assistant");
    expect(out.model).toBe("claude-opus-5");
    expect(out.content).toEqual([{ type: "text", text: "Hello there!" }]);
    expect(out.stop_reason).toBe("end_turn");
    expect(out.usage).toEqual({ input_tokens: 10, output_tokens: 7 });
  });

  test("translates tool call response", () => {
    const out = openAIToAnthropicResponse(config, {
      id: "chatcmpl-xyz",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Let me look that up.",
            tool_calls: [
              {
                id: "call_123",
                type: "function",
                function: { name: "get_weather", arguments: '{"city":"Paris"}' },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    });

    expect(out.content).toHaveLength(2);
    expect(out.content[0]).toEqual({ type: "text", text: "Let me look that up." });
    expect(out.content[1]).toEqual({
      type: "tool_use",
      id: "call_123",
      name: "get_weather",
      input: { city: "Paris" },
    });
    expect(out.stop_reason).toBe("tool_use");
  });

  test("maps finish reasons", () => {
    jest.resetModules();
    const { mapFinishReason } = require("./main");
    expect(mapFinishReason("stop")).toBe("end_turn");
    expect(mapFinishReason("length")).toBe("max_tokens");
    expect(mapFinishReason("tool_calls")).toBe("tool_use");
    expect(mapFinishReason("content_filter")).toBe("end_turn");
    expect(mapFinishReason("unknown")).toBe("end_turn");
  });
});

function makeFakeRes() {
  let written = "";
  return {
    get written() {
      return written;
    },
    setHeader: jest.fn(),
    flushHeaders: jest.fn(),
    write: jest.fn((chunk) => {
      written += chunk;
    }),
    status: jest.fn().mockReturnThis(),
    send: jest.fn(),
    json: jest.fn(),
    type: jest.fn().mockReturnThis(),
    end: jest.fn(),
    headersSent: false,
  };
}

describe("translateOpenAIStreamToAnthropic", () => {
  const config = { proxyModelName: "claude-opus-5" };

  function runStream(sseData) {
    jest.resetModules();
    const { translateOpenAIStreamToAnthropic } = require("./main");
    const stream = new PassThrough();
    const res = makeFakeRes();
    const promise = translateOpenAIStreamToAnthropic(config, stream, res);
    stream.end(sseData);
    return promise.then(() => res.written);
  }

  function parseEvents(raw) {
    const events = [];
    for (const block of raw.split("\n\n")) {
      const trimmed = block.trim();
      if (!trimmed) continue;
      let eventType = "";
      const dataLines = [];
      for (const line of trimmed.split("\n")) {
        if (line.startsWith("event: ")) eventType = line.slice(7);
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      }
      if (dataLines.length === 0) continue;
      const parsed = JSON.parse(dataLines.join("\n"));
      parsed.__event = eventType;
      events.push(parsed);
    }
    return events;
  }

  test("translates text stream", async () => {
    const sse = [
      'data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
      "data: [DONE]",
    ].join("\n\n") + "\n\n";

    const raw = await runStream(sse);
    const events = parseEvents(raw);

    expect(events.map((e) => e.__event)).toEqual([
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ]);

    expect(events[0].message.id).toBe("msg_xyz");
    expect(events[0].message.model).toBe("claude-opus-5");
    expect(events[1].content_block).toEqual({ type: "text", text: "" });

    const text = events
      .filter((e) => e.__event === "content_block_delta" && e.delta.type === "text_delta")
      .map((e) => e.delta.text)
      .join("");
    expect(text).toBe("Hello");

    expect(events[5].delta.stop_reason).toBe("end_turn");
    expect(events[5].usage.output_tokens).toBe(2);
  });

  test("translates tool call stream", async () => {
    const chunk = (delta, extra) =>
      "data: " + JSON.stringify(Object.assign({ id: "chatcmpl-abc", choices: [{ index: 0, delta, finish_reason: null }] }, extra || {}));
    const toolDelta = (argumentsStr) => ({ tool_calls: [{ index: 0, function: { arguments: argumentsStr } }] });
    const sse = [
      chunk({ role: "assistant", content: "Checking" }),
      chunk({ tool_calls: [{ index: 0, id: "call_9", type: "function", function: { name: "get_weather", arguments: "" } }] }),
      chunk(toolDelta('{"ci')),
      chunk(toolDelta('ty":"Paris"}')),
      chunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] }),
      "data: [DONE]",
    ].join("\n\n") + "\n\n";

    const raw = await runStream(sse);
    const events = parseEvents(raw);

    expect(events.map((e) => e.__event)).toEqual([
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_stop",
      "content_block_start",
      "content_block_delta",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ]);

    expect(events[1].content_block).toEqual({ type: "text", text: "" });
    expect(events[4].content_block).toEqual({ type: "tool_use", id: "call_9", name: "get_weather", input: {} });

    const partial = events
      .filter((e) => e.__event === "content_block_delta" && e.index === 1 && e.delta.type === "input_json_delta")
      .map((e) => e.delta.partial_json)
      .join("");
    expect(JSON.parse(partial)).toEqual({ city: "Paris" });

    expect(events[8].delta.stop_reason).toBe("tool_use");
  });
});

describe("Routing /v1/messages to chat completions", () => {
  let forwardHandler;
  let routedFetch;

  beforeAll(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
    process.env.TARGET_BASE_URL = "https://upstream.example.com";
    process.env.PROVIDER_API_KEY = "test-key";
    process.env.PROXY_MODEL_NAME = "claude-opus-5";
    process.env.TARGET_MODEL_NAME = "test-target";
    require("./main");
    routedFetch = require("node-fetch");
    const { app } = require("./main");
    const route = app._router.stack.find((layer) => layer.route && layer.route.path === "/v1/messages");
    forwardHandler = route.route.stack[1].handle;
  });

  beforeEach(() => {
    routedFetch.mockReset();
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("claude model routes to chat completions with translation", async () => {
    routedFetch.mockResolvedValue({
      status: 200,
      headers: { forEach: jest.fn() },
      text: async () =>
        JSON.stringify({
          id: "chatcmpl-abc123",
          choices: [
            { index: 0, message: { role: "assistant", content: "Hi from upstream" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        }),
    });

    const req = {
      method: "POST",
      body: {
        model: "claude-opus-4-5",
        max_tokens: 100,
        system: "You are helpful.",
        messages: [{ role: "user", content: "Hello" }],
      },
      headers: {},
    };
    const res = makeFakeRes();

    await forwardHandler(req, res, jest.fn());

    expect(routedFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = routedFetch.mock.calls[0];
    expect(url).toBe("https://upstream.example.com/v1/chat/completions");
    const sentBody = JSON.parse(opts.body);
    expect(sentBody.model).toBe("test-target");
    expect(sentBody.messages[0]).toEqual({ role: "system", content: "You are helpful." });
    expect(sentBody.messages[1]).toEqual({ role: "user", content: "Hello" });
    expect(sentBody.max_tokens).toBe(100);

    const responseJson = JSON.parse(res.send.mock.calls[0][0]);
    expect(responseJson.type).toBe("message");
    expect(responseJson.model).toBe("claude-opus-5");
    expect(responseJson.id).toBe("msg_abc123");
    expect(responseJson.content).toEqual([{ type: "text", text: "Hi from upstream" }]);
    expect(responseJson.stop_reason).toBe("end_turn");
    expect(responseJson.usage).toEqual({ input_tokens: 10, output_tokens: 5 });
  });

  test("claude model streaming routes to chat completions with SSE translation", async () => {
    const stream = new PassThrough();
    Object.assign(stream, { status: 200, headers: { forEach: jest.fn() } });
    routedFetch.mockResolvedValue(stream);

    const req = {
      method: "POST",
      body: {
        model: "claude-sonnet-4-5",
        stream: true,
        messages: [{ role: "user", content: "Hello" }],
      },
      headers: {},
    };
    const res = makeFakeRes();

    const promise = forwardHandler(req, res, jest.fn());
    stream.end(
      [
        'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}',
        "data: [DONE]",
      ].join("\n\n") + "\n\n"
    );
    await promise;

    expect(routedFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = routedFetch.mock.calls[0];
    expect(url).toBe("https://upstream.example.com/v1/chat/completions");
    expect(JSON.parse(opts.body).stream).toBe(true);

    const raw = res.written;
    expect(res.setHeader).toHaveBeenCalledWith("Content-Type", "text/event-stream");
    expect(raw).toContain("event: message_start");
    expect(raw).toContain("event: content_block_start");
    expect(raw).toContain("event: content_block_delta");
    expect(raw).toContain("event: content_block_stop");
    expect(raw).toContain("event: message_delta");
    expect(raw).toContain("event: message_stop");

    const text = raw
      .split("event: content_block_delta")
      .slice(1)
      .map((block) => {
        const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
        return JSON.parse(dataLine.slice(6)).delta.text || "";
      })
      .join("");
    expect(text).toBe("Hi there");
  });

  test("non-claude model is not routed to chat completions", async () => {
    routedFetch.mockResolvedValue({
      status: 200,
      headers: { forEach: jest.fn() },
      text: async () => JSON.stringify({ type: "message", content: [{ type: "text", text: "direct" }] }),
    });

    const req = {
      method: "POST",
      body: {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
      },
      headers: {},
    };
    const res = makeFakeRes();

    await forwardHandler(req, res, jest.fn());

    expect(routedFetch).toHaveBeenCalledTimes(1);
    const [url] = routedFetch.mock.calls[0];
    expect(url).toBe("https://upstream.example.com/v1/messages");
  });
});
