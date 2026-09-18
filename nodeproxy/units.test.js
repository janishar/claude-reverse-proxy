/**
 * Unit tests for the pure helpers: config parsing, body handling, model masking
 * and error shaping. Routing and translation live in the other suites.
 */

const originalEnv = process.env;

function loadMain(env = {}) {
  jest.resetModules();
  process.env = { ...originalEnv, ...env };
  for (const key of [
    "SERVER_PORT", "TARGET_BASE_URL", "PROXY_MODEL_NAME", "TARGET_MODEL_NAME",
    "PROXY_MODEL_ID", "PROXY_MODEL_OBJECT", "PROXY_MODEL_CREATED", "PROXY_MODEL_OWNED_BY",
    "PROVIDER_API_KEY", "PROXY_API_KEY", "MESSAGES_UPSTREAM_MODE",
    "ANTHROPIC_MODEL_MARKERS", "MAX_BODY_MB", "TARGET_CHAT_PATH", "TARGET_MESSAGES_PATH",
  ]) {
    if (!(key in env)) delete process.env[key];
  }
  return require("./main");
}

afterAll(() => {
  process.env = originalEnv;
});

describe("config", () => {
  test("uses documented defaults", () => {
    const { config } = loadMain({ PROXY_MODEL_CREATED: "1789100219" });
    expect(config.serverPort).toBe(8080);
    expect(config.targetBaseUrl).toBe("http://localhost:8000");
    expect(config.targetChatPath).toBe("/v1/chat/completions");
    expect(config.targetMessagesPath).toBe("/v1/messages");
    expect(config.proxyModelName).toBe("claude-opus-5");
    expect(config.targetModelName).toBe("nvidia/nemotron-3-super-120b-a12b");
    expect(config.proxyModelId).toBe("claude-opus-5");
    expect(config.proxyModelObject).toBe("model");
    expect(config.proxyModelCreated).toBe(1789100219);
    expect(config.proxyModelOwnedBy).toBe("cloud-proxy");
    expect(config.providerApiKey).toBe("");
    expect(config.proxyApiKey).toBe("");
    expect(config.messagesUpstreamMode).toBe("auto");
    expect(config.anthropicModelMarkers).toEqual([
      "opus", "sonnet", "haik", "fable", "anthropic", "claude-",
    ]);
  });

  test("reads every value from the environment", () => {
    const { config } = loadMain({
      SERVER_PORT: "9000",
      TARGET_BASE_URL: "https://custom.api.com",
      TARGET_CHAT_PATH: "/custom/chat",
      TARGET_MESSAGES_PATH: "/custom/messages",
      PROXY_MODEL_NAME: "custom-proxy",
      TARGET_MODEL_NAME: "custom-target",
      PROXY_MODEL_ID: "custom-id",
      PROXY_MODEL_OBJECT: "custom-object",
      PROXY_MODEL_CREATED: "9999999999",
      PROXY_MODEL_OWNED_BY: "custom-owner",
      PROVIDER_API_KEY: "upstream",
      PROXY_API_KEY: "downstream",
      MESSAGES_UPSTREAM_MODE: "PASSTHROUGH",
    });
    expect(config.serverPort).toBe(9000);
    expect(config.targetBaseUrl).toBe("https://custom.api.com");
    expect(config.targetChatPath).toBe("/custom/chat");
    expect(config.targetMessagesPath).toBe("/custom/messages");
    expect(config.proxyModelName).toBe("custom-proxy");
    expect(config.targetModelName).toBe("custom-target");
    expect(config.proxyModelId).toBe("custom-id");
    expect(config.proxyModelObject).toBe("custom-object");
    expect(config.proxyModelCreated).toBe(9999999999);
    expect(config.proxyModelOwnedBy).toBe("custom-owner");
    expect(config.providerApiKey).toBe("upstream");
    expect(config.proxyApiKey).toBe("downstream");
    // The mode is lowercased, so the env var is not case sensitive.
    expect(config.messagesUpstreamMode).toBe("passthrough");
  });

  test("trims trailing slashes from the base URL", () => {
    expect(loadMain({ TARGET_BASE_URL: "https://api.example.com///" }).config.targetBaseUrl).toBe(
      "https://api.example.com",
    );
  });

  test("custom markers replace the defaults", () => {
    const { config, isClaudeModelName } = loadMain({
      ANTHROPIC_MODEL_MARKERS: " NEMOTRON , mistral ,, ",
    });
    expect(config.anthropicModelMarkers).toEqual(["nemotron", "mistral"]);
    expect(isClaudeModelName("nvidia/nemotron-3")).toBe(true);
    expect(isClaudeModelName("claude-opus-4-5")).toBe(false);
  });
});

describe("parseRequestBody", () => {
  let parseRequestBody;
  beforeAll(() => {
    ({ parseRequestBody } = loadMain());
  });

  test("parses a JSON buffer", () => {
    expect(parseRequestBody({ body: Buffer.from('{"model":"m"}') })).toEqual({ body: { model: "m" } });
  });

  test("flags a malformed buffer instead of throwing", () => {
    expect(parseRequestBody({ body: Buffer.from("not json") })).toEqual({ invalid: true });
  });

  test("an empty buffer is no body", () => {
    expect(parseRequestBody({ body: Buffer.alloc(0) })).toEqual({ body: null });
  });

  test("body-parser's empty-object marker is no body", () => {
    // express.raw leaves {} behind when there was nothing to parse, e.g. on a GET.
    expect(parseRequestBody({ body: {} })).toEqual({ body: null });
  });

  test("a JSON value that is not an object is rejected", () => {
    // These APIs only ever take a JSON object.
    for (const raw of ['"a string"', "[1,2,3]", "null", "42", "true"]) {
      expect(parseRequestBody({ body: Buffer.from(raw) })).toEqual({ invalid: true });
    }
  });

  test("an already-parsed object is accepted", () => {
    expect(parseRequestBody({ body: { model: "m" } })).toEqual({ body: { model: "m" } });
  });

  test("a missing body is no body", () => {
    expect(parseRequestBody({})).toEqual({ body: null });
    expect(parseRequestBody(undefined)).toEqual({ body: null });
  });
});

describe("model masking", () => {
  let main;
  beforeAll(() => {
    main = loadMain({ PROXY_MODEL_NAME: "claude-opus-5", TARGET_MODEL_NAME: "upstream" });
  });

  test("shouldMaskModel covers the proxy name and any anthropic model", () => {
    expect(main.shouldMaskModel("claude-opus-5")).toBe(true);
    expect(main.shouldMaskModel("claude-sonnet-4-5")).toBe(true);
    expect(main.shouldMaskModel("CLAUDE-HAIKU-3")).toBe(true);
    expect(main.shouldMaskModel("gpt-4o")).toBe(false);
    expect(main.shouldMaskModel("")).toBe(false);
    expect(main.shouldMaskModel(undefined)).toBe(false);
  });

  test("modifyBodyForProxy swaps the model without mutating the input", () => {
    const original = { model: "claude-opus-5", temperature: 0.5 };
    const masked = main.modifyBodyForProxy(original);

    expect(masked.model).toBe("upstream");
    expect(masked.temperature).toBe(0.5);
    expect(original.model).toBe("claude-opus-5");
  });

  test("modifyBodyForProxy leaves foreign and absent models alone", () => {
    expect(main.modifyBodyForProxy({ model: "gpt-4o" }).model).toBe("gpt-4o");
    expect(main.modifyBodyForProxy({ input: "hi" })).toEqual({ input: "hi" });
    expect(main.modifyBodyForProxy(null)).toBeNull();
  });

  test("replaceModelInResponse masks every place the model appears", () => {
    const out = JSON.parse(
      main.replaceModelInResponse(
        JSON.stringify({
          model: "upstream",
          choices: [{ message: { model: "upstream", content: "hi" } }],
          data: [{ model: "upstream" }],
          message: { model: "upstream" },
        }),
      ),
    );
    expect(out.model).toBe("claude-opus-5");
    expect(out.choices[0].message.model).toBe("claude-opus-5");
    expect(out.data[0].model).toBe("claude-opus-5");
    // message_start events in a streamed anthropic response.
    expect(out.message.model).toBe("claude-opus-5");
  });

  test("replaceModelInResponse passes other models and non-JSON through", () => {
    expect(main.replaceModelInResponse('{"model":"other"}')).toBe('{"model":"other"}');
    for (const raw of ["not json", "", "[1,2,3]"]) {
      expect(main.replaceModelInResponse(raw)).toBe(raw);
    }
  });

  test("maskModelInSSELine only touches data frames", () => {
    const masked = main.maskModelInSSELine('data: {"model":"upstream","choices":[]}\n');
    expect(masked).toContain("claude-opus-5");
    expect(masked).not.toContain("upstream");
    expect(masked.endsWith("\n")).toBe(true);

    for (const line of [
      "data: [DONE]\n", "\n", "event: ping\n", "id: 1\n", ": comment\n",
      "data: \n", "data: not json\n",
    ]) {
      expect(main.maskModelInSSELine(line)).toBe(line);
    }
  });
});

describe("upstream headers", () => {
  test("hop-by-hop and host headers are dropped, the rest are relayed", () => {
    const { buildUpstreamHeaders } = loadMain();
    const headers = buildUpstreamHeaders(
      {
        headers: {
          host: "localhost:8080",
          connection: "keep-alive",
          "transfer-encoding": "chunked",
          "content-length": "123",
          "accept-encoding": "gzip",
          "anthropic-version": "2023-06-01",
          "user-agent": "claude-cli",
        },
      },
      false,
    );

    expect(headers.host).toBeUndefined();
    expect(headers.connection).toBeUndefined();
    expect(headers["transfer-encoding"]).toBeUndefined();
    expect(headers["content-length"]).toBeUndefined();
    expect(headers["accept-encoding"]).toBeUndefined();
    expect(headers["anthropic-version"]).toBe("2023-06-01");
    expect(headers["user-agent"]).toBe("claude-cli");
    expect(headers["content-type"]).toBe("application/json");
    expect(headers.accept).toBe("application/json");
  });

  test("a streaming request asks for an event stream", () => {
    const { buildUpstreamHeaders } = loadMain();
    expect(buildUpstreamHeaders({ headers: {} }, true).accept).toBe("text/event-stream");
  });

  test("the caller's own Accept header is preserved", () => {
    const { buildUpstreamHeaders } = loadMain();
    expect(buildUpstreamHeaders({ headers: { accept: "application/xml" } }, false).accept).toBe(
      "application/xml",
    );
  });

  test("the provider key replaces the caller's credential", () => {
    const { buildUpstreamHeaders } = loadMain({ PROVIDER_API_KEY: "upstream-key" });
    const headers = buildUpstreamHeaders(
      { headers: { authorization: "Bearer client-key", "x-api-key": "client-key" } },
      false,
    );
    expect(headers.authorization).toBe("Bearer upstream-key");
    expect(headers["x-api-key"]).toBeUndefined();
  });

  test("without a provider key nothing is added or stripped", () => {
    const { buildUpstreamHeaders } = loadMain();
    const headers = buildUpstreamHeaders({ headers: { authorization: "Bearer client-key" } }, false);
    expect(headers.authorization).toBe("Bearer client-key");
  });
});

describe("error shaping", () => {
  let main;
  beforeAll(() => {
    main = loadMain();
  });

  test("anthropicErrorType maps status codes", () => {
    const cases = {
      400: "invalid_request_error",
      401: "authentication_error",
      403: "permission_error",
      404: "not_found_error",
      413: "request_too_large",
      429: "rate_limit_error",
      500: "api_error",
      502: "api_error",
      529: "overloaded_error",
    };
    for (const [status, want] of Object.entries(cases)) {
      expect(main.anthropicErrorType(Number(status))).toBe(want);
    }
  });

  test("openAIErrorToAnthropic extracts the message from every shape", () => {
    const cases = [
      [400, '{"error":{"message":"bad model"}}', "bad model"],
      [500, '{"message":"boom"}', "boom"],
      [502, "upstream exploded", "upstream exploded"],
      [500, "", "Upstream returned 500"],
    ];
    for (const [status, body, wantMessage] of cases) {
      const out = JSON.parse(main.openAIErrorToAnthropic(status, body));
      expect(out.type).toBe("error");
      expect(out.error.message).toBe(wantMessage);
      expect(out.error.type).toBe(main.anthropicErrorType(status));
    }
  });
});
