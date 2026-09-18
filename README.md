<div align="center">

# Claude Reverse Proxy

**Talk to any OpenAI-compatible model as if it were Claude.**

A small local proxy that exposes the Anthropic Messages API and speaks OpenAI chat completions upstream — masking the model name in both directions, translating tool calls, images and streaming events on the way through.

Written three times over, identically, in **Go**, **Python** and **Node.js**.

[![CI](https://github.com/janishar/claude-reverse-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/janishar/claude-reverse-proxy/actions/workflows/ci.yml)
[![Go](https://img.shields.io/badge/Go-1.21%2B-00ADD8?logo=go&logoColor=white)](goproxy)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](pyproxy)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933?logo=node.js&logoColor=white)](nodeproxy)
[![Tests](https://img.shields.io/badge/tests-181%20passing-success)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

```
   Claude Code / Anthropic SDK                          NVIDIA NIM · vLLM · Ollama
   POST /v1/messages                                    POST /v1/chat/completions
   { "model": "claude-opus-5" }                         { "model": "nemotron-3-…" }
            │                                                        ▲
            │              ┌──────────────────────┐                  │
            └─────────────▶│  claude-reverse-proxy │──────────────────┘
                           │  mask · translate     │
            ◀──────────────│  stream · mask back   │◀─────────────────┐
   { "type": "message",    └──────────────────────┘        { "choices": [ … ] }
     "model": "claude-opus-5" }
```

Point Claude Code at it and your local model answers. Point an OpenAI SDK at it and nothing is translated at all — only the model name changes.

## Contents

- [Why](#why)
- [Quick start](#quick-start)
- [Use it with Claude Code](#use-it-with-claude-code)
- [How it works](#how-it-works)
- [Endpoints](#endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## Why

Anthropic clients speak `/v1/messages`. Almost every other inference provider speaks `/v1/chat/completions`. The formats differ in ways that matter — system prompts, content blocks, tool calls, streaming events — so you cannot simply rewrite the URL.

This proxy does the translation, in both directions, including streaming. It also **masks the model name**: clients ask for `claude-opus-5`, the upstream is asked for whatever you configured, and the answer comes back saying `claude-opus-5` again. A client that only accepts Claude model names never has to know.

Three separate implementations are kept byte-for-byte equivalent, enforced by a [parity check](#testing) that runs all three against one upstream and diffs every response.

## Quick start

```bash
git clone https://github.com/janishar/claude-reverse-proxy.git
cd claude-reverse-proxy
make setup          # installs Node deps and creates .env from the template
```

Edit `.env`:

```bash
TARGET_BASE_URL=https://integrate.api.nvidia.com
TARGET_MODEL_NAME=nvidia/nemotron-3-super-120b-a12b
PROVIDER_API_KEY=nvapi-your-key-here
PROXY_API_KEY=pick-any-local-secret
```

Run whichever runtime you prefer — they are interchangeable:

```bash
make run IMPL=go      # or: make run IMPL=py
                      #     make run IMPL=node
```

Check it:

```bash
make smoke            # model list, messages, streaming, chat completions
```

<details>
<summary>Without <code>make</code></summary>

```bash
cp .env.example .env            # then edit it
./scripts/run.sh go             # or py | node
./scripts/smoke.sh
```

Or run a proxy directly, configured entirely by environment variables:

```bash
cd goproxy && go build && ./goproxy
cd pyproxy && python3 main.py          # stdlib only, nothing to install
cd nodeproxy && npm install && npm start
```
</details>

## Use it with Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=pick-any-local-secret   # matches PROXY_API_KEY
export ANTHROPIC_MODEL=claude-opus-5                # matches PROXY_MODEL_NAME
claude
```

Tool use, images, system prompts and streaming all survive the round trip.

## How it works

### 1. Model masking, on every route, in both directions

| Direction | Rule |
|---|---|
| **Request** | `model` becomes `TARGET_MODEL_NAME` when it equals `PROXY_MODEL_NAME` **or** looks like any Anthropic model (`claude-sonnet-4-5`, `claude-opus-4-1`, …) |
| **Response** | `TARGET_MODEL_NAME` becomes `PROXY_MODEL_NAME` in `model`, `choices[].message.model`, `data[].model`, and inside streamed `message_start` events |

The caller only ever sees the model it asked for. `GET /v1/models` advertises that same name without touching the upstream.

### 2. The `/v1/messages` relay

```
                            model is Anthropic?
 POST /v1/messages ─────────────────┬───────────────▶ yes ──▶ translate ──▶ POST {TARGET_CHAT_PATH}
   (Anthropic format)               │                                              │
                                    │                       Anthropic format ◀── translate back
                                    │
                                    └──▶ no ──▶ POST {TARGET_MESSAGES_PATH} unchanged
                                                     │
                                                     └── 404? ──▶ translate ──▶ POST {TARGET_CHAT_PATH}
```

**Anthropic → OpenAI**

| Anthropic | OpenAI |
|---|---|
| `system` | a leading `{"role": "system"}` message |
| `content` blocks | flattened text, or `image_url` parts for images |
| `tool_use` blocks | `tool_calls` on the assistant message |
| `tool_result` blocks | `{"role": "tool", "tool_call_id": …}` messages |
| `tools[].input_schema` | `tools[].function.parameters` |
| `tool_choice` `auto`/`any`/`none`/`tool` | `auto`/`required`/`none`/`{"type":"function",…}` |
| `stop_sequences` | `stop` |
| `metadata.user_id` | `user` |

`max_tokens`, `temperature`, `top_p`, `stream`, `seed`, `n`, `frequency_penalty`, `presence_penalty` and `response_format` pass through unchanged.

**OpenAI → Anthropic**

| OpenAI | Anthropic |
|---|---|
| `choices[0].message.content` | a `text` content block |
| `choices[0].message.tool_calls` | `tool_use` content blocks |
| `finish_reason` `stop`/`length`/`tool_calls` | `stop_reason` `end_turn`/`max_tokens`/`tool_use` |
| `usage.prompt_tokens` / `completion_tokens` | `usage.input_tokens` / `output_tokens` |
| any error body | `{"type": "error", "error": {"type": …, "message": …}}` |

**Streaming** is translated chunk by chunk into the Anthropic event sequence:

```
message_start → content_block_start → content_block_delta … → content_block_stop → message_delta → message_stop
```

Text arrives as `text_delta`; streamed tool arguments arrive as `input_json_delta`.

### 3. Everything else is a plain forward

Requests to the OpenAI routes are **not** translated. Shapes are preserved in both directions, SSE streams are piped through as they arrive, and only the model name is touched — exactly what an OpenAI SDK expects.

## Endpoints

| Endpoint | Behaviour |
|---|---|
| `GET /v1/models` | Served locally: the one configured model. Never calls the upstream |
| `POST /v1/messages` | Anthropic Messages API — relayed through chat completions |
| `POST /v1/chat/completions` | Forwarded; model masked; streams piped |
| `POST /v1/responses` | Forwarded; model masked |
| `POST /v1/embeddings` | Forwarded; model masked |
| `POST /v1/moderations` | Forwarded; model masked |
| `POST /v1/completions` | Forwarded; model masked |
| `/v1/*` | Catch-all; inbound path and query preserved |

<details>
<summary><b>Example requests</b></summary>

**Messages API** — the upstream never sees Anthropic format:

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $PROXY_API_KEY" \
  -d '{
    "model": "claude-opus-5",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Which is larger, 9.11 or 9.8?"}]
  }'
```

**Streaming:**

```bash
curl -N http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $PROXY_API_KEY" \
  -d '{"model":"claude-opus-5","max_tokens":256,"stream":true,
       "messages":[{"role":"user","content":"Count to five."}]}'
```

**Tool use:**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $PROXY_API_KEY" \
  -d '{
    "model": "claude-opus-5",
    "max_tokens": 1024,
    "tools": [{
      "name": "get_weather",
      "description": "Get the current weather in a city",
      "input_schema": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
    }],
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
  }'
```

**OpenAI chat completions** — forwarded, not translated:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -d '{"model":"claude-opus-5","messages":[{"role":"user","content":"Hello"}]}'
```
</details>

## Configuration

Every implementation reads the same environment variables. `make setup` writes a starter `.env` from [`.env.example`](.env.example).

### Core

| Variable | Default | Description |
|---|---|---|
| `SERVER_PORT` | `8080` | Port the proxy listens on |
| `TARGET_BASE_URL` | `http://localhost:8000` | Upstream API base URL |
| `PROXY_MODEL_NAME` | `claude-opus-5` | Model name clients use |
| `TARGET_MODEL_NAME` | `nvidia/nemotron-3-super-120b-a12b` | Model name sent upstream |
| `PROVIDER_API_KEY` | *(empty)* | Upstream credential, sent as `Authorization: Bearer …`. Leave empty for keyless local servers |
| `PROXY_API_KEY` | *(empty)* | If set, incoming requests must present it as `Authorization: Bearer …` or `x-api-key` |

### Model list

| Variable | Default | Description |
|---|---|---|
| `PROXY_MODEL_ID` | `claude-opus-5` | `id` returned by `/v1/models` |
| `PROXY_MODEL_OBJECT` | `model` | `object` field |
| `PROXY_MODEL_CREATED` | *now* | `created` timestamp |
| `PROXY_MODEL_OWNED_BY` | `cloud-proxy` | `owned_by` field |

### Upstream paths

| Variable | Default |
|---|---|
| `TARGET_CHAT_PATH` | `/v1/chat/completions` — the relay target |
| `TARGET_MESSAGES_PATH` | `/v1/messages` — used only when a request is not translated |
| `TARGET_MODELS_PATH` | `/v1/models` |
| `TARGET_RESPONSES_PATH` | `/v1/responses` |
| `TARGET_EMBEDDINGS_PATH` | `/v1/embeddings` |
| `TARGET_MODERATIONS_PATH` | `/v1/moderations` |
| `TARGET_COMPLETIONS_PATH` | `/v1/completions` |

### Relay behaviour

| Variable | Default | Description |
|---|---|---|
| `MESSAGES_UPSTREAM_MODE` | `auto` | `auto`, `translate` or `passthrough` |
| `ANTHROPIC_MODEL_MARKERS` | `opus,sonnet,haik,fable,anthropic,claude-` | Substrings that mark a model name as Anthropic |
| `MAX_BODY_MB` | `50` | Maximum request body size; larger requests get a `413` |

**`MESSAGES_UPSTREAM_MODE`**

| Value | `/v1/messages` behaviour |
|---|---|
| `auto` *(default)* | Translate when the model is Anthropic; otherwise try the upstream Messages API and fall back to translation on a `404` |
| `translate` | Always translate, whatever the model |
| `passthrough` | Never translate — the upstream really does implement the Anthropic Messages API |

**`ANTHROPIC_MODEL_MARKERS`** decides both what gets relayed and what gets masked. Set it when your model names do not look like Claude:

```bash
ANTHROPIC_MODEL_MARKERS=my-model,internal-llm
```

### Authentication

- **Incoming** — when `PROXY_API_KEY` is set, requests must carry it as `Authorization: Bearer <key>` (OpenAI style) or `x-api-key: <key>` (Anthropic style). Otherwise `401`.
- **Outgoing** — when `PROVIDER_API_KEY` is set it *replaces* whatever credential the caller sent, and the caller's `x-api-key` is stripped, so a local token is never leaked upstream. When unset, no credential is added — which is what a local model server wants.

<details>
<summary><b>Configuration recipes</b></summary>

**NVIDIA NIM**
```bash
TARGET_BASE_URL=https://integrate.api.nvidia.com
TARGET_MODEL_NAME=nvidia/nemotron-3-super-120b-a12b
PROVIDER_API_KEY=nvapi-…
```

**Local Ollama, no key**
```bash
TARGET_BASE_URL=http://localhost:11434
TARGET_MODEL_NAME=llama3
```

**vLLM**
```bash
TARGET_BASE_URL=http://localhost:8000
TARGET_MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
```

**An upstream that already speaks Anthropic**
```bash
TARGET_BASE_URL=https://api.anthropic.com
MESSAGES_UPSTREAM_MODE=passthrough
```
</details>

## Development

### Make targets

```bash
make                 # list every target
make setup           # install deps, create .env
make run IMPL=go     # run one implementation (go | py | node)
make test            # all three suites plus the parity check
make test-go         # one suite
make cover           # Go coverage summary
make parity          # prove all three behave identically
make smoke           # curl a running proxy
make lint            # shellcheck the scripts, as CI does
make fmt vet clean
```

### Scripts

| Script | Purpose |
|---|---|
| [`scripts/run.sh`](scripts/run.sh) | Start one implementation with `.env` loaded |
| [`scripts/test.sh`](scripts/test.sh) | Run one suite or all of them |
| [`scripts/smoke.sh`](scripts/smoke.sh) | Exercise a running proxy with curl |
| [`scripts/parity_check.py`](scripts/parity_check.py) | Boot all three against one upstream and diff every response |

All of them read `.env`; anything already exported in your shell wins, so one-offs work:

```bash
SERVER_PORT=9000 ./scripts/run.sh node
```

### Debugging in VS Code

`.vscode/launch.json` ships ready to use — press <kbd>F5</kbd> and pick a configuration. Every one loads `.env`, so set your keys there first.

| Group | Configurations |
|---|---|
| **Run** | `Go: proxy` · `Python: proxy` · `Node: proxy` |
| **Test** | `Go: all tests` · `Go: test at cursor` · `Python: all tests` · `Python: current test file` · `Node: all tests` · `Node: current test file` |
| **Tools** | `Parity: all three` · `Attach: Node inspector (9229)` |

Breakpoints work in all three. `Go: test at cursor` runs whatever test name you have selected in the editor.

Tasks (<kbd>⇧⌘B</kbd> / <kbd>Ctrl+Shift+B</kbd>) cover `build: go`, `test: all`, per-language test runs, `parity check`, `smoke test` and `setup`.

Recommended extensions are listed in `.vscode/extensions.json` — VS Code offers to install them on first open.

## Testing

```bash
make test
```

| Suite | Tests | What it covers |
|---|---:|---|
| **goproxy** | 62 | Config, model list, masking, forwarding, streaming, auth, translation |
| **pyproxy** | 60 | Same ground, plus the relay end to end over real sockets |
| **nodeproxy** | 59 | Units, translation, and end-to-end against a real upstream server |
| **parity** | 10 checks | All three booted against one upstream, every response diffed |

Go statement coverage is **90.2%** (`make cover`).

Each language keeps three suites with the same split:

- **units / main** — configuration, helpers, model masking, error shaping
- **claude_route** — the Anthropic ⇄ OpenAI translation itself
- **forward** — the proxy driven over HTTP against a real upstream

The parity check is the interesting one. It starts all three proxies against a single fake upstream, sends an identical set of requests — tool calls, images, streaming, auth failures, malformed bodies — and diffs both the answers and the exact bytes each proxy sent upstream. It has caught real drift: JSON spacing differences and disagreements over how to handle a non-object request body.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request:

| Job | Matrix | What it does |
|---|---|---|
| **Go** | 1.21, stable | `gofmt` check, `go vet`, build, `go test -race`, coverage summary |
| **Python** | 3.9 – 3.13 | Byte-compile, then the full suite |
| **Node** | 18, 20, 22 | `npm ci`, then the full suite |
| **Parity** | — | All three built and diffed against one upstream |
| **Shell scripts** | — | `bash -n`, executable bits, `shellcheck` |
| **macOS** | — | All three suites on `macos-latest` |

A final `CI` job gates on every other one, so branch protection only needs that single required check even as the matrix grows.

## Project structure

```
claude-reverse-proxy/
├── goproxy/                   # Go implementation
│   ├── main.go
│   ├── main_test.go           # config, model list, helpers
│   ├── forward_test.go        # forwarding, masking, streaming, auth
│   └── claude_route_test.go   # Anthropic ⇄ OpenAI translation
├── pyproxy/                   # Python implementation (stdlib only)
│   ├── main.py
│   ├── test_main.py           # config, routing, auth
│   └── test_claude_route.py   # translation and relay
├── nodeproxy/                 # Node.js implementation
│   ├── main.js
│   ├── units.test.js          # config, helpers, masking
│   ├── claude_route.test.js   # translation
│   └── forward.test.js        # end to end
├── scripts/                   # run, test, smoke, parity
├── .github/workflows/ci.yml   # test matrix, parity and lint on every push
├── .vscode/                   # launch configs, tasks, recommended extensions
├── Makefile
├── .env.example
└── .gitignore
```

## Requirements

| Implementation | Needs | Dependencies |
|---|---|---|
| **goproxy** | Go 1.21+ | none — stdlib only |
| **pyproxy** | Python 3.9+ | none — stdlib only |
| **nodeproxy** | Node.js 18+ | `express`, `node-fetch` |

Nothing is required at runtime beyond the language itself for Go and Python.

## FAQ

<details>
<summary><b>Which implementation should I run?</b></summary>

Any of them — the parity check exists to keep that true. Go builds to a single binary, Python needs no dependencies at all, Node fits if you are already in that ecosystem.
</details>

<details>
<summary><b>Does streaming work?</b></summary>

Yes, on both routes. `/v1/messages` streams are translated into Anthropic events chunk by chunk. OpenAI-route streams are piped straight through with only the model name rewritten inside each `data:` frame.
</details>

<details>
<summary><b>My model is not named like Claude and nothing is being relayed.</b></summary>

The relay triggers on the model name. Either set `PROXY_MODEL_NAME` to the name your client sends, add a marker with `ANTHROPIC_MODEL_MARKERS`, or force it with `MESSAGES_UPSTREAM_MODE=translate`.
</details>

<details>
<summary><b>Can I run it without an upstream API key?</b></summary>

Yes. Leave `PROVIDER_API_KEY` empty and no credential is added — the right behaviour for Ollama, vLLM or llama.cpp on localhost.
</details>

<details>
<summary><b>Is this safe to expose publicly?</b></summary>

It is built for localhost. There is no TLS, no rate limiting and no per-user accounting. If you expose it, put it behind a reverse proxy that provides those, and always set `PROXY_API_KEY`.
</details>

## Contributing

The one rule that matters: **a change to one implementation is a change to all three.**

1. Make the change in Go, Python and Node.
2. Add or update tests in each language's suite.
3. Run `make test` — all three suites plus the parity check must pass.

```bash
make fmt && make test
```

CI runs the same suites across the Go, Python and Node version matrices, plus the parity check and `shellcheck`, so `make test` passing locally is a good predictor of a green build.

If you add a config option, it needs a default in all three, a row in the [Configuration](#configuration) table, and a line in `.env.example`.

## License

[MIT](LICENSE)
