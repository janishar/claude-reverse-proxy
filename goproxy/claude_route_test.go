package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsClaudeModelName(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"claude-opus-4-5", true},
		{"CLAUDE-SONNET-4-20250514", true},
		{"Claude-Sonnet-4", true},
		{"claude-haiku-3-5", true},
		{"claude-haiki-1", true},
		{"anthropic.claude-fable-1", true},
		{"gpt-4o", false},
		{"nvidia/nemotron-3-super-120b-a12b", false},
		{"claude", false},
		{"", false},
	}
	for _, c := range cases {
		if got := isClaudeModelName(c.model); got != c.want {
			t.Errorf("isClaudeModelName(%q) = %v, want %v", c.model, got, c.want)
		}
	}
}

func TestShouldRouteClaudeToChat(t *testing.T) {
	autoCfg := Config{ProxyModelName: "claude-opus-5"}
	customProxyCfg := Config{ProxyModelName: "my-model"}
	translateCfg := Config{ProxyModelName: "claude-opus-5", MessagesUpstreamMode: modeTranslate}
	passthroughCfg := Config{ProxyModelName: "claude-opus-5", MessagesUpstreamMode: modePassthrough}

	cases := []struct {
		name  string
		cfg   Config
		model string
		want  bool
	}{
		{"auto/claude", autoCfg, "claude-opus-4-5", true},
		{"auto/claude upper", autoCfg, "CLAUDE-SONNET-4", true},
		{"auto/haiku", autoCfg, "claude-haiku-3-5", true},
		{"auto/proxy model name", customProxyCfg, "my-model", true},
		{"auto/openai", autoCfg, "gpt-4o", false},
		{"auto/empty", autoCfg, "", false},
		{"auto/upstream model", autoCfg, "nvidia/nemotron-3-super-120b-a12b", false},
		{"translate/openai", translateCfg, "gpt-4o", true},
		{"passthrough/claude", passthroughCfg, "claude-opus-4-5", false},
	}
	for _, c := range cases {
		if got := shouldRouteClaudeToChat(c.cfg, c.model); got != c.want {
			t.Errorf("%s: shouldRouteClaudeToChat(%q) = %v, want %v", c.name, c.model, got, c.want)
		}
	}
}

func TestAnthropicToOpenAIRequest(t *testing.T) {
	cfg := Config{TargetModelName: "target-model", ProxyModelName: "claude-opus-5"}
	anthropicReq := map[string]interface{}{
		"model":          "claude-opus-4-5",
		"max_tokens":     1024,
		"temperature":    0.7,
		"top_p":          0.9,
		"stop_sequences": []interface{}{"\n\n", "HALT"},
		"system":         "You are a helpful assistant.",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
			map[string]interface{}{
				"role": "assistant",
				"content": []interface{}{
					map[string]interface{}{"type": "text", "text": "Let me check that."},
					map[string]interface{}{
						"type":  "tool_use",
						"id":    "toolu_01",
						"name":  "get_weather",
						"input": map[string]interface{}{"city": "Paris"},
					},
				},
			},
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "toolu_01",
						"content":     "Sunny, 25C",
					},
				},
			},
			map[string]interface{}{"role": "user", "content": "Thanks"},
		},
		"tools": []interface{}{
			map[string]interface{}{
				"name":         "get_weather",
				"description":  "Get the weather",
				"input_schema": map[string]interface{}{"type": "object", "properties": map[string]interface{}{"city": map[string]interface{}{"type": "string"}}},
			},
		},
		"tool_choice": map[string]interface{}{"type": "auto"},
	}

	out := anthropicToOpenAIRequest(cfg, anthropicReq)

	if out["model"] != "target-model" {
		t.Errorf("model = %v, want target-model", out["model"])
	}
	if toInt64(out["max_tokens"]) != 1024 {
		t.Errorf("max_tokens = %v, want 1024", out["max_tokens"])
	}
	if out["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", out["temperature"])
	}
	if out["top_p"] != 0.9 {
		t.Errorf("top_p = %v, want 0.9", out["top_p"])
	}
	if stop, ok := out["stop"].([]interface{}); !ok || len(stop) != 2 || stop[0] != "\n\n" {
		t.Errorf("stop = %v, want stop_sequences mapped", out["stop"])
	}
	if _, hasTopK := out["top_k"]; anthropicReq["top_k"] != nil && hasTopK {
		t.Error("top_k should not be forwarded")
	}

	msgs, ok := out["messages"].([]interface{})
	if !ok || len(msgs) != 5 {
		t.Fatalf("messages length = %d, want 5 (system + 4)", len(msgs))
	}

	sysMsg, _ := msgs[0].(map[string]interface{})
	if sysMsg["role"] != "system" || sysMsg["content"] != "You are a helpful assistant." {
		t.Errorf("system message = %v", sysMsg)
	}

	assistantMsg, _ := msgs[2].(map[string]interface{})
	if assistantMsg["role"] != "assistant" || assistantMsg["content"] != "Let me check that." {
		t.Errorf("assistant message = %v", assistantMsg)
	}
	toolCalls, ok := assistantMsg["tool_calls"].([]interface{})
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %v, want 1 entry", assistantMsg["tool_calls"])
	}
	tc, _ := toolCalls[0].(map[string]interface{})
	if tc["id"] != "toolu_01" || tc["type"] != "function" {
		t.Errorf("tool call = %v", tc)
	}
	fn, _ := tc["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("tool call function = %v", fn)
	}
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil || args["city"] != "Paris" {
		t.Errorf("tool call arguments = %v (err %v)", fn["arguments"], err)
	}

	toolMsg, _ := msgs[3].(map[string]interface{})
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "toolu_01" || toolMsg["content"] != "Sunny, 25C" {
		t.Errorf("tool message = %v", toolMsg)
	}

	lastMsg, _ := msgs[4].(map[string]interface{})
	if lastMsg["role"] != "user" || lastMsg["content"] != "Thanks" {
		t.Errorf("last message = %v", lastMsg)
	}

	tools, ok := out["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want 1 entry", out["tools"])
	}
	oaiTool, _ := tools[0].(map[string]interface{})
	if oaiTool["type"] != "function" {
		t.Errorf("tool type = %v", oaiTool["type"])
	}
	oaiFn, _ := oaiTool["function"].(map[string]interface{})
	if oaiFn["name"] != "get_weather" || oaiFn["parameters"] == nil {
		t.Errorf("tool function = %v", oaiFn)
	}

	if out["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v, want auto", out["tool_choice"])
	}
}

func TestAnthropicToOpenAIRequestToolChoiceVariants(t *testing.T) {
	cfg := Config{TargetModelName: "target-model"}

	out := anthropicToOpenAIRequest(cfg, map[string]interface{}{
		"messages":    []interface{}{},
		"tool_choice": map[string]interface{}{"type": "any"},
	})
	if out["tool_choice"] != "required" {
		t.Errorf("tool_choice any = %v, want required", out["tool_choice"])
	}

	out = anthropicToOpenAIRequest(cfg, map[string]interface{}{
		"messages":    []interface{}{},
		"tool_choice": map[string]interface{}{"type": "tool", "name": "get_weather"},
	})
	tc, _ := out["tool_choice"].(map[string]interface{})
	fn, _ := tc["function"].(map[string]interface{})
	if tc["type"] != "function" || fn["name"] != "get_weather" {
		t.Errorf("tool_choice tool = %v", out["tool_choice"])
	}
}

func TestAnthropicToOpenAIRequestImages(t *testing.T) {
	cfg := Config{TargetModelName: "target-model"}
	anthropicReq := map[string]interface{}{
		"model": "claude-opus-4-5",
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "text", "text": "What is in this image?"},
					map[string]interface{}{
						"type": "image",
						"source": map[string]interface{}{
							"type":       "base64",
							"media_type": "image/png",
							"data":       "iVBORw0KGgo=",
						},
					},
					map[string]interface{}{
						"type": "image",
						"source": map[string]interface{}{
							"type": "url",
							"url":  "https://example.com/cat.png",
						},
					},
				},
			},
		},
	}

	out := anthropicToOpenAIRequest(cfg, anthropicReq)
	msgs := out["messages"].([]interface{})
	userMsg, _ := msgs[0].(map[string]interface{})
	parts, ok := userMsg["content"].([]interface{})
	if !ok || len(parts) != 3 {
		t.Fatalf("content parts = %v, want 3", userMsg["content"])
	}

	b64, _ := parts[0].(map[string]interface{})
	b64URL, _ := b64["image_url"].(map[string]interface{})
	if b64["type"] != "image_url" || b64URL["url"] != "data:image/png;base64,iVBORw0KGgo=" {
		t.Errorf("base64 image part = %v", b64)
	}

	url, _ := parts[1].(map[string]interface{})
	urlMap, _ := url["image_url"].(map[string]interface{})
	if urlMap["url"] != "https://example.com/cat.png" {
		t.Errorf("url image part = %v", url)
	}

	text, _ := parts[2].(map[string]interface{})
	if text["type"] != "text" || text["text"] != "What is in this image?" {
		t.Errorf("text part = %v", text)
	}
}

func TestOpenAIToAnthropicResponse(t *testing.T) {
	cfg := Config{TargetModelName: "target-model", ProxyModelName: "claude-opus-5"}
	openAIResp := map[string]interface{}{
		"id":     "chatcmpl-abc123",
		"object": "chat.completion",
		"choices": []interface{}{
			map[string]interface{}{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "Hello there!"},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     10,
			"completion_tokens": 7,
			"total_tokens":      17,
		},
	}

	out := openAIToAnthropicResponse(cfg, openAIResp)

	if out["id"] != "msg_abc123" {
		t.Errorf("id = %v, want msg_abc123", out["id"])
	}
	if out["type"] != "message" || out["role"] != "assistant" {
		t.Errorf("type/role = %v/%v", out["type"], out["role"])
	}
	if out["model"] != "claude-opus-5" {
		t.Errorf("model = %v, want claude-opus-5", out["model"])
	}
	content, _ := out["content"].([]interface{})
	if len(content) != 1 {
		t.Fatalf("content = %v, want 1 block", out["content"])
	}
	block, _ := content[0].(map[string]interface{})
	if block["type"] != "text" || block["text"] != "Hello there!" {
		t.Errorf("content block = %v", block)
	}
	if out["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", out["stop_reason"])
	}
	usage, _ := out["usage"].(map[string]interface{})
	if usage["input_tokens"] != int64(10) || usage["output_tokens"] != int64(7) {
		t.Errorf("usage = %v", usage)
	}
}

func TestOpenAIToAnthropicResponseToolCalls(t *testing.T) {
	cfg := Config{TargetModelName: "target-model", ProxyModelName: "claude-opus-5"}
	openAIResp := map[string]interface{}{
		"id": "chatcmpl-xyz",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Let me look that up.",
					"tool_calls": []interface{}{
						map[string]interface{}{
							"id":   "call_123",
							"type": "function",
							"function": map[string]interface{}{
								"name":      "get_weather",
								"arguments": `{"city":"Paris"}`,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
	}

	out := openAIToAnthropicResponse(cfg, openAIResp)
	content, _ := out["content"].([]interface{})
	if len(content) != 2 {
		t.Fatalf("content = %v, want 2 blocks", out["content"])
	}
	if content[0].(map[string]interface{})["type"] != "text" {
		t.Errorf("block 0 = %v, want text", content[0])
	}
	toolUse, _ := content[1].(map[string]interface{})
	if toolUse["type"] != "tool_use" || toolUse["id"] != "call_123" || toolUse["name"] != "get_weather" {
		t.Errorf("tool_use block = %v", toolUse)
	}
	input, _ := toolUse["input"].(map[string]interface{})
	if input["city"] != "Paris" {
		t.Errorf("tool_use input = %v", toolUse["input"])
	}
	if out["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", out["stop_reason"])
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"stop":           "end_turn",
		"length":         "max_tokens",
		"tool_calls":     "tool_use",
		"content_filter": "end_turn",
		"unknown":        "end_turn",
	}
	for in, want := range cases {
		if got := mapFinishReason(in); got != want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func parseSSEEvents(t *testing.T, raw string) []map[string]interface{} {
	t.Helper()
	var events []map[string]interface{}
	for _, block := range strings.Split(raw, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var eventType string
		var dataLines []string
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(line, "event: ") {
				eventType = strings.TrimPrefix(line, "event: ")
			} else if strings.HasPrefix(line, "data: ") {
				dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
			}
		}
		if len(dataLines) == 0 {
			continue
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(strings.Join(dataLines, "\n")), &m); err != nil {
			t.Fatalf("invalid event data %q: %v", strings.Join(dataLines, "\n"), err)
		}
		m["__event"] = eventType
		events = append(events, m)
	}
	return events
}

func TestTranslateOpenAIStreamToAnthropicText(t *testing.T) {
	cfg := Config{ProxyModelName: "claude-opus-5"}
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	rec := httptest.NewRecorder()
	translateOpenAIStreamToAnthropic(cfg, strings.NewReader(sse), rec)

	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	events := parseSSEEvents(t, rec.Body.String())
	var eventTypes []string
	for _, e := range events {
		eventTypes = append(eventTypes, e["__event"].(string))
	}
	want := []string{"message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if len(eventTypes) != len(want) {
		t.Fatalf("events = %v, want %v", eventTypes, want)
	}
	for i := range want {
		if eventTypes[i] != want[i] {
			t.Fatalf("events = %v, want %v", eventTypes, want)
		}
	}

	startMsg, _ := events[0]["message"].(map[string]interface{})
	if startMsg["id"] != "msg_xyz" || startMsg["model"] != "claude-opus-5" || startMsg["role"] != "assistant" {
		t.Errorf("message_start = %v", startMsg)
	}

	blockStart, _ := events[1]["content_block"].(map[string]interface{})
	if blockStart["type"] != "text" {
		t.Errorf("content_block_start = %v", blockStart)
	}

	var text string
	for _, e := range events {
		if e["__event"] == "content_block_delta" {
			delta, _ := e["delta"].(map[string]interface{})
			if delta["type"] == "text_delta" {
				text += delta["text"].(string)
			}
		}
	}
	if text != "Hello" {
		t.Errorf("streamed text = %q, want %q", text, "Hello")
	}

	delta, _ := events[5]["delta"].(map[string]interface{})
	if delta["stop_reason"] != "end_turn" {
		t.Errorf("message_delta stop_reason = %v", delta["stop_reason"])
	}
	deltaUsage, _ := events[5]["usage"].(map[string]interface{})
	if deltaUsage["output_tokens"] != float64(2) {
		t.Errorf("message_delta usage = %v", deltaUsage)
	}
}

func TestTranslateOpenAIStreamToAnthropicToolCalls(t *testing.T) {
	cfg := Config{ProxyModelName: "claude-opus-5"}
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"role":"assistant","content":"Checking"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	rec := httptest.NewRecorder()
	translateOpenAIStreamToAnthropic(cfg, strings.NewReader(sse), rec)

	events := parseSSEEvents(t, rec.Body.String())
	var eventTypes []string
	for _, e := range events {
		eventTypes = append(eventTypes, e["__event"].(string))
	}
	want := []string{"message_start", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if fmt.Sprint(eventTypes) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", eventTypes, want)
	}

	textBlockStart, _ := events[1]["content_block"].(map[string]interface{})
	if textBlockStart["type"] != "text" {
		t.Errorf("block 0 = %v, want text", textBlockStart)
	}
	toolBlockStart, _ := events[4]["content_block"].(map[string]interface{})
	if toolBlockStart["type"] != "tool_use" || toolBlockStart["id"] != "call_9" || toolBlockStart["name"] != "get_weather" {
		t.Errorf("block 1 = %v, want tool_use", toolBlockStart)
	}

	var partialJSON string
	for _, e := range events {
		if e["__event"] == "content_block_delta" && e["index"] == float64(1) {
			delta, _ := e["delta"].(map[string]interface{})
			if delta["type"] == "input_json_delta" {
				partialJSON += delta["partial_json"].(string)
			}
		}
	}
	var input map[string]interface{}
	if err := json.Unmarshal([]byte(partialJSON), &input); err != nil {
		t.Fatalf("partial JSON %q is invalid: %v", partialJSON, err)
	}
	if input["city"] != "Paris" {
		t.Errorf("tool input = %v", input)
	}

	delta, _ := events[8]["delta"].(map[string]interface{})
	if delta["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", delta["stop_reason"])
	}
}

type recordingUpstream struct {
	mux          *http.ServeMux
	server       *httptest.Server
	receivedPath string
	receivedBody map[string]interface{}
}

func (u *recordingUpstream) close() {
	u.server.Close()
}

func newRecordingUpstream() *recordingUpstream {
	u := &recordingUpstream{mux: http.NewServeMux()}
	u.server = httptest.NewServer(u.mux)
	return u
}

func (u *recordingUpstream) recordAndRespond(status int, payload string) {
	u.mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		u.receivedPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&u.receivedBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, payload)
	})
}

func (u *recordingUpstream) recordAndStream(payload string) {
	u.mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		u.receivedPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&u.receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, payload)
	})
}

func postMessagesRequest(handler http.HandlerFunc, body map[string]interface{}) *httptest.ResponseRecorder {
	payload, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec
}

func TestMessagesRoutesClaudeToChatCompletions(t *testing.T) {
	u := newRecordingUpstream()
	defer u.close()
	u.recordAndRespond(http.StatusOK, `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi from upstream"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	cfg := Config{
		TargetBaseURL:      u.server.URL,
		TargetChatPath:     "/v1/chat/completions",
		TargetMessagesPath: "/v1/messages",
		TargetModelName:    "test-target",
		ProxyModelName:     "claude-opus-5",
	}
	handler := forwardToUpstream(cfg, cfg.TargetMessagesPath)

	rec := postMessagesRequest(handler, map[string]interface{}{
		"model":      "claude-opus-4-5",
		"max_tokens": 100,
		"system":     "You are helpful.",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}

	if u.receivedPath != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", u.receivedPath)
	}
	if u.receivedBody["model"] != "test-target" {
		t.Errorf("upstream model = %v, want test-target", u.receivedBody["model"])
	}
	msgs, _ := u.receivedBody["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("upstream messages = %v, want 2", u.receivedBody["messages"])
	}
	sysMsg, _ := msgs[0].(map[string]interface{})
	if sysMsg["role"] != "system" || sysMsg["content"] != "You are helpful." {
		t.Errorf("upstream system message = %v", sysMsg)
	}
	if u.receivedBody["max_tokens"] != float64(100) {
		t.Errorf("upstream max_tokens = %v", u.receivedBody["max_tokens"])
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if resp["type"] != "message" || resp["role"] != "assistant" {
		t.Errorf("response type/role = %v/%v", resp["type"], resp["role"])
	}
	if resp["model"] != "claude-opus-5" {
		t.Errorf("response model = %v, want claude-opus-5", resp["model"])
	}
	if resp["id"] != "msg_abc123" {
		t.Errorf("response id = %v, want msg_abc123", resp["id"])
	}
	content, _ := resp["content"].([]interface{})
	if len(content) != 1 || content[0].(map[string]interface{})["text"] != "Hi from upstream" {
		t.Errorf("response content = %v", resp["content"])
	}
	if resp["stop_reason"] != "end_turn" {
		t.Errorf("response stop_reason = %v", resp["stop_reason"])
	}
	usage, _ := resp["usage"].(map[string]interface{})
	if usage["input_tokens"] != float64(10) || usage["output_tokens"] != float64(5) {
		t.Errorf("response usage = %v", usage)
	}
}

func TestMessagesRoutesClaudeStreamingToChatCompletions(t *testing.T) {
	u := newRecordingUpstream()
	defer u.close()
	u.recordAndStream(strings.Join([]string{
		`data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n")

	cfg := Config{
		TargetBaseURL:      u.server.URL,
		TargetChatPath:     "/v1/chat/completions",
		TargetMessagesPath: "/v1/messages",
		TargetModelName:    "test-target",
		ProxyModelName:     "claude-opus-5",
	}
	handler := forwardToUpstream(cfg, cfg.TargetMessagesPath)

	rec := postMessagesRequest(handler, map[string]interface{}{
		"model":  "claude-sonnet-4-5",
		"stream": true,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if u.receivedPath != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", u.receivedPath)
	}
	if u.receivedBody["stream"] != true {
		t.Errorf("upstream stream = %v, want true", u.receivedBody["stream"])
	}

	events := parseSSEEvents(t, rec.Body.String())
	var eventTypes []string
	for _, e := range events {
		eventTypes = append(eventTypes, e["__event"].(string))
	}
	want := []string{"message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if fmt.Sprint(eventTypes) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", eventTypes, want)
	}

	var text string
	for _, e := range events {
		if e["__event"] == "content_block_delta" {
			delta, _ := e["delta"].(map[string]interface{})
			if delta["type"] == "text_delta" {
				text += delta["text"].(string)
			}
		}
	}
	if text != "Hi there" {
		t.Errorf("streamed text = %q, want %q", text, "Hi there")
	}
}

func TestMessagesNonClaudeModelNotRoutedToChat(t *testing.T) {
	u := newRecordingUpstream()
	defer u.close()
	u.mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"type":"message","content":[{"type":"text","text":"direct"}]}`)
	})

	cfg := Config{
		TargetBaseURL:      u.server.URL,
		TargetChatPath:     "/v1/chat/completions",
		TargetMessagesPath: "/v1/messages",
		TargetModelName:    "test-target",
		ProxyModelName:     "claude-opus-5",
	}
	handler := forwardToUpstream(cfg, cfg.TargetMessagesPath)

	rec := postMessagesRequest(handler, map[string]interface{}{
		"model": "gpt-4o",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if u.receivedPath == "/v1/chat/completions" {
		t.Error("non-claude model should not be routed to chat completions")
	}
	if !strings.Contains(rec.Body.String(), "direct") {
		t.Errorf("expected passthrough of messages response, got %s", rec.Body.String())
	}
}

func TestMessagesPassthroughModeNotRoutedToChat(t *testing.T) {
	u := newRecordingUpstream()
	defer u.close()
	u.mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"type":"message","content":[{"type":"text","text":"direct"}]}`)
	})

	cfg := Config{
		TargetBaseURL:        u.server.URL,
		TargetChatPath:       "/v1/chat/completions",
		TargetMessagesPath:   "/v1/messages",
		TargetModelName:      "test-target",
		ProxyModelName:       "claude-opus-5",
		MessagesUpstreamMode: modePassthrough,
	}
	handler := forwardToUpstream(cfg, cfg.TargetMessagesPath)

	rec := postMessagesRequest(handler, map[string]interface{}{
		"model": "claude-opus-4-5",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if u.receivedPath == "/v1/chat/completions" {
		t.Error("passthrough mode should not route to chat completions")
	}
	if !strings.Contains(rec.Body.String(), "direct") {
		t.Errorf("expected passthrough of messages response, got %s", rec.Body.String())
	}
}

func TestMessagesFallsBackToChatWhenUpstreamHasNoMessagesAPI(t *testing.T) {
	u := newRecordingUpstream()
	defer u.close()
	u.recordAndRespond(http.StatusOK, `{
		"id": "chatcmpl-fallback",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "translated"}, "finish_reason": "stop"}]
	}`)
	u.mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	})

	cfg := Config{
		TargetBaseURL:      u.server.URL,
		TargetChatPath:     "/v1/chat/completions",
		TargetMessagesPath: "/v1/messages",
		TargetModelName:    "test-target",
		ProxyModelName:     "claude-opus-5",
	}
	handler := forwardToUpstream(cfg, cfg.TargetMessagesPath)

	// gpt-4o is not an Anthropic model, so it is offered to the upstream
	// Messages API first; the 404 forces the chat-completions translation.
	rec := postMessagesRequest(handler, map[string]interface{}{
		"model": "gpt-4o",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if u.receivedPath != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", u.receivedPath)
	}
	if u.receivedBody["model"] != "test-target" {
		t.Errorf("upstream model = %v, want test-target", u.receivedBody["model"])
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if resp["type"] != "message" || resp["model"] != "claude-opus-5" {
		t.Errorf("response = %v, want anthropic message with masked model", resp)
	}
}

func TestMessagesUpstreamErrorBecomesAnthropicError(t *testing.T) {
	u := newRecordingUpstream()
	defer u.close()
	u.recordAndRespond(http.StatusTooManyRequests, `{"error":{"message":"slow down","type":"rate_limit_exceeded"}}`)

	cfg := Config{
		TargetBaseURL:      u.server.URL,
		TargetChatPath:     "/v1/chat/completions",
		TargetMessagesPath: "/v1/messages",
		TargetModelName:    "test-target",
		ProxyModelName:     "claude-opus-5",
	}
	handler := forwardToUpstream(cfg, cfg.TargetMessagesPath)

	rec := postMessagesRequest(handler, map[string]interface{}{
		"model": "claude-opus-4-5",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if resp["type"] != "error" {
		t.Errorf("response type = %v, want error", resp["type"])
	}
	errObj, _ := resp["error"].(map[string]interface{})
	if errObj["type"] != "rate_limit_error" || errObj["message"] != "slow down" {
		t.Errorf("error = %v", errObj)
	}
}
