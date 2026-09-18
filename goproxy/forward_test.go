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

type capturedRequest struct {
	path    string
	query   string
	headers http.Header
	body    map[string]interface{}
}

// newProxyUpstream returns a fake upstream plus a pointer to the last request
// it saw, so tests can assert on what the proxy actually forwarded.
func newProxyUpstream(t *testing.T, handle func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, *capturedRequest) {
	t.Helper()
	captured := &capturedRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.Path
		captured.query = r.URL.RawQuery
		captured.headers = r.Header.Clone()
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
			captured.body = body
		}
		handle(w, r)
	}))
	t.Cleanup(server.Close)
	return server, captured
}

func proxyConfig(baseURL string) Config {
	return Config{
		TargetBaseURL:         baseURL,
		TargetChatPath:        "/v1/chat/completions",
		TargetMessagesPath:    "/v1/messages",
		TargetEmbeddingsPath:  "/v1/embeddings",
		TargetCompletionsPath: "/v1/completions",
		ProxyModelName:        "claude-opus-5",
		TargetModelName:       "test-target",
		ProviderAPIKey:        "upstream-key",
	}
}

func postJSON(t *testing.T, handler http.Handler, path string, payload map[string]interface{}, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec
}

func TestChatCompletionsMasksModelBothWays(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl-1","model":"test-target","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, cfg.TargetChatPath)

	rec := postJSON(t, handler, "/v1/chat/completions", map[string]interface{}{
		"model":       "claude-opus-5",
		"messages":    []interface{}{map[string]interface{}{"role": "user", "content": "hello"}},
		"temperature": 0.5,
	}, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if captured.body["model"] != "test-target" {
		t.Errorf("upstream model = %v, want test-target", captured.body["model"])
	}
	if captured.body["temperature"] != 0.5 {
		t.Errorf("temperature = %v, want it forwarded unchanged", captured.body["temperature"])
	}
	// An OpenAI-shaped request must reach the upstream untranslated.
	if _, hasSystem := captured.body["system"]; hasSystem {
		t.Error("chat completions request should not be rewritten into Anthropic shape")
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if resp["model"] != "claude-opus-5" {
		t.Errorf("response model = %v, want claude-opus-5", resp["model"])
	}
	if resp["choices"] == nil {
		t.Error("chat completions response should keep its OpenAI shape")
	}
}

func TestForwardMasksAnyAnthropicModelName(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"test-target"}`)
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, cfg.TargetChatPath)

	postJSON(t, handler, "/v1/chat/completions", map[string]interface{}{
		"model":    "claude-sonnet-4-5",
		"messages": []interface{}{},
	}, nil)

	if captured.body["model"] != "test-target" {
		t.Errorf("upstream model = %v, want test-target", captured.body["model"])
	}
}

func TestForwardLeavesForeignModelAlone(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"gpt-4o"}`)
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, cfg.TargetChatPath)

	rec := postJSON(t, handler, "/v1/chat/completions", map[string]interface{}{
		"model":    "gpt-4o",
		"messages": []interface{}{},
	}, nil)

	if captured.body["model"] != "gpt-4o" {
		t.Errorf("upstream model = %v, want gpt-4o untouched", captured.body["model"])
	}
	if !strings.Contains(rec.Body.String(), "gpt-4o") {
		t.Errorf("response = %s, want gpt-4o untouched", rec.Body.String())
	}
}

func TestForwardUsesProviderKeyAndDropsClientCredentials(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{}`)
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, cfg.TargetChatPath)

	postJSON(t, handler, "/v1/chat/completions", map[string]interface{}{"model": "claude-opus-5"}, map[string]string{
		"Authorization": "Bearer client-proxy-key",
		"X-Api-Key":     "client-proxy-key",
	})

	if got := captured.headers.Get("Authorization"); got != "Bearer upstream-key" {
		t.Errorf("upstream Authorization = %q, want the provider key", got)
	}
	if got := captured.headers.Get("X-Api-Key"); got != "" {
		t.Errorf("upstream X-Api-Key = %q, want the client credential stripped", got)
	}
}

func TestForwardStreamPassesSSEThroughWithMaskedModel(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-target\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, cfg.TargetChatPath)

	rec := postJSON(t, handler, "/v1/chat/completions", map[string]interface{}{
		"model":  "claude-opus-5",
		"stream": true,
	}, nil)

	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if captured.headers.Get("Accept") != "text/event-stream" {
		t.Errorf("upstream Accept = %q, want text/event-stream", captured.headers.Get("Accept"))
	}
	body := rec.Body.String()
	if strings.Contains(body, "test-target") {
		t.Errorf("streamed body leaks the upstream model: %s", body)
	}
	if !strings.Contains(body, `"model":"claude-opus-5"`) {
		t.Errorf("streamed body = %s, want the masked model", body)
	}
	// OpenAI clients on this route must still receive OpenAI chunks.
	if !strings.Contains(body, `"choices"`) || strings.Contains(body, "content_block_delta") {
		t.Errorf("streamed body should stay in OpenAI shape: %s", body)
	}
	if !strings.HasSuffix(strings.TrimRight(body, "\n"), "data: [DONE]") {
		t.Errorf("streamed body should end with [DONE]: %q", body)
	}
}

func TestWildcardForwardsInboundPathAndQuery(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{}`)
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, "")

	req := httptest.NewRequest(http.MethodGet, "/v1/files?limit=2", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if captured.path != "/v1/files" {
		t.Errorf("upstream path = %q, want /v1/files", captured.path)
	}
	if captured.query != "limit=2" {
		t.Errorf("upstream query = %q, want limit=2", captured.query)
	}
}

func TestForwardRejectsInvalidJSON(t *testing.T) {
	upstream, _ := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream should not be reached for invalid JSON")
	})
	cfg := proxyConfig(upstream.URL)
	handler := forwardToUpstream(cfg, cfg.TargetChatPath)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestAuthMiddlewareAcceptsAnthropicApiKeyHeader(t *testing.T) {
	cfg := Config{ProxyAPIKey: "secret"}
	handler := authMiddleware(cfg, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	for name, header := range map[string]map[string]string{
		"bearer":    {"Authorization": "Bearer secret"},
		"x-api-key": {"X-Api-Key": "secret"},
	} {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
		for k, v := range header {
			req.Header.Set(k, v)
		}
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("%s: status = %d, want 200", name, rec.Code)
		}
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", rec.Code)
	}
}
