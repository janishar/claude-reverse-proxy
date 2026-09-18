package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// clearProxyEnv removes every variable loadConfig reads, so a test starts from
// documented defaults regardless of what the developer has exported.
func clearProxyEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{
		"SERVER_PORT", "TARGET_BASE_URL", "TARGET_MODELS_PATH", "TARGET_CHAT_PATH",
		"TARGET_RESPONSES_PATH", "TARGET_EMBEDDINGS_PATH", "TARGET_MODERATIONS_PATH",
		"TARGET_COMPLETIONS_PATH", "TARGET_MESSAGES_PATH", "PROXY_MODEL_NAME",
		"TARGET_MODEL_NAME", "PROXY_MODEL_ID", "PROXY_MODEL_OBJECT", "PROXY_MODEL_CREATED",
		"PROXY_MODEL_OWNED_BY", "PROVIDER_API_KEY", "PROXY_API_KEY",
		"MESSAGES_UPSTREAM_MODE", "ANTHROPIC_MODEL_MARKERS", "MAX_BODY_MB",
	} {
		t.Setenv(key, "")
		os.Unsetenv(key)
	}
	defaults := defaultClaudeModelMarkers()
	t.Cleanup(func() { claudeModelMarkers = defaults })
}

func defaultClaudeModelMarkers() []string {
	return []string{"opus", "sonnet", "haik", "fable", "anthropic", "claude-"}
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

func TestConfigDefaults(t *testing.T) {
	clearProxyEnv(t)
	cfg := loadConfig()

	checks := []struct {
		name string
		got  interface{}
		want interface{}
	}{
		{"ServerPort", cfg.ServerPort, "8080"},
		{"TargetBaseURL", cfg.TargetBaseURL, "http://localhost:8000"},
		{"TargetChatPath", cfg.TargetChatPath, "/v1/chat/completions"},
		{"TargetMessagesPath", cfg.TargetMessagesPath, "/v1/messages"},
		{"TargetModelsPath", cfg.TargetModelsPath, "/v1/models"},
		{"TargetResponsesPath", cfg.TargetResponsesPath, "/v1/responses"},
		{"TargetEmbeddingsPath", cfg.TargetEmbeddingsPath, "/v1/embeddings"},
		{"TargetModerationsPath", cfg.TargetModerationsPath, "/v1/moderations"},
		{"TargetCompletionsPath", cfg.TargetCompletionsPath, "/v1/completions"},
		{"ProxyModelName", cfg.ProxyModelName, "claude-opus-5"},
		{"TargetModelName", cfg.TargetModelName, "nvidia/nemotron-3-super-120b-a12b"},
		{"ProxyModelID", cfg.ProxyModelID, "claude-opus-5"},
		{"ProxyModelObject", cfg.ProxyModelObject, "model"},
		{"ProxyModelOwnedBy", cfg.ProxyModelOwnedBy, "cloud-proxy"},
		{"ProviderAPIKey", cfg.ProviderAPIKey, ""},
		{"ProxyAPIKey", cfg.ProxyAPIKey, ""},
		{"MessagesUpstreamMode", cfg.MessagesUpstreamMode, modeAuto},
		{"MaxBodyBytes", cfg.MaxBodyBytes, int64(50 * 1024 * 1024)},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if cfg.ProxyModelCreated == 0 {
		t.Error("ProxyModelCreated should default to the current timestamp")
	}
}

func TestConfigFromEnv(t *testing.T) {
	clearProxyEnv(t)
	for key, value := range map[string]string{
		"SERVER_PORT": "9000", "TARGET_BASE_URL": "https://custom.api.com",
		"TARGET_CHAT_PATH": "/custom/chat", "TARGET_MESSAGES_PATH": "/custom/messages",
		"PROXY_MODEL_NAME": "custom-proxy", "TARGET_MODEL_NAME": "custom-target",
		"PROXY_MODEL_ID": "custom-id", "PROXY_MODEL_OBJECT": "custom-object",
		"PROXY_MODEL_CREATED": "9999999999", "PROXY_MODEL_OWNED_BY": "custom-owner",
		"PROVIDER_API_KEY": "upstream", "PROXY_API_KEY": "downstream",
		"MESSAGES_UPSTREAM_MODE": "PASSTHROUGH", "MAX_BODY_MB": "7",
	} {
		t.Setenv(key, value)
	}

	cfg := loadConfig()

	checks := []struct {
		name string
		got  interface{}
		want interface{}
	}{
		{"ServerPort", cfg.ServerPort, "9000"},
		{"TargetBaseURL", cfg.TargetBaseURL, "https://custom.api.com"},
		{"TargetChatPath", cfg.TargetChatPath, "/custom/chat"},
		{"TargetMessagesPath", cfg.TargetMessagesPath, "/custom/messages"},
		{"ProxyModelName", cfg.ProxyModelName, "custom-proxy"},
		{"TargetModelName", cfg.TargetModelName, "custom-target"},
		{"ProxyModelID", cfg.ProxyModelID, "custom-id"},
		{"ProxyModelObject", cfg.ProxyModelObject, "custom-object"},
		{"ProxyModelCreated", cfg.ProxyModelCreated, int64(9999999999)},
		{"ProxyModelOwnedBy", cfg.ProxyModelOwnedBy, "custom-owner"},
		{"ProviderAPIKey", cfg.ProviderAPIKey, "upstream"},
		{"ProxyAPIKey", cfg.ProxyAPIKey, "downstream"},
		{"MaxBodyBytes", cfg.MaxBodyBytes, int64(7 * 1024 * 1024)},
		// The mode is lowercased so the env var is not case sensitive.
		{"MessagesUpstreamMode", cfg.MessagesUpstreamMode, modePassthrough},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestConfigTrimsTrailingSlashFromBaseURL(t *testing.T) {
	clearProxyEnv(t)
	t.Setenv("TARGET_BASE_URL", "https://api.example.com///")
	if got := loadConfig().TargetBaseURL; got != "https://api.example.com" {
		t.Errorf("TargetBaseURL = %q, want the trailing slashes trimmed", got)
	}
}

func TestConfigOverridesModelMarkers(t *testing.T) {
	clearProxyEnv(t)
	t.Setenv("ANTHROPIC_MODEL_MARKERS", " NEMOTRON , mistral ,, ")
	loadConfig()

	if !isClaudeModelName("nvidia/nemotron-3") {
		t.Error("custom marker should match, case-insensitively and trimmed")
	}
	if isClaudeModelName("claude-opus-4-5") {
		t.Error("custom markers should replace the defaults, not extend them")
	}
}

func TestParseModelMarkers(t *testing.T) {
	cases := []struct {
		raw  string
		want []string
	}{
		{"opus,sonnet", []string{"opus", "sonnet"}},
		{" Opus , SONNET ", []string{"opus", "sonnet"}},
		{"opus,,sonnet,", []string{"opus", "sonnet"}},
		{"", nil},
		{"  ,  ", nil},
	}
	for _, c := range cases {
		got := parseModelMarkers(c.raw)
		if fmt.Sprint(got) != fmt.Sprint(c.want) {
			t.Errorf("parseModelMarkers(%q) = %v, want %v", c.raw, got, c.want)
		}
	}
}

func TestGetEnv(t *testing.T) {
	t.Setenv("PROXY_TEST_VALUE", "set")
	if got := getEnv("PROXY_TEST_VALUE", "fallback"); got != "set" {
		t.Errorf("getEnv = %q, want the environment value", got)
	}
	os.Unsetenv("PROXY_TEST_VALUE")
	if got := getEnv("PROXY_TEST_VALUE", "fallback"); got != "fallback" {
		t.Errorf("getEnv = %q, want the fallback", got)
	}
	t.Setenv("PROXY_TEST_VALUE", "")
	if got := getEnv("PROXY_TEST_VALUE", "fallback"); got != "fallback" {
		t.Errorf("getEnv = %q, want an empty value to fall back", got)
	}
}

func TestGetEnvInt(t *testing.T) {
	t.Setenv("PROXY_TEST_INT", "42")
	if got := getEnvInt("PROXY_TEST_INT", 7); got != 42 {
		t.Errorf("getEnvInt = %d, want 42", got)
	}
	t.Setenv("PROXY_TEST_INT", "not-a-number")
	if got := getEnvInt("PROXY_TEST_INT", 7); got != 7 {
		t.Errorf("getEnvInt = %d, want the fallback for an unparseable value", got)
	}
	os.Unsetenv("PROXY_TEST_INT")
	if got := getEnvInt("PROXY_TEST_INT", 7); got != 7 {
		t.Errorf("getEnvInt = %d, want the fallback when unset", got)
	}
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

func TestModelsEndpointServesTheConfiguredModel(t *testing.T) {
	cfg := Config{
		ProxyModelID:      "test-id",
		ProxyModelObject:  "model",
		ProxyModelCreated: 1789100219,
		ProxyModelOwnedBy: "test-owner",
	}
	rec := httptest.NewRecorder()
	modelsHandler(cfg)(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var resp ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if resp.Object != "list" || len(resp.Data) != 1 {
		t.Fatalf("response = %+v, want a list of one model", resp)
	}
	want := Model{ID: "test-id", Object: "model", Created: 1789100219, OwnedBy: "test-owner"}
	if resp.Data[0] != want {
		t.Errorf("model = %+v, want %+v", resp.Data[0], want)
	}
}

func TestModelsEndpointDoesNotCallUpstream(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("GET /v1/models must be served locally")
	})
	cfg := proxyConfig(upstream.URL)

	rec := httptest.NewRecorder()
	newMux(cfg).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if captured.path != "" {
		t.Errorf("upstream was called at %q", captured.path)
	}
}

func TestModelsEndpointForwardsNonGET(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[]}`)
	})
	cfg := proxyConfig(upstream.URL)

	rec := postJSON(t, newMux(cfg), "/v1/models", map[string]interface{}{"model": "claude-opus-5"}, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if captured.path != "/v1/models" {
		t.Errorf("upstream path = %q, want /v1/models", captured.path)
	}
}

func TestMuxRoutesEachEndpointToItsTargetPath(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{}`)
	})
	cfg := proxyConfig(upstream.URL)
	cfg.TargetChatPath = "/custom/chat"
	cfg.TargetResponsesPath = "/custom/responses"
	cfg.TargetEmbeddingsPath = "/custom/embeddings"
	cfg.TargetModerationsPath = "/custom/moderations"
	cfg.TargetCompletionsPath = "/custom/completions"
	mux := newMux(cfg)

	cases := map[string]string{
		"/v1/chat/completions": "/custom/chat",
		"/v1/responses":        "/custom/responses",
		"/v1/embeddings":       "/custom/embeddings",
		"/v1/moderations":      "/custom/moderations",
		"/v1/completions":      "/custom/completions",
	}
	for inbound, wantPath := range cases {
		captured.path = ""
		postJSON(t, mux, inbound, map[string]interface{}{"model": "claude-opus-5"}, nil)
		if captured.path != wantPath {
			t.Errorf("%s reached upstream at %q, want %q", inbound, captured.path, wantPath)
		}
	}
}

// ---------------------------------------------------------------------------
// Request handling edge cases
// ---------------------------------------------------------------------------

func TestForwardAcceptsEmptyBody(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{}`)
	})
	cfg := proxyConfig(upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(""))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	forwardToUpstream(cfg, cfg.TargetChatPath)(rec, req)

	// An empty body is not malformed JSON; it is simply forwarded.
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if captured.path != "/v1/chat/completions" {
		t.Errorf("upstream path = %q", captured.path)
	}
}

func TestForwardRejectsOversizedBody(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("an oversized request must not reach the upstream")
	})
	cfg := proxyConfig(upstream.URL)
	cfg.MaxBodyBytes = 1024

	body := fmt.Sprintf(`{"model":"claude-opus-5","pad":%q}`, strings.Repeat("x", 4096))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	forwardToUpstream(cfg, cfg.TargetChatPath)(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", rec.Code)
	}
	if captured.path != "" {
		t.Error("upstream should not have been called")
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("413 body is not valid JSON: %v", err)
	}
	errObj, _ := resp["error"].(map[string]interface{})
	if errObj["message"] != "Request body too large" {
		t.Errorf("error = %v", resp["error"])
	}
}

func TestForwardAllowsBodyUnderTheLimit(t *testing.T) {
	upstream, captured := newProxyUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{}`)
	})
	cfg := proxyConfig(upstream.URL)
	cfg.MaxBodyBytes = 8192

	postJSON(t, forwardToUpstream(cfg, cfg.TargetChatPath), "/v1/chat/completions", map[string]interface{}{
		"model": "claude-opus-5",
		"pad":   strings.Repeat("x", 1024),
	}, nil)

	if captured.path != "/v1/chat/completions" {
		t.Errorf("a body under the limit should be forwarded, got path %q", captured.path)
	}
}

func TestForwardReturnsProxyErrorWhenUpstreamIsUnreachable(t *testing.T) {
	cfg := proxyConfig("http://127.0.0.1:1")

	rec := postJSON(t, forwardToUpstream(cfg, cfg.TargetChatPath), "/v1/chat/completions",
		map[string]interface{}{"model": "claude-opus-5"}, nil)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("502 body is not valid JSON: %v", err)
	}
	if _, ok := resp["error"].(map[string]interface{}); !ok {
		t.Errorf("body = %s, want an error envelope", rec.Body.String())
	}
}

func TestMessagesReturnsAnthropicErrorWhenUpstreamIsUnreachable(t *testing.T) {
	cfg := proxyConfig("http://127.0.0.1:1")

	rec := postJSON(t, forwardToUpstream(cfg, cfg.TargetMessagesPath), "/v1/messages",
		map[string]interface{}{"model": "claude-opus-5"}, nil)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("502 body is not valid JSON: %v", err)
	}
	// Anthropic clients need the error envelope, not the OpenAI one.
	if resp["type"] != "error" {
		t.Errorf("body = %s, want an anthropic error envelope", rec.Body.String())
	}
}

// ---------------------------------------------------------------------------
// Model masking units
// ---------------------------------------------------------------------------

func TestShouldMaskModel(t *testing.T) {
	cfg := Config{ProxyModelName: "my-proxy-model", TargetModelName: "upstream"}
	cases := []struct {
		model string
		want  bool
	}{
		{"my-proxy-model", true},
		{"claude-opus-4-5", true},
		{"CLAUDE-SONNET-4", true},
		{"gpt-4o", false},
		{"nvidia/nemotron-3-super-120b-a12b", false},
		{"", false},
	}
	for _, c := range cases {
		if got := shouldMaskModel(cfg, c.model); got != c.want {
			t.Errorf("shouldMaskModel(%q) = %v, want %v", c.model, got, c.want)
		}
	}
}

func TestMaskRequestModel(t *testing.T) {
	cfg := Config{ProxyModelName: "claude-opus-5", TargetModelName: "upstream"}

	t.Run("masks and preserves other fields", func(t *testing.T) {
		original := []byte(`{"model":"claude-opus-5","temperature":0.5}`)
		var parsed map[string]interface{}
		json.Unmarshal(original, &parsed)

		var out map[string]interface{}
		json.Unmarshal(maskRequestModel(cfg, parsed, original), &out)

		if out["model"] != "upstream" {
			t.Errorf("model = %v, want upstream", out["model"])
		}
		if out["temperature"] != 0.5 {
			t.Errorf("temperature = %v, want it preserved", out["temperature"])
		}
	})

	t.Run("returns the original bytes when nothing is masked", func(t *testing.T) {
		original := []byte(`{"model":"gpt-4o"}`)
		var parsed map[string]interface{}
		json.Unmarshal(original, &parsed)

		if got := maskRequestModel(cfg, parsed, original); string(got) != string(original) {
			t.Errorf("got %s, want the body byte-for-byte", got)
		}
	})

	t.Run("a body without a model is untouched", func(t *testing.T) {
		original := []byte(`{"input":"hi"}`)
		var parsed map[string]interface{}
		json.Unmarshal(original, &parsed)

		if got := maskRequestModel(cfg, parsed, original); string(got) != string(original) {
			t.Errorf("got %s, want the body byte-for-byte", got)
		}
	})
}

func TestReplaceModelInResponse(t *testing.T) {
	cfg := Config{ProxyModelName: "claude-opus-5", TargetModelName: "upstream"}

	t.Run("masks every place the model appears", func(t *testing.T) {
		in := []byte(`{"model":"upstream",
			"choices":[{"message":{"model":"upstream","content":"hi"}}],
			"data":[{"model":"upstream"}],
			"message":{"model":"upstream"}}`)

		var out map[string]interface{}
		json.Unmarshal(replaceModelInResponse(cfg, in), &out)

		if out["model"] != "claude-opus-5" {
			t.Errorf("model = %v", out["model"])
		}
		choice := out["choices"].([]interface{})[0].(map[string]interface{})
		if choice["message"].(map[string]interface{})["model"] != "claude-opus-5" {
			t.Errorf("choices[].message.model = %v", choice["message"])
		}
		if out["data"].([]interface{})[0].(map[string]interface{})["model"] != "claude-opus-5" {
			t.Errorf("data[].model = %v", out["data"])
		}
		// message_start events in a streamed Anthropic response.
		if out["message"].(map[string]interface{})["model"] != "claude-opus-5" {
			t.Errorf("message.model = %v", out["message"])
		}
	})

	t.Run("leaves other model names alone", func(t *testing.T) {
		in := []byte(`{"model":"some-other-model"}`)
		if got := replaceModelInResponse(cfg, in); string(got) != string(in) {
			t.Errorf("got %s, want it untouched", got)
		}
	})

	t.Run("passes non-JSON through unchanged", func(t *testing.T) {
		for _, in := range []string{"not json", "", "[1,2,3]"} {
			if got := replaceModelInResponse(cfg, []byte(in)); string(got) != in {
				t.Errorf("replaceModelInResponse(%q) = %q, want it untouched", in, got)
			}
		}
	})
}

func TestMaskModelInSSELine(t *testing.T) {
	cfg := Config{ProxyModelName: "claude-opus-5", TargetModelName: "upstream"}

	t.Run("rewrites the model inside a data frame", func(t *testing.T) {
		got := string(maskModelInSSELine(cfg, []byte(`data: {"model":"upstream","choices":[]}`+"\n")))
		if !strings.Contains(got, "claude-opus-5") || strings.Contains(got, "upstream") {
			t.Errorf("got %q, want the model masked", got)
		}
		if !strings.HasSuffix(got, "\n") {
			t.Errorf("got %q, want the line terminator preserved", got)
		}
	})

	t.Run("leaves non-data lines and sentinels alone", func(t *testing.T) {
		for _, line := range []string{
			"data: [DONE]\n", "\n", "event: ping\n", "id: 1\n", ": comment\n",
			"data: \n", "data: not json\n",
		} {
			if got := string(maskModelInSSELine(cfg, []byte(line))); got != line {
				t.Errorf("maskModelInSSELine(%q) = %q, want it untouched", line, got)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

func TestToInt64(t *testing.T) {
	cases := []struct {
		in   interface{}
		want int64
	}{
		{float64(7), 7}, {7, 7}, {int64(7), 7},
		{"7", 0}, {nil, 0}, {true, 0}, {float64(7.9), 7},
	}
	for _, c := range cases {
		if got := toInt64(c.in); got != c.want {
			t.Errorf("toInt64(%#v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestContentToText(t *testing.T) {
	cases := []struct {
		name string
		in   interface{}
		want string
	}{
		{"string", "hello", "hello"},
		{"blocks", []interface{}{
			map[string]interface{}{"type": "text", "text": "a"},
			map[string]interface{}{"type": "text", "text": "b"},
		}, "a\nb"},
		{"skips non-text blocks", []interface{}{
			map[string]interface{}{"type": "image"},
			map[string]interface{}{"type": "text", "text": "only me"},
		}, "only me"},
		{"empty list", []interface{}{}, ""},
		{"nil", nil, ""},
		{"number", 42, ""},
	}
	for _, c := range cases {
		if got := contentToText(c.in); got != c.want {
			t.Errorf("%s: contentToText = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestExtractBearerToken(t *testing.T) {
	cases := map[string]string{
		"Bearer abc":   "abc",
		"bearer abc":   "abc",
		"BEARER abc":   "abc",
		"Bearer a b c": "a b c",
		"Basic abc":    "",
		"abc":          "",
		"":             "",
		"Bearer":       "",
	}
	for header, want := range cases {
		if got := extractBearerToken(header); got != want {
			t.Errorf("extractBearerToken(%q) = %q, want %q", header, got, want)
		}
	}
}

func TestAnthropicErrorType(t *testing.T) {
	cases := map[int]string{
		400: "invalid_request_error",
		401: "authentication_error",
		403: "permission_error",
		404: "not_found_error",
		413: "request_too_large",
		429: "rate_limit_error",
		500: "api_error",
		502: "api_error",
		529: "overloaded_error",
	}
	for status, want := range cases {
		if got := anthropicErrorType(status); got != want {
			t.Errorf("anthropicErrorType(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestOpenAIErrorToAnthropic(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		wantMsg string
	}{
		{"openai envelope", 400, `{"error":{"message":"bad model","type":"invalid_request_error"}}`, "bad model"},
		{"bare message", 500, `{"message":"boom"}`, "boom"},
		{"plain text", 502, "upstream exploded", "upstream exploded"},
		{"empty body falls back to the status", 500, "", "Internal Server Error"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var out map[string]interface{}
			if err := json.Unmarshal(openAIErrorToAnthropic(c.status, []byte(c.body)), &out); err != nil {
				t.Fatalf("not valid JSON: %v", err)
			}
			if out["type"] != "error" {
				t.Errorf("type = %v, want error", out["type"])
			}
			errObj, _ := out["error"].(map[string]interface{})
			if errObj["message"] != c.wantMsg {
				t.Errorf("message = %v, want %q", errObj["message"], c.wantMsg)
			}
			if errObj["type"] != anthropicErrorType(c.status) {
				t.Errorf("error type = %v, want %q", errObj["type"], anthropicErrorType(c.status))
			}
		})
	}
}

func TestModelStructTags(t *testing.T) {
	data, err := json.Marshal(Model{ID: "m", Object: "model", Created: 1, OwnedBy: "me"})
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"id"`, `"object"`, `"created"`, `"owned_by"`} {
		if !strings.Contains(string(data), field) {
			t.Errorf("serialised model %s is missing %s", data, field)
		}
	}
}
