package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Messages upstream modes, selected with MESSAGES_UPSTREAM_MODE.
const (
	// modeAuto translates /v1/messages to /v1/chat/completions when the
	// requested model is an Anthropic one, and otherwise forwards the request
	// to the upstream Messages API (falling back to translation on a 404).
	modeAuto = "auto"
	// modeTranslate always translates /v1/messages to /v1/chat/completions.
	modeTranslate = "translate"
	// modePassthrough never translates: /v1/messages always goes upstream as-is.
	modePassthrough = "passthrough"
)

type Config struct {
	ServerPort            string
	TargetBaseURL         string
	TargetModelsPath      string
	TargetChatPath        string
	TargetResponsesPath   string
	TargetEmbeddingsPath  string
	TargetModerationsPath string
	TargetCompletionsPath string
	TargetMessagesPath    string
	ProxyModelName        string
	TargetModelName       string
	ProxyModelID          string
	ProxyModelObject      string
	ProxyModelCreated     int64
	ProxyModelOwnedBy     string
	ProviderAPIKey        string
	ProxyAPIKey           string
	MessagesUpstreamMode  string
	MaxBodyBytes          int64
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.ParseInt(value, 10, 64); err == nil {
			return intVal
		}
	}
	return defaultValue
}

func loadConfig() Config {
	now := time.Now().Unix()
	if markers := parseModelMarkers(os.Getenv("ANTHROPIC_MODEL_MARKERS")); len(markers) > 0 {
		claudeModelMarkers = markers
	}
	return Config{
		ServerPort:            getEnv("SERVER_PORT", "8080"),
		TargetBaseURL:         strings.TrimRight(getEnv("TARGET_BASE_URL", "http://localhost:8000"), "/"),
		TargetModelsPath:      getEnv("TARGET_MODELS_PATH", "/v1/models"),
		TargetChatPath:        getEnv("TARGET_CHAT_PATH", "/v1/chat/completions"),
		TargetResponsesPath:   getEnv("TARGET_RESPONSES_PATH", "/v1/responses"),
		TargetEmbeddingsPath:  getEnv("TARGET_EMBEDDINGS_PATH", "/v1/embeddings"),
		TargetModerationsPath: getEnv("TARGET_MODERATIONS_PATH", "/v1/moderations"),
		TargetCompletionsPath: getEnv("TARGET_COMPLETIONS_PATH", "/v1/completions"),
		TargetMessagesPath:    getEnv("TARGET_MESSAGES_PATH", "/v1/messages"),
		ProxyModelName:        getEnv("PROXY_MODEL_NAME", "claude-opus-5"),
		TargetModelName:       getEnv("TARGET_MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b"),
		ProxyModelID:          getEnv("PROXY_MODEL_ID", "claude-opus-5"),
		ProxyModelObject:      getEnv("PROXY_MODEL_OBJECT", "model"),
		ProxyModelCreated:     getEnvInt("PROXY_MODEL_CREATED", now),
		ProxyModelOwnedBy:     getEnv("PROXY_MODEL_OWNED_BY", "cloud-proxy"),
		ProviderAPIKey:        getEnv("PROVIDER_API_KEY", ""),
		ProxyAPIKey:           getEnv("PROXY_API_KEY", ""),
		MessagesUpstreamMode:  strings.ToLower(getEnv("MESSAGES_UPSTREAM_MODE", modeAuto)),
		MaxBodyBytes:          getEnvInt("MAX_BODY_MB", 50) * 1024 * 1024,
	}
}

func extractBearerToken(authHeader string) string {
	if authHeader == "" {
		return ""
	}
	parts := strings.SplitN(authHeader, " ", 2)
	if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
		return ""
	}
	return parts[1]
}

// authMiddleware accepts either "Authorization: Bearer <key>" (OpenAI style) or
// "x-api-key: <key>" (Anthropic style), since /v1/messages clients use the latter.
func authMiddleware(cfg Config, next http.HandlerFunc) http.HandlerFunc {
	if cfg.ProxyAPIKey == "" {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		token := extractBearerToken(r.Header.Get("Authorization"))
		if token == "" {
			token = r.Header.Get("X-Api-Key")
		}
		if token != cfg.ProxyAPIKey {
			writeProxyError(w, http.StatusUnauthorized, "Unauthorized")
			return
		}
		next(w, r)
	}
}

// ---------------------------------------------------------------------------
// Model masking
// ---------------------------------------------------------------------------

// claudeModelMarkers are substrings that identify an Anthropic model name.
// Overridable with ANTHROPIC_MODEL_MARKERS (comma separated).
var claudeModelMarkers = []string{"opus", "sonnet", "haik", "fable", "anthropic", "claude-"}

func parseModelMarkers(raw string) []string {
	var markers []string
	for _, part := range strings.Split(raw, ",") {
		if part = strings.ToLower(strings.TrimSpace(part)); part != "" {
			markers = append(markers, part)
		}
	}
	return markers
}

func isClaudeModelName(model string) bool {
	lower := strings.ToLower(model)
	for _, marker := range claudeModelMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// shouldMaskModel reports whether an inbound model name should be swapped for
// TARGET_MODEL_NAME: either it is the name this proxy advertises, or it is some
// other Anthropic model the upstream would not recognise.
func shouldMaskModel(cfg Config, model string) bool {
	if model == "" {
		return false
	}
	return model == cfg.ProxyModelName || isClaudeModelName(model)
}

func maskRequestModel(cfg Config, reqBody map[string]interface{}, original []byte) []byte {
	model, _ := reqBody["model"].(string)
	if !shouldMaskModel(cfg, model) {
		return original
	}
	reqBody["model"] = cfg.TargetModelName
	masked, err := json.Marshal(reqBody)
	if err != nil {
		return original
	}
	return masked
}

// replaceModelInResponse hides the upstream model name from clients, so a
// caller that asked for PROXY_MODEL_NAME is answered by PROXY_MODEL_NAME.
func replaceModelInResponse(cfg Config, body []byte) []byte {
	var resp map[string]interface{}
	if err := json.Unmarshal(body, &resp); err != nil {
		return body
	}

	changed := false
	maskField := func(container map[string]interface{}) {
		if model, ok := container["model"].(string); ok && model == cfg.TargetModelName {
			container["model"] = cfg.ProxyModelName
			changed = true
		}
	}

	maskField(resp)

	if choices, ok := resp["choices"].([]interface{}); ok {
		for _, choice := range choices {
			if choiceMap, ok := choice.(map[string]interface{}); ok {
				if msg, ok := choiceMap["message"].(map[string]interface{}); ok {
					maskField(msg)
				}
			}
		}
	}

	if data, ok := resp["data"].([]interface{}); ok {
		for _, item := range data {
			if itemMap, ok := item.(map[string]interface{}); ok {
				maskField(itemMap)
			}
		}
	}

	// Streaming Anthropic events carry the model inside message_start.
	if message, ok := resp["message"].(map[string]interface{}); ok {
		maskField(message)
	}

	if !changed {
		return body
	}
	newBody, err := json.Marshal(resp)
	if err != nil {
		return body
	}
	return newBody
}

// shouldRouteClaudeToChat decides whether an inbound /v1/messages request has
// to be relayed through the upstream's /v1/chat/completions endpoint.
func shouldRouteClaudeToChat(cfg Config, model string) bool {
	switch cfg.MessagesUpstreamMode {
	case modePassthrough:
		return false
	case modeTranslate:
		return true
	}
	return isClaudeModelName(model) || (model != "" && model == cfg.ProxyModelName)
}

// ---------------------------------------------------------------------------
// Anthropic Messages -> OpenAI Chat Completions
// ---------------------------------------------------------------------------

func toInt64(v interface{}) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int:
		return int64(n)
	case int64:
		return n
	case json.Number:
		i, _ := n.Int64()
		return i
	}
	return 0
}

func contentToText(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, item := range v {
			if block, ok := item.(map[string]interface{}); ok && block["type"] == "text" {
				if text, ok := block["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func imageToOpenAIPart(block map[string]interface{}) interface{} {
	source, _ := block["source"].(map[string]interface{})
	if source == nil {
		return nil
	}
	switch source["type"] {
	case "base64":
		mediaType, _ := source["media_type"].(string)
		data, _ := source["data"].(string)
		if mediaType == "" || data == "" {
			return nil
		}
		return map[string]interface{}{
			"type":      "image_url",
			"image_url": map[string]interface{}{"url": fmt.Sprintf("data:%s;base64,%s", mediaType, data)},
		}
	case "url":
		if url, _ := source["url"].(string); url != "" {
			return map[string]interface{}{
				"type":      "image_url",
				"image_url": map[string]interface{}{"url": url},
			}
		}
	}
	return nil
}

func anthropicToOpenAIRequest(cfg Config, req map[string]interface{}) map[string]interface{} {
	out := map[string]interface{}{"model": cfg.TargetModelName}
	messages := []interface{}{}

	if sys, ok := req["system"]; ok {
		if text := contentToText(sys); text != "" {
			messages = append(messages, map[string]interface{}{"role": "system", "content": text})
		}
	}

	if msgs, ok := req["messages"].([]interface{}); ok {
		for _, m := range msgs {
			msg, ok := m.(map[string]interface{})
			if !ok {
				continue
			}
			role, _ := msg["role"].(string)
			content := msg["content"]

			if role == "user" {
				if s, ok := content.(string); ok {
					messages = append(messages, map[string]interface{}{"role": "user", "content": s})
					continue
				}
				if blocks, ok := content.([]interface{}); ok {
					var textParts []string
					var imageParts []interface{}
					var toolMsgs []interface{}
					for _, b := range blocks {
						bm, ok := b.(map[string]interface{})
						if !ok {
							continue
						}
						switch bm["type"] {
						case "text":
							if t, ok := bm["text"].(string); ok {
								textParts = append(textParts, t)
							}
						case "image":
							if part := imageToOpenAIPart(bm); part != nil {
								imageParts = append(imageParts, part)
							}
						case "tool_result":
							toolID, _ := bm["tool_use_id"].(string)
							toolMsgs = append(toolMsgs, map[string]interface{}{
								"role":         "tool",
								"tool_call_id": toolID,
								"content":      contentToText(bm["content"]),
							})
						}
					}
					// Tool results have to precede the turn's own text so the
					// upstream sees each tool_call answered before new input.
					messages = append(messages, toolMsgs...)
					if len(imageParts) > 0 {
						parts := append([]interface{}{}, imageParts...)
						if len(textParts) > 0 {
							parts = append(parts, map[string]interface{}{"type": "text", "text": strings.Join(textParts, "\n")})
						}
						messages = append(messages, map[string]interface{}{"role": "user", "content": parts})
					} else if len(textParts) > 0 {
						messages = append(messages, map[string]interface{}{"role": "user", "content": strings.Join(textParts, "\n")})
					}
				}
				continue
			}

			// assistant (or any other role): string content passes through,
			// block content expands text blocks + tool_use into tool_calls
			if s, ok := content.(string); ok {
				messages = append(messages, map[string]interface{}{"role": role, "content": s})
				continue
			}
			if blocks, ok := content.([]interface{}); ok {
				var textParts []string
				var toolCalls []interface{}
				for _, b := range blocks {
					bm, ok := b.(map[string]interface{})
					if !ok {
						continue
					}
					switch bm["type"] {
					case "text":
						if t, ok := bm["text"].(string); ok {
							textParts = append(textParts, t)
						}
					case "tool_use":
						id, _ := bm["id"].(string)
						name, _ := bm["name"].(string)
						argsJSON, _ := json.Marshal(bm["input"])
						toolCalls = append(toolCalls, map[string]interface{}{
							"id":   id,
							"type": "function",
							"function": map[string]interface{}{
								"name":      name,
								"arguments": string(argsJSON),
							},
						})
					}
				}
				m := map[string]interface{}{"role": role}
				if len(textParts) > 0 {
					m["content"] = strings.Join(textParts, "\n")
				} else {
					m["content"] = ""
				}
				if len(toolCalls) > 0 {
					m["tool_calls"] = toolCalls
				}
				messages = append(messages, m)
			}
		}
	}
	out["messages"] = messages

	for _, key := range []string{"max_tokens", "temperature", "top_p", "stream", "frequency_penalty", "presence_penalty", "seed", "n"} {
		if v, ok := req[key]; ok {
			out[key] = v
		}
	}
	if v, ok := req["stop_sequences"]; ok {
		out["stop"] = v
	}
	if v, ok := req["response_format"]; ok {
		out["response_format"] = v
	}
	if v, ok := req["metadata"].(map[string]interface{}); ok {
		if user, ok := v["user_id"].(string); ok && user != "" {
			out["user"] = user
		}
	}

	if tools, ok := req["tools"].([]interface{}); ok && len(tools) > 0 {
		var oaiTools []interface{}
		for _, t := range tools {
			tm, ok := t.(map[string]interface{})
			if !ok {
				continue
			}
			fn := map[string]interface{}{}
			if name, ok := tm["name"].(string); ok {
				fn["name"] = name
			}
			if desc, ok := tm["description"].(string); ok {
				fn["description"] = desc
			}
			if schema, ok := tm["input_schema"]; ok {
				fn["parameters"] = schema
			}
			oaiTools = append(oaiTools, map[string]interface{}{"type": "function", "function": fn})
		}
		if len(oaiTools) > 0 {
			out["tools"] = oaiTools
		}
	}

	if tc, ok := req["tool_choice"]; ok {
		if tcm, ok := tc.(map[string]interface{}); ok {
			switch tcm["type"] {
			case "auto":
				out["tool_choice"] = "auto"
			case "any":
				out["tool_choice"] = "required"
			case "none":
				out["tool_choice"] = "none"
			case "tool":
				if name, ok := tcm["name"].(string); ok {
					out["tool_choice"] = map[string]interface{}{
						"type":     "function",
						"function": map[string]interface{}{"name": name},
					}
				}
			}
		} else if s, ok := tc.(string); ok {
			out["tool_choice"] = s
		}
	}

	return out
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions -> Anthropic Messages
// ---------------------------------------------------------------------------

func mapFinishReason(finishReason string) string {
	switch finishReason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func anthropicErrorType(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case 529:
		return "overloaded_error"
	}
	return "api_error"
}

// openAIErrorToAnthropic reshapes an upstream error into the envelope Anthropic
// clients expect, so a failure on /v1/messages is still parseable by them.
func openAIErrorToAnthropic(status int, body []byte) []byte {
	message := strings.TrimSpace(string(body))
	var parsed map[string]interface{}
	if json.Unmarshal(body, &parsed) == nil {
		if e, ok := parsed["error"].(map[string]interface{}); ok {
			if m, ok := e["message"].(string); ok && m != "" {
				message = m
			}
		} else if m, ok := parsed["message"].(string); ok && m != "" {
			message = m
		}
	}
	if message == "" {
		message = http.StatusText(status)
	}
	out, _ := json.Marshal(map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    anthropicErrorType(status),
			"message": message,
		},
	})
	return out
}

func openAIToAnthropicResponse(cfg Config, resp map[string]interface{}) map[string]interface{} {
	out := map[string]interface{}{
		"id":            fmt.Sprintf("msg_%d", time.Now().UnixMilli()),
		"type":          "message",
		"role":          "assistant",
		"model":         cfg.ProxyModelName,
		"content":       []interface{}{},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage":         map[string]interface{}{"input_tokens": 0, "output_tokens": 0},
	}

	if id, ok := resp["id"].(string); ok && id != "" {
		out["id"] = "msg_" + strings.TrimPrefix(id, "chatcmpl-")
	}

	if choices, ok := resp["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				if c, ok := msg["content"].(string); ok && c != "" {
					out["content"] = append(out["content"].([]interface{}), map[string]interface{}{"type": "text", "text": c})
				}
				if tcs, ok := msg["tool_calls"].([]interface{}); ok {
					for _, tc := range tcs {
						tcm, ok := tc.(map[string]interface{})
						if !ok {
							continue
						}
						fn, _ := tcm["function"].(map[string]interface{})
						name, _ := fn["name"].(string)
						id, _ := tcm["id"].(string)
						var input interface{} = map[string]interface{}{}
						if args, ok := fn["arguments"].(string); ok && args != "" {
							var parsedArgs interface{}
							if err := json.Unmarshal([]byte(args), &parsedArgs); err != nil {
								input = args
							} else {
								input = parsedArgs
							}
						}
						out["content"] = append(out["content"].([]interface{}), map[string]interface{}{
							"type":  "tool_use",
							"id":    id,
							"name":  name,
							"input": input,
						})
					}
				}
			}
			if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
				out["stop_reason"] = mapFinishReason(fr)
			}
		}
	}

	if usage, ok := resp["usage"].(map[string]interface{}); ok {
		out["usage"] = map[string]interface{}{
			"input_tokens":  toInt64(usage["prompt_tokens"]),
			"output_tokens": toInt64(usage["completion_tokens"]),
		}
	}

	return out
}

func translateOpenAIStreamToAnthropic(cfg Config, upstream io.Reader, w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	msgID := fmt.Sprintf("msg_%d", time.Now().UnixMilli())
	started := false
	textOpen := false
	blockIndex := 0
	var openToolBlocks map[int]int
	inputTokens := int64(0)
	outputTokens := int64(0)
	stopReason := "end_turn"
	stopSent := false

	send := func(eventType string, data map[string]interface{}) {
		payload, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, payload)
		if flusher != nil {
			flusher.Flush()
		}
	}

	emitStart := func() {
		if started {
			return
		}
		started = true
		send("message_start", map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id":            msgID,
				"type":          "message",
				"role":          "assistant",
				"model":         cfg.ProxyModelName,
				"content":       []interface{}{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage":         map[string]interface{}{"input_tokens": inputTokens, "output_tokens": 0},
			},
		})
	}

	closeText := func() {
		if textOpen {
			send("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": blockIndex})
			textOpen = false
			blockIndex++
		}
	}

	closeAll := func() {
		closeText()
		for _, idx := range openToolBlocks {
			send("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": idx})
		}
		openToolBlocks = nil
	}

	emitStop := func() {
		if stopSent {
			return
		}
		stopSent = true
		emitStart()
		closeAll()
		send("message_delta", map[string]interface{}{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": stopReason, "stop_sequence": nil},
			"usage": map[string]interface{}{"input_tokens": inputTokens, "output_tokens": outputTokens},
		})
		send("message_stop", map[string]interface{}{"type": "message_stop"})
	}

	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunkID, ok := chunk["id"].(string); ok && chunkID != "" && !started {
			msgID = "msg_" + strings.TrimPrefix(chunkID, "chatcmpl-")
		}

		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			if pt, ok := usage["prompt_tokens"]; ok {
				inputTokens = toInt64(pt)
			}
			if ct, ok := usage["completion_tokens"]; ok {
				outputTokens = toInt64(ct)
			}
		}

		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			choice, _ := choices[0].(map[string]interface{})
			delta, _ := choice["delta"].(map[string]interface{})

			if delta != nil {
				if text, ok := delta["content"].(string); ok && text != "" {
					if !textOpen {
						emitStart()
						send("content_block_start", map[string]interface{}{
							"type":          "content_block_start",
							"index":         blockIndex,
							"content_block": map[string]interface{}{"type": "text", "text": ""},
						})
						textOpen = true
					}
					send("content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": blockIndex,
						"delta": map[string]interface{}{"type": "text_delta", "text": text},
					})
				}
				if tcs, ok := delta["tool_calls"].([]interface{}); ok {
					if openToolBlocks == nil {
						openToolBlocks = map[int]int{}
					}
					for _, tc := range tcs {
						tcm, ok := tc.(map[string]interface{})
						if !ok {
							continue
						}
						fn, _ := tcm["function"].(map[string]interface{})
						tcIndex := int(toInt64(tcm["index"]))
						if _, exists := openToolBlocks[tcIndex]; !exists {
							emitStart()
							closeText()
							tcID, _ := tcm["id"].(string)
							tcName, _ := fn["name"].(string)
							openToolBlocks[tcIndex] = blockIndex
							send("content_block_start", map[string]interface{}{
								"type":  "content_block_start",
								"index": blockIndex,
								"content_block": map[string]interface{}{
									"type":  "tool_use",
									"id":    tcID,
									"name":  tcName,
									"input": map[string]interface{}{},
								},
							})
							blockIndex++
						}
						if args, ok := fn["arguments"].(string); ok && args != "" {
							send("content_block_delta", map[string]interface{}{
								"type":  "content_block_delta",
								"index": openToolBlocks[tcIndex],
								"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": args},
							})
						}
					}
				}
			}

			if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
				stopReason = mapFinishReason(fr)
			}
		}
	}

	emitStop()
}

// ---------------------------------------------------------------------------
// Upstream forwarding
// ---------------------------------------------------------------------------

// No client timeout: streamed completions stay open far longer than any
// sensible request deadline. The transport bounds connect/header latency.
var httpClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConnsPerHost:   32,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: 180 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

var hopByHopHeaders = map[string]bool{
	"connection":          true,
	"keep-alive":          true,
	"proxy-authenticate":  true,
	"proxy-authorization": true,
	"te":                  true,
	"trailer":             true,
	"transfer-encoding":   true,
	"upgrade":             true,
}

func isEventStream(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}

// doUpstream sends body to targetURL, carrying over the client's headers minus
// the ones that belong to this hop, and swapping in the provider credential.
func doUpstream(cfg Config, r *http.Request, targetURL string, body []byte, wantStream bool) (*http.Response, error) {
	req, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for key, values := range r.Header {
		lower := strings.ToLower(key)
		if hopByHopHeaders[lower] || lower == "content-length" || lower == "accept-encoding" || lower == "host" {
			continue
		}
		for _, v := range values {
			req.Header.Add(key, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	if wantStream {
		req.Header.Set("Accept", "text/event-stream")
	} else if req.Header.Get("Accept") == "" {
		req.Header.Set("Accept", "application/json")
	}
	if cfg.ProviderAPIKey != "" {
		// Never leak the caller's own credential to the upstream.
		req.Header.Set("Authorization", "Bearer "+cfg.ProviderAPIKey)
		req.Header.Del("X-Api-Key")
	}
	req.ContentLength = int64(len(body))
	return httpClient.Do(req)
}

func copyResponseHeaders(w http.ResponseWriter, src http.Header) {
	for key, values := range src {
		lower := strings.ToLower(key)
		// Content-Length/Encoding are dropped: the body may be rewritten here
		// and the transport has already decoded any upstream compression.
		if hopByHopHeaders[lower] || lower == "content-length" || lower == "content-encoding" {
			continue
		}
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}
}

// maskModelInSSELine rewrites the model name inside one `data:` frame,
// leaving every other SSE line (event:, id:, blank separators) untouched.
func maskModelInSSELine(cfg Config, line []byte) []byte {
	trimmed := bytes.TrimRight(line, "\r\n")
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return line
	}
	payload := bytes.TrimSpace(trimmed[len("data:"):])
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return line
	}
	masked := replaceModelInResponse(cfg, payload)
	if bytes.Equal(masked, payload) {
		return line
	}
	return append(append([]byte("data: "), masked...), '\n')
}

func streamUpstreamResponse(cfg Config, w http.ResponseWriter, resp *http.Response) {
	copyResponseHeaders(w, resp.Header)
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if _, werr := w.Write(maskModelInSSELine(cfg, line)); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

func writeUpstreamResponse(cfg Config, w http.ResponseWriter, resp *http.Response) {
	if isEventStream(resp.Header.Get("Content-Type")) {
		streamUpstreamResponse(cfg, w, resp)
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeProxyError(w, http.StatusBadGateway, "Failed to read upstream response")
		return
	}
	body = replaceModelInResponse(cfg, body)

	copyResponseHeaders(w, resp.Header)
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

// handleClaudeViaChatCompletions relays one Anthropic Messages request through
// the upstream's OpenAI-compatible chat completions endpoint and converts the
// answer (streamed or not) back into Anthropic shape.
func handleClaudeViaChatCompletions(cfg Config, w http.ResponseWriter, r *http.Request, parsed map[string]interface{}) {
	openAIReq := anthropicToOpenAIRequest(cfg, parsed)
	body, err := json.Marshal(openAIReq)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "Failed to encode upstream request")
		return
	}

	stream, _ := openAIReq["stream"].(bool)
	chatResp, err := doUpstream(cfg, r, cfg.TargetBaseURL+cfg.TargetChatPath, body, stream)
	if err != nil {
		log.Printf("forward to chat completions failed: %v", err)
		writeAnthropicError(w, http.StatusBadGateway, "Failed to forward request to upstream")
		return
	}
	defer chatResp.Body.Close()

	if chatResp.StatusCode >= 400 {
		errBody, _ := io.ReadAll(chatResp.Body)
		out := openAIErrorToAnthropic(chatResp.StatusCode, errBody)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Length", strconv.Itoa(len(out)))
		w.WriteHeader(chatResp.StatusCode)
		w.Write(out)
		return
	}

	if stream {
		translateOpenAIStreamToAnthropic(cfg, chatResp.Body, w)
		return
	}

	respBytes, err := io.ReadAll(chatResp.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "Failed to read upstream response")
		return
	}
	var respMap map[string]interface{}
	if err := json.Unmarshal(respBytes, &respMap); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write(replaceModelInResponse(cfg, respBytes))
		return
	}
	out, _ := json.Marshal(openAIToAnthropicResponse(cfg, respMap))
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.Write(out)
}

// writeProxyError answers with the JSON envelope the OpenAI routes use.
func writeProxyError(w http.ResponseWriter, status int, message string) {
	out, _ := json.Marshal(map[string]interface{}{
		"error": map[string]interface{}{"message": message, "type": "proxy_error"},
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(status)
	w.Write(out)
}

func writeAnthropicError(w http.ResponseWriter, status int, message string) {
	out, _ := json.Marshal(map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    anthropicErrorType(status),
			"message": message,
		},
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(status)
	w.Write(out)
}

// handleMessages serves /v1/messages: Anthropic models are relayed through the
// chat completions endpoint, anything else is offered to the upstream Messages
// API first and only translated if the upstream does not implement it.
func handleMessages(cfg Config, w http.ResponseWriter, r *http.Request, body []byte, parsed map[string]interface{}) {
	model, _ := parsed["model"].(string)

	if parsed != nil && shouldRouteClaudeToChat(cfg, model) {
		handleClaudeViaChatCompletions(cfg, w, r, parsed)
		return
	}

	outBody := body
	if parsed != nil {
		outBody = maskRequestModel(cfg, parsed, body)
	}
	stream, _ := parsed["stream"].(bool)

	messagesResp, err := doUpstream(cfg, r, cfg.TargetBaseURL+cfg.TargetMessagesPath, outBody, stream)
	if err != nil {
		log.Printf("forward to messages failed: %v", err)
		writeAnthropicError(w, http.StatusBadGateway, "Failed to forward request to upstream")
		return
	}
	defer messagesResp.Body.Close()

	if messagesResp.StatusCode == http.StatusNotFound && parsed != nil {
		// Upstream has no Messages API: translate and retry against chat completions.
		io.Copy(io.Discard, messagesResp.Body)
		handleClaudeViaChatCompletions(cfg, w, r, parsed)
		return
	}

	writeUpstreamResponse(cfg, w, messagesResp)
}

// maxDrainBytes bounds how much of an oversized request we are willing to read
// and discard before answering it.
const maxDrainBytes = 256 << 20

func methodHasBody(method string) bool {
	return method == http.MethodPost || method == http.MethodPut || method == http.MethodPatch
}

// forwardToUpstream builds the handler for one proxied path. An empty
// targetPath means "reuse the inbound path" (the /v1/ wildcard).
func forwardToUpstream(cfg Config, targetPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rawBody := r.Body
		if cfg.MaxBodyBytes > 0 {
			r.Body = http.MaxBytesReader(w, r.Body, cfg.MaxBodyBytes)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				// Drain what the client is still sending: replying before the
				// request is complete makes the client see a reset, not the 413.
				io.CopyN(io.Discard, rawBody, maxDrainBytes)
				writeProxyError(w, http.StatusRequestEntityTooLarge, "Request body too large")
				return
			}
			writeProxyError(w, http.StatusBadRequest, "Failed to read request body")
			return
		}
		r.Body.Close()

		var parsed map[string]interface{}
		if len(body) > 0 && methodHasBody(r.Method) {
			// These APIs only ever take a JSON object; a bare string, list or
			// null would be forwarded as something the upstream cannot use.
			if err := json.Unmarshal(body, &parsed); err != nil || parsed == nil {
				writeProxyError(w, http.StatusBadRequest, "Invalid JSON")
				return
			}
		}

		if targetPath != "" && targetPath == cfg.TargetMessagesPath {
			handleMessages(cfg, w, r, body, parsed)
			return
		}

		path := targetPath
		if path == "" {
			path = r.URL.Path
		}
		if r.URL.RawQuery != "" {
			path += "?" + r.URL.RawQuery
		}

		if parsed != nil {
			body = maskRequestModel(cfg, parsed, body)
		}
		stream, _ := parsed["stream"].(bool)

		resp, err := doUpstream(cfg, r, cfg.TargetBaseURL+path, body, stream)
		if err != nil {
			log.Printf("forward request failed: %v", err)
			writeProxyError(w, http.StatusBadGateway, "Failed to forward request")
			return
		}
		defer resp.Body.Close()

		writeUpstreamResponse(cfg, w, resp)
	}
}

func modelsHandler(cfg Config) http.HandlerFunc {
	// Only GET is served locally; anything else on this path is proxied, so a
	// client can still reach an upstream that does more with /v1/models.
	forward := forwardToUpstream(cfg, "")
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			forward(w, r)
			return
		}

		response := ModelsResponse{
			Object: "list",
			Data: []Model{
				{
					ID:      cfg.ProxyModelID,
					Object:  cfg.ProxyModelObject,
					Created: cfg.ProxyModelCreated,
					OwnedBy: cfg.ProxyModelOwnedBy,
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func newMux(cfg Config) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/v1/models", authMiddleware(cfg, modelsHandler(cfg)))
	mux.Handle("/v1/chat/completions", authMiddleware(cfg, forwardToUpstream(cfg, cfg.TargetChatPath)))
	mux.Handle("/v1/responses", authMiddleware(cfg, forwardToUpstream(cfg, cfg.TargetResponsesPath)))
	mux.Handle("/v1/embeddings", authMiddleware(cfg, forwardToUpstream(cfg, cfg.TargetEmbeddingsPath)))
	mux.Handle("/v1/moderations", authMiddleware(cfg, forwardToUpstream(cfg, cfg.TargetModerationsPath)))
	mux.Handle("/v1/completions", authMiddleware(cfg, forwardToUpstream(cfg, cfg.TargetCompletionsPath)))
	mux.Handle("/v1/messages", authMiddleware(cfg, forwardToUpstream(cfg, cfg.TargetMessagesPath)))
	mux.Handle("/v1/", authMiddleware(cfg, forwardToUpstream(cfg, "")))
	return mux
}

func main() {
	cfg := loadConfig()

	if _, err := url.Parse(cfg.TargetBaseURL); err != nil {
		log.Fatalf("Invalid TARGET_BASE_URL: %v", err)
	}

	server := &http.Server{
		Addr:              ":" + cfg.ServerPort,
		Handler:           newMux(cfg),
		ReadHeaderTimeout: 30 * time.Second,
	}

	log.Printf("Server starting on %s", server.Addr)
	log.Printf("Target API: %s", cfg.TargetBaseURL)
	log.Printf("Model mapping: %s -> %s", cfg.ProxyModelName, cfg.TargetModelName)
	log.Printf("Messages relay mode: %s (markers: %s)", cfg.MessagesUpstreamMode, strings.Join(claudeModelMarkers, ","))
	if cfg.ProxyAPIKey != "" {
		log.Printf("Proxy API key authentication: ENABLED")
	}
	log.Fatal(server.ListenAndServe())
}
