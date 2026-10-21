package middleware

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRequestLogger(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&output, nil))

	router := gin.New()
	router.Use(RequestID())
	router.Use(RequestLogger(logger))

	router.GET("/users/:id", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(
		http.MethodGet,
		"/users/123?secret=query-secret",
		nil,
	)
	req.Header.Set("Authorization", "Bearer header-secret")

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, recorder.Code)
	}

	var entry map[string]any
	if err := json.Unmarshal(output.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse JSON log: %v; output: %s", err, output.String())
	}

	if entry["msg"] != "http request" {
		t.Errorf("unexpected log message: %v", entry["msg"])
	}
	if entry["method"] != http.MethodGet {
		t.Errorf("expected GET method, got %v", entry["method"])
	}
	if entry["route"] != "/users/:id" {
		t.Errorf("expected route /users/:id, got %v", entry["route"])
	}
	if entry["status"] != float64(http.StatusOK) {
		t.Errorf("expected status 200, got %v", entry["status"])
	}

	requestID, ok := entry["request_id"].(string)
	if !ok || requestID == "" {
		t.Errorf("expected a non-empty request ID, got %v", entry["request_id"])
	}

	if _, ok := entry["duration_ms"].(float64); !ok {
		t.Errorf("expected duration_ms in log, got %v", entry["duration_ms"])
	}

	if strings.Contains(output.String(), "query-secret") {
		t.Error("log must not contain query parameters")
	}
	if strings.Contains(output.String(), "header-secret") {
		t.Error("log must not contain authorization headers")
	}
}
