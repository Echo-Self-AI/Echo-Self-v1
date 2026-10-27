package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

func newHealthTestRouter(handler *HealthHandler) *gin.Engine {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.GET("/health", handler.Liveness)
	router.GET("/ready", handler.Readiness)

	return router
}

func TestHealthLiveness(t *testing.T) {
	handler := &HealthHandler{}
	router := newHealthTestRouter(handler)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body["status"] != "ok" {
		t.Fatalf("expected status ok, got %q", body["status"])
	}
}

func TestHealthReadinessWhenDatabaseUnavailable(t *testing.T) {
	handler := &HealthHandler{
		pingDatabase: func(context.Context) error {
			return errors.New("simulated database outage")
		},
	}
	router := newHealthTestRouter(handler)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}

	var body APIErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body.Error.Code != "DATABASE_UNAVAILABLE" {
		t.Fatalf("expected DATABASE_UNAVAILABLE, got %q", body.Error.Code)
	}
}

func TestHealthReadinessWhenDatabaseIsHealthy(t *testing.T) {
	handler := &HealthHandler{
		pingDatabase: func(context.Context) error {
			return nil
		},
	}
	router := newHealthTestRouter(handler)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}

func TestHealthReadinessWhenRedisUnavailable(t *testing.T) {
	handler := &HealthHandler{
		pingDatabase: func(context.Context) error {
			return nil
		},
		pingRedis: func(context.Context) error {
			return errors.New("simulated redis outage")
		},
	}
	router := newHealthTestRouter(handler)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}

	var body APIErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body.Error.Code != "REDIS_UNAVAILABLE" {
		t.Fatalf("expected REDIS_UNAVAILABLE, got %q", body.Error.Code)
	}
}

func TestHealthReadinessWhenAllHealthy(t *testing.T) {
	handler := &HealthHandler{
		pingDatabase: func(context.Context) error {
			return nil
		},
		pingRedis: func(context.Context) error {
			return nil
		},
	}
	router := newHealthTestRouter(handler)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body["status"] != "ready" || body["database"] != "ok" || body["redis"] != "ok" {
		t.Fatalf("unexpected body: %+v", body)
	}
}

// Keep the GORM import temporarily only if your editor reports it is needed.
// Otherwise remove it from the imports.
var _ *gorm.DB
