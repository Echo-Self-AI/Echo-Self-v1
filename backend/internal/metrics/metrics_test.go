package metrics

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestMetricsRegistry_RecordAndExpose(t *testing.T) {
	reg := NewRegistry()

	reg.IncActiveRequests()
	reg.RecordHTTPRequest("GET", "/api/v1/auth/me", 200, 50*time.Millisecond)
	reg.DecActiveRequests()

	reg.RecordJobState("avatar_generation", "completed")
	reg.RecordJobDuration("avatar_generation", 1500*time.Millisecond)
	reg.RecordJobRetry("avatar_generation")
	reg.RecordLeaseRenewalFailure()
	reg.RecordStaleJobRecovery()
	reg.RecordOutboxDispatch("success", 20*time.Millisecond)

	output := reg.ExposePrometheus()

	if !strings.Contains(output, "echoself_http_active_requests 0") {
		t.Errorf("expected active requests 0 in output, got:\n%s", output)
	}

	if !strings.Contains(output, `echoself_http_requests_total{method="GET",status="200",path="/api/v1/auth/me"} 1`) {
		t.Errorf("expected http_requests_total metric in output, got:\n%s", output)
	}

	if !strings.Contains(output, `echoself_jobs_total{type="avatar_generation",status="completed"} 1`) {
		t.Errorf("expected jobs_total metric in output, got:\n%s", output)
	}

	if !strings.Contains(output, `echoself_job_retries_total{type="avatar_generation"} 1`) {
		t.Errorf("expected job_retries_total in output, got:\n%s", output)
	}

	if !strings.Contains(output, "echoself_job_lease_renewal_failures_total 1") {
		t.Errorf("expected lease failure count 1 in output, got:\n%s", output)
	}

	if !strings.Contains(output, "echoself_job_stale_recoveries_total 1") {
		t.Errorf("expected stale recovery count 1 in output, got:\n%s", output)
	}

	if !strings.Contains(output, `echoself_outbox_dispatches_total{status="success"} 1`) {
		t.Errorf("expected outbox dispatch metric in output, got:\n%s", output)
	}
}

func TestMetrics_MiddlewareAndHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	router.Use(Middleware())
	router.GET("/metrics", Handler())
	router.GET("/test-ping", func(c *gin.Context) {
		c.String(http.StatusOK, "pong")
	})

	// Make request to test-ping
	req := httptest.NewRequest(http.MethodGet, "/test-ping", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	// Fetch metrics
	mReq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	mW := httptest.NewRecorder()
	router.ServeHTTP(mW, mReq)

	if mW.Code != http.StatusOK {
		t.Fatalf("expected 200 on /metrics, got %d", mW.Code)
	}

	body := mW.Body.String()
	if !strings.Contains(body, `echoself_http_requests_total{method="GET",status="200",path="/test-ping"} 1`) {
		t.Errorf("expected /test-ping in metrics, got:\n%s", body)
	}
}
