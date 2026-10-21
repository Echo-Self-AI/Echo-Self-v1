package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

func TestRateLimitBlocksExcessRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(RateLimit())
	router.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	var lastStatus int

	for i := 0; i < 25; i++ {
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		req.RemoteAddr = "192.0.2.10:12345"

		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, req)
		lastStatus = recorder.Code
	}

	if lastStatus != http.StatusTooManyRequests {
		t.Fatalf("expected status %d, got %d",
			http.StatusTooManyRequests, lastStatus)
	}
}

func TestRateLimitWithConfigBlocksAfterBurst(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(RateLimitWithConfig(rate.Limit(0.001), 2))
	router.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		req.RemoteAddr = "192.0.2.20:12345"

		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, req)

		if recorder.Code != http.StatusOK {
			t.Fatalf("request %d: expected status %d, got %d",
				i+1, http.StatusOK, recorder.Code)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.RemoteAddr = "192.0.2.20:12345"

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusTooManyRequests {
		t.Fatalf("expected status %d after burst, got %d",
			http.StatusTooManyRequests, recorder.Code)
	}
}

func TestRateLimitIgnoresSpoofedForwardedIP(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()

	if err := router.SetTrustedProxies(nil); err != nil {
		t.Fatalf("failed to configure trusted proxies: %v", err)
	}

	router.Use(RateLimitWithConfig(0.001, 1))
	router.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	makeRequest := func(forwardedIP string) int {
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		req.RemoteAddr = "192.0.2.30:12345"
		req.Header.Set("X-Forwarded-For", forwardedIP)

		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, req)
		return recorder.Code
	}

	firstStatus := makeRequest("203.0.113.10")
	if firstStatus != http.StatusOK {
		t.Fatalf("first request: expected %d, got %d",
			http.StatusOK, firstStatus)
	}

	secondStatus := makeRequest("203.0.113.20")
	if secondStatus != http.StatusTooManyRequests {
		t.Fatalf("second request should share the same client IP limit: expected %d, got %d",
			http.StatusTooManyRequests, secondStatus)
	}
}

func TestRateLimiterStoreCleansUpStaleClients(t *testing.T) {
	store := newRateLimiterStore()

	// Insert a client that was last seen 11 minutes ago.
	staleLimiter := &clientLimiter{
		limiter:  rate.NewLimiter(1, 1),
		lastSeen: time.Now().Add(-11 * time.Minute),
	}

	store.clients["192.0.2.40"] = staleLimiter

	// Force the cleanup check to run.
	store.lastCleanup = time.Now().Add(-2 * time.Minute)

	// Access another client to trigger cleanup.
	store.getLimiter("192.0.2.41", rate.Limit(1), 1)

	store.mu.Lock()
	defer store.mu.Unlock()

	if _, exists := store.clients["192.0.2.40"]; exists {
		t.Fatal("expected stale client to be removed")
	}

	if _, exists := store.clients["192.0.2.41"]; !exists {
		t.Fatal("expected active client to remain in the store")
	}
}
