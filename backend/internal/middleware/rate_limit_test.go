package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

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
