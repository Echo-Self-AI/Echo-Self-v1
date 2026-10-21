package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
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
		t.Fatalf(
			"expected final request to return %d, got %d",
			http.StatusTooManyRequests,
			lastStatus,
		)
	}
}
