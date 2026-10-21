package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func TestRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(RequestID())
	router.GET("/test", func(c *gin.Context) {
		value, exists := c.Get(RequestIDKey)
		if !exists {
			t.Fatal("expected request ID in Gin context")
		}

		requestID, ok := value.(string)
		if !ok {
			t.Fatal("expected request ID to be a string")
		}

		if _, err := uuid.Parse(requestID); err != nil {
			t.Fatalf("expected a valid UUID, got %q", requestID)
		}

		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d",
			http.StatusOK, recorder.Code)
	}

	headerID := recorder.Header().Get("X-Request-ID")
	if headerID == "" {
		t.Fatal("expected X-Request-ID response header")
	}

	if _, err := uuid.Parse(headerID); err != nil {
		t.Fatalf("expected a valid UUID in response header, got %q", headerID)
	}
}

func TestRequestIDDoesNotTrustClientHeader(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(RequestID())
	router.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("X-Request-ID", "client-controlled-id")

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	if got := recorder.Header().Get("X-Request-ID"); got == "client-controlled-id" {
		t.Fatal("server should generate its own request ID")
	}

	if _, err := uuid.Parse(recorder.Header().Get("X-Request-ID")); err != nil {
		t.Fatalf("expected server-generated UUID, got %q",
			recorder.Header().Get("X-Request-ID"))
	}
}
