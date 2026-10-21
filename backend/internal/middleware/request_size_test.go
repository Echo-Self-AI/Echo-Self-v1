package middleware

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRequestSizeLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(RequestSizeLimit(8))
	router.POST("/test", func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			if _, ok := err.(*http.MaxBytesError); ok {
				c.Status(http.StatusRequestEntityTooLarge)
				return
			}
			c.Status(http.StatusBadRequest)
			return
		}

		c.String(http.StatusOK, string(body))
	})

	tests := []struct {
		name       string
		body       string
		wantStatus int
	}{
		{
			name:       "body within limit",
			body:       "12345678",
			wantStatus: http.StatusOK,
		},
		{
			name:       "body exceeds limit",
			body:       "123456789",
			wantStatus: http.StatusRequestEntityTooLarge,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(
				http.MethodPost,
				"/test",
				strings.NewReader(tt.body),
			)
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			if recorder.Code != tt.wantStatus {
				t.Fatalf("expected status %d, got %d",
					tt.wantStatus, recorder.Code)
			}
		})
	}
}
