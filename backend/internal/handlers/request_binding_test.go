package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestBindJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		body       string
		maxBytes   int64
		wantStatus int
	}{
		{
			name:       "valid JSON",
			body:       `{"email":"test@example.com"}`,
			maxBytes:   1024,
			wantStatus: http.StatusOK,
		},
		{
			name:       "malformed JSON",
			body:       `{"email":`,
			maxBytes:   1024,
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "oversized JSON",
			body:       `{"email":"test@example.com"}`,
			maxBytes:   4,
			wantStatus: http.StatusRequestEntityTooLarge,
		},
	}

	type requestPayload struct {
		Email string `json:"email"`
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.Use(func(c *gin.Context) {
				c.Request.Body = http.MaxBytesReader(
					c.Writer,
					c.Request.Body,
					tt.maxBytes,
				)
				c.Next()
			})

			router.POST("/test", func(c *gin.Context) {
				var payload requestPayload
				if !bindJSON(c, &payload) {
					return
				}
				c.Status(http.StatusOK)
			})

			req := httptest.NewRequest(
				http.MethodPost,
				"/test",
				strings.NewReader(tt.body),
			)
			req.Header.Set("Content-Type", "application/json")

			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			if recorder.Code != tt.wantStatus {
				t.Fatalf(
					"expected status %d, got %d",
					tt.wantStatus,
					recorder.Code,
				)
			}
		})
	}
}
