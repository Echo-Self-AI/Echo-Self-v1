package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

func TestAuthRequired(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tokens, err := services.NewTokenService(
		"test-secret-that-is-at-least-32-bytes-long",
	)
	if err != nil {
		t.Fatal(err)
	}

	router := gin.New()
	router.GET(
		"/protected",
		AuthRequired(tokens),
		func(c *gin.Context) {
			userID, exists := c.Get(UserIDKey)
			if !exists {
				t.Fatal("expected authenticated user ID")
			}

			c.JSON(http.StatusOK, gin.H{"user_id": userID})
		},
	)

	t.Run("rejects missing token", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/protected", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401, got %d", rec.Code)
		}
	})

	t.Run("accepts valid token", func(t *testing.T) {
		token, err := tokens.GenerateAccessToken("user-123")
		if err != nil {
			t.Fatal(err)
		}

		req := httptest.NewRequest(http.MethodGet, "/protected", nil)
		req.Header.Set("Authorization", "Bearer "+token)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
		}
	})

	t.Run("rejects invalid token", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/protected", nil)
		req.Header.Set("Authorization", "Bearer invalid-token")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401, got %d", rec.Code)
		}
	})
}
