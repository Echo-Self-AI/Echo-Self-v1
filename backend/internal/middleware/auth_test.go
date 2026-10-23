package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
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

	assertErrorResponse := func(
		t *testing.T,
		rec *httptest.ResponseRecorder,
		expectedCode string,
	) {
		t.Helper()

		var body struct {
			Error struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}

		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("failed to decode error response: %v", err)
		}

		if body.Error.Code != expectedCode {
			t.Errorf("expected error code %q, got %q",
				expectedCode, body.Error.Code)
		}

		if body.Error.Message == "" {
			t.Error("expected a non-empty error message")
		}
	}

	t.Run("rejects missing token", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/protected", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401, got %d", rec.Code)
		}

		assertErrorResponse(t, rec, "UNAUTHORIZED")
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

		assertErrorResponse(t, rec, "INVALID_ACCESS_TOKEN")
	})

	t.Run("rejects malformed authorization headers", func(t *testing.T) {
		tests := []struct {
			name   string
			header string
		}{
			{name: "missing bearer scheme", header: "some-token"},
			{name: "missing token", header: "Bearer"},
			{name: "extra fields", header: "Bearer token extra"},
			{name: "wrong scheme", header: "Basic some-token"},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				req := httptest.NewRequest(http.MethodGet, "/protected", nil)
				req.Header.Set("Authorization", tt.header)
				rec := httptest.NewRecorder()

				router.ServeHTTP(rec, req)

				if rec.Code != http.StatusUnauthorized {
					t.Fatalf("expected 401, got %d", rec.Code)
				}

				assertErrorResponse(t, rec, "UNAUTHORIZED")
			})
		}
	})

	t.Run("rejects token signed with another secret", func(t *testing.T) {
		otherTokens, err := services.NewTokenService(
			"a-different-secret-that-is-at-least-32-bytes",
		)
		if err != nil {
			t.Fatal(err)
		}

		token, err := otherTokens.GenerateAccessToken("user-123")
		if err != nil {
			t.Fatal(err)
		}

		req := httptest.NewRequest(http.MethodGet, "/protected", nil)
		req.Header.Set("Authorization", "Bearer "+token)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401, got %d: %s", rec.Code, rec.Body.String())
		}

		assertErrorResponse(t, rec, "INVALID_ACCESS_TOKEN")
	})

	t.Run("rejects expired token", func(t *testing.T) {
		expiredClaims := jwt.RegisteredClaims{
			Subject:   "user-123",
			IssuedAt:  jwt.NewNumericDate(time.Now().Add(-2 * time.Hour)),
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(-1 * time.Hour)),
		}

		expiredToken := jwt.NewWithClaims(
			jwt.SigningMethodHS256,
			expiredClaims,
		)

		signedToken, err := expiredToken.SignedString(
			[]byte("test-secret-that-is-at-least-32-bytes-long"),
		)
		if err != nil {
			t.Fatal(err)
		}

		req := httptest.NewRequest(http.MethodGet, "/protected", nil)
		req.Header.Set("Authorization", "Bearer "+signedToken)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401, got %d: %s", rec.Code, rec.Body.String())
		}

		assertErrorResponse(t, rec, "INVALID_ACCESS_TOKEN")
	})
}
