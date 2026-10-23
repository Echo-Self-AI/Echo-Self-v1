package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type lifecycleResponse struct {
	ID           string `json:"id"`
	Email        string `json:"email"`
	DisplayName  string `json:"display_name"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int64  `json:"expires_in"`
	Error        *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func performJSONRequest(
	t *testing.T,
	router http.Handler,
	method string,
	path string,
	body any,
) *httptest.ResponseRecorder {
	t.Helper()

	var requestBody bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&requestBody).Encode(body); err != nil {
			t.Fatalf("encode request body: %v", err)
		}
	}

	req := httptest.NewRequest(method, path, &requestBody)
	req.Header.Set("Content-Type", "application/json")

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)
	return recorder
}

func decodeLifecycleResponse(
	t *testing.T,
	recorder *httptest.ResponseRecorder,
) lifecycleResponse {
	t.Helper()

	var response lifecycleResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response (status %d): %v; body: %s",
			recorder.Code, err, recorder.Body.String())
	}
	return response
}

func TestAuthLifecycleRegisterLoginRefreshLogout(t *testing.T) {
	gin.SetMode(gin.TestMode)

	db := database.SetupTestDB(t)

	users := repositories.NewUserRepository(db)
	sessions := repositories.NewRefreshSessionRepository(db)
	verificationRepo := repositories.NewEmailVerificationRepository(db)
	resetRepo := repositories.NewPasswordResetRepository(db)

	tokenService, err := services.NewTokenService(
		"test-secret-with-at-least-32-characters-long",
	)
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	authService := services.NewAuthService(users, tokenService, sessions)
	verificationService := services.NewEmailVerificationService(
		users,
		verificationRepo,
	)
	resetService := services.NewPasswordResetService(users, resetRepo)

	authHandler := NewAuthHandler(
		authService,
		verificationService,
		resetService,
		true,
	)

	router := gin.New()
	RegisterAuthRoutes(router, authHandler, tokenService)

	email := fmt.Sprintf("lifecycle-%s@example.com", uuid.NewString())
	password := "test-password-123"
	displayName := "Lifecycle Test"

	// 1. Register a unique user through the HTTP endpoint.
	registerRecorder := performJSONRequest(
		t,
		router,
		http.MethodPost,
		"/api/v1/auth/register",
		map[string]string{
			"email":        email,
			"password":     password,
			"display_name": displayName,
		},
	)

	if registerRecorder.Code != http.StatusCreated {
		t.Fatalf("register: expected %d, got %d; body: %s",
			http.StatusCreated, registerRecorder.Code, registerRecorder.Body.String())
	}

	// 2. Log in and obtain a refresh token.
	loginRecorder := performJSONRequest(
		t,
		router,
		http.MethodPost,
		"/api/v1/auth/login",
		map[string]string{
			"email":    email,
			"password": password,
		},
	)

	if loginRecorder.Code != http.StatusOK {
		t.Fatalf("login: expected %d, got %d; body: %s",
			http.StatusOK, loginRecorder.Code, loginRecorder.Body.String())
	}

	loginResponse := decodeLifecycleResponse(t, loginRecorder)
	if loginResponse.AccessToken == "" || loginResponse.RefreshToken == "" {
		t.Fatal("login response must include access_token and refresh_token")
	}

	// 3. Refresh the session; the original refresh token is now consumed.
	refreshRecorder := performJSONRequest(
		t,
		router,
		http.MethodPost,
		"/api/v1/auth/refresh",
		map[string]string{
			"refresh_token": loginResponse.RefreshToken,
		},
	)

	if refreshRecorder.Code != http.StatusOK {
		t.Fatalf("refresh: expected %d, got %d; body: %s",
			http.StatusOK, refreshRecorder.Code, refreshRecorder.Body.String())
	}

	refreshResponse := decodeLifecycleResponse(t, refreshRecorder)
	if refreshResponse.AccessToken == "" || refreshResponse.RefreshToken == "" {
		t.Fatal("refresh response must include new access_token and refresh_token")
	}
	if refreshResponse.RefreshToken == loginResponse.RefreshToken {
		t.Fatal("refresh must issue a new refresh token")
	}

	// 4. Log out using the replacement refresh token.
	logoutBody := map[string]string{
		"refresh_token": refreshResponse.RefreshToken,
	}
	logoutRecorder := performJSONRequest(
		t,
		router,
		http.MethodPost,
		"/api/v1/auth/logout",
		logoutBody,
	)

	if logoutRecorder.Code != http.StatusNoContent {
		t.Fatalf("logout: expected %d, got %d; body: %s",
			http.StatusNoContent, logoutRecorder.Code, logoutRecorder.Body.String())
	}

	// 5. Repeating logout should be safe and return the same status.
	repeatedLogoutRecorder := performJSONRequest(
		t,
		router,
		http.MethodPost,
		"/api/v1/auth/logout",
		logoutBody,
	)

	if repeatedLogoutRecorder.Code != http.StatusNoContent {
		t.Fatalf("repeated logout: expected %d, got %d; body: %s",
			http.StatusNoContent, repeatedLogoutRecorder.Code,
			repeatedLogoutRecorder.Body.String())
	}

	// 6. The logged-out refresh token must no longer work.
	reuseRecorder := performJSONRequest(
		t,
		router,
		http.MethodPost,
		"/api/v1/auth/refresh",
		logoutBody,
	)

	if reuseRecorder.Code != http.StatusUnauthorized {
		t.Fatalf("refresh after logout: expected %d, got %d; body: %s",
			http.StatusUnauthorized, reuseRecorder.Code, reuseRecorder.Body.String())
	}

	reuseResponse := decodeLifecycleResponse(t, reuseRecorder)
	if reuseResponse.Error == nil ||
		reuseResponse.Error.Code != "INVALID_REFRESH_TOKEN" {
		t.Fatalf("expected INVALID_REFRESH_TOKEN error, got: %s",
			reuseRecorder.Body.String())
	}
}
