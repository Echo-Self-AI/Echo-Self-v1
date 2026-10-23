package handlers

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func TestUserProfileLifecycle(t *testing.T) {
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
	verificationService := services.NewEmailVerificationService(users, verificationRepo)
	resetService := services.NewPasswordResetService(users, resetRepo)
	userService := services.NewUserService(users)

	authHandler := NewAuthHandler(authService, verificationService, resetService, true)
	userHandler := NewUserHandler(userService)

	router := gin.New()
	RegisterAuthRoutes(router, authHandler, tokenService)

	usersGroup := router.Group("/api/v1/users", middleware.AuthRequired(tokenService))
	{
		usersGroup.GET("/me", userHandler.Me)
		usersGroup.PATCH("/me", userHandler.UpdateProfile)
		usersGroup.DELETE("/me", userHandler.DeleteAccount)
	}

	email := fmt.Sprintf("profile-lifecycle-%s@example.com", uuid.NewString())
	password := "secure-password-123"
	displayName := "Original Name"

	// 1. Register & Login
	regRec := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/register", map[string]string{
		"email":        email,
		"password":     password,
		"display_name": displayName,
	})
	if regRec.Code != http.StatusCreated {
		t.Fatalf("register failed: %d", regRec.Code)
	}

	loginRec := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/login", map[string]string{
		"email":    email,
		"password": password,
	})
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login failed: %d", loginRec.Code)
	}
	loginResp := decodeLifecycleResponse(t, loginRec)
	accessToken := loginResp.AccessToken
	refreshToken := loginResp.RefreshToken

	// 2. GET /api/v1/users/me without token -> 401
	unauthReq := httptest.NewRequest(http.MethodGet, "/api/v1/users/me", nil)
	unauthRec := httptest.NewRecorder()
	router.ServeHTTP(unauthRec, unauthReq)
	if unauthRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for unauthenticated request, got %d", unauthRec.Code)
	}

	// 3. GET /api/v1/users/me with token -> 200
	getReq := httptest.NewRequest(http.MethodGet, "/api/v1/users/me", nil)
	getReq.Header.Set("Authorization", "Bearer "+accessToken)
	getRec := httptest.NewRecorder()
	router.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for authenticated profile get, got %d; body: %s", getRec.Code, getRec.Body.String())
	}
	getResp := decodeLifecycleResponse(t, getRec)
	if getResp.DisplayName != displayName {
		t.Errorf("expected display name %s, got %s", displayName, getResp.DisplayName)
	}
	if getResp.Email != email {
		t.Errorf("expected email %s, got %s", email, getResp.Email)
	}

	// 4. PATCH /api/v1/users/me -> 200
	patchRec := httptest.NewRecorder()
	patchReqBody := `{"display_name": "Updated Profile Name"}`
	patchReq := httptest.NewRequest(http.MethodPatch, "/api/v1/users/me", strings.NewReader(patchReqBody))
	patchReq.Header.Set("Authorization", "Bearer "+accessToken)
	patchReq.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(patchRec, patchReq)
	if patchRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for profile update, got %d; body: %s", patchRec.Code, patchRec.Body.String())
	}
	patchResp := decodeLifecycleResponse(t, patchRec)
	if patchResp.DisplayName != "Updated Profile Name" {
		t.Errorf("expected updated display name 'Updated Profile Name', got '%s'", patchResp.DisplayName)
	}

	// 5. PATCH with invalid/empty display name -> 400
	badPatchRec := httptest.NewRecorder()
	badPatchReq := httptest.NewRequest(http.MethodPatch, "/api/v1/users/me", strings.NewReader(`{"display_name": "   "}`))
	badPatchReq.Header.Set("Authorization", "Bearer "+accessToken)
	badPatchReq.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(badPatchRec, badPatchReq)
	if badPatchRec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty display name, got %d; body: %s", badPatchRec.Code, badPatchRec.Body.String())
	}

	// 6. DELETE /api/v1/users/me -> 204 No Content
	delRec := httptest.NewRecorder()
	delReq := httptest.NewRequest(http.MethodDelete, "/api/v1/users/me", nil)
	delReq.Header.Set("Authorization", "Bearer "+accessToken)
	router.ServeHTTP(delRec, delReq)
	if delRec.Code != http.StatusNoContent {
		t.Fatalf("expected 204 for account deletion, got %d; body: %s", delRec.Code, delRec.Body.String())
	}

	// 7. Subsequent GET /api/v1/users/me with the same access token -> 401 USER_NOT_FOUND
	afterDelRec := httptest.NewRecorder()
	afterDelReq := httptest.NewRequest(http.MethodGet, "/api/v1/users/me", nil)
	afterDelReq.Header.Set("Authorization", "Bearer "+accessToken)
	router.ServeHTTP(afterDelRec, afterDelReq)
	if afterDelRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 after deletion, got %d; body: %s", afterDelRec.Code, afterDelRec.Body.String())
	}
	afterDelResp := decodeLifecycleResponse(t, afterDelRec)
	if afterDelResp.Error == nil || afterDelResp.Error.Code != "USER_NOT_FOUND" {
		t.Fatalf("expected USER_NOT_FOUND, got: %s", afterDelRec.Body.String())
	}

	// 8. Subsequent refresh attempt with refresh token -> 401 INVALID_REFRESH_TOKEN (session was revoked!)
	refRec := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/refresh", map[string]string{
		"refresh_token": refreshToken,
	})
	if refRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 refreshing after deletion, got %d", refRec.Code)
	}
}
