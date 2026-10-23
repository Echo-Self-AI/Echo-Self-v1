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

func TestConsentHTTPLifecycleAndSecurity(t *testing.T) {
	gin.SetMode(gin.TestMode)

	db := database.SetupTestDB(t)

	users := repositories.NewUserRepository(db)
	sessions := repositories.NewRefreshSessionRepository(db)
	verificationRepo := repositories.NewEmailVerificationRepository(db)
	resetRepo := repositories.NewPasswordResetRepository(db)
	consentRepo := repositories.NewConsentRepository(db)

	tokenService, err := services.NewTokenService(
		"test-secret-with-at-least-32-characters-long",
	)
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	authService := services.NewAuthService(users, tokenService, sessions)
	verificationService := services.NewEmailVerificationService(users, verificationRepo)
	resetService := services.NewPasswordResetService(users, resetRepo)
	consentService := services.NewConsentService(consentRepo)

	authHandler := NewAuthHandler(authService, verificationService, resetService, true)
	consentHandler := NewConsentHandler(consentService)

	router := gin.New()
	RegisterAuthRoutes(router, authHandler, tokenService)

	consentsGroup := router.Group("/api/v1/consents", middleware.AuthRequired(tokenService))
	{
		consentsGroup.POST("", consentHandler.Grant)
		consentsGroup.GET("", consentHandler.List)
		consentsGroup.GET("/:id", consentHandler.Get)
		consentsGroup.POST("/:id/revoke", consentHandler.Revoke)
		consentsGroup.GET("/:id/audit", consentHandler.AuditLogs)
	}

	// Register two separate users to test ownership boundaries
	userAEmail := fmt.Sprintf("consent-a-%s@example.com", uuid.NewString())
	userBEmail := fmt.Sprintf("consent-b-%s@example.com", uuid.NewString())
	password := "secure-pass-123"

	regA := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/register", map[string]string{
		"email": userAEmail, "password": password, "display_name": "User A",
	})
	if regA.Code != http.StatusCreated {
		t.Fatalf("register A: %d", regA.Code)
	}
	loginA := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/login", map[string]string{
		"email": userAEmail, "password": password,
	})
	tokenA := decodeLifecycleResponse(t, loginA).AccessToken

	regB := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/register", map[string]string{
		"email": userBEmail, "password": password, "display_name": "User B",
	})
	if regB.Code != http.StatusCreated {
		t.Fatalf("register B: %d", regB.Code)
	}
	loginB := performJSONRequest(t, router, http.MethodPost, "/api/v1/auth/login", map[string]string{
		"email": userBEmail, "password": password,
	})
	tokenB := decodeLifecycleResponse(t, loginB).AccessToken

	// 1. Unauthenticated request to /api/v1/consents -> 401
	unauthReq := httptest.NewRequest(http.MethodGet, "/api/v1/consents", nil)
	unauthRec := httptest.NewRecorder()
	router.ServeHTTP(unauthRec, unauthReq)
	if unauthRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for unauthenticated request, got %d", unauthRec.Code)
	}

	// 2. Grant consent for User A -> 201 Created
	grantBody := `{"consent_type": "voice_cloning", "policy_version": "v1.0"}`
	grantReq := httptest.NewRequest(http.MethodPost, "/api/v1/consents", strings.NewReader(grantBody))
	grantReq.Header.Set("Authorization", "Bearer "+tokenA)
	grantReq.Header.Set("Content-Type", "application/json")
	grantRec := httptest.NewRecorder()
	router.ServeHTTP(grantRec, grantReq)
	if grantRec.Code != http.StatusCreated {
		t.Fatalf("expected 201 for granting consent, got %d; body: %s", grantRec.Code, grantRec.Body.String())
	}
	grantResp := decodeLifecycleResponse(t, grantRec)
	consentIDA := grantResp.ID
	if consentIDA == "" {
		t.Fatal("expected consent ID in response")
	}

	// 3. Duplicate active grant for User A -> 409 Conflict
	dupReq := httptest.NewRequest(http.MethodPost, "/api/v1/consents", strings.NewReader(grantBody))
	dupReq.Header.Set("Authorization", "Bearer "+tokenA)
	dupReq.Header.Set("Content-Type", "application/json")
	dupRec := httptest.NewRecorder()
	router.ServeHTTP(dupRec, dupReq)
	if dupRec.Code != http.StatusConflict {
		t.Fatalf("expected 409 for duplicate active grant, got %d; body: %s", dupRec.Code, dupRec.Body.String())
	}

	// 4. Listing consents for User A returns 1 record; for User B returns 0 records
	listReqA := httptest.NewRequest(http.MethodGet, "/api/v1/consents", nil)
	listReqA.Header.Set("Authorization", "Bearer "+tokenA)
	listRecA := httptest.NewRecorder()
	router.ServeHTTP(listRecA, listReqA)
	if listRecA.Code != http.StatusOK {
		t.Fatalf("list A: %d", listRecA.Code)
	}

	listReqB := httptest.NewRequest(http.MethodGet, "/api/v1/consents", nil)
	listReqB.Header.Set("Authorization", "Bearer "+tokenB)
	listRecB := httptest.NewRecorder()
	router.ServeHTTP(listRecB, listReqB)
	if listRecB.Code != http.StatusOK {
		t.Fatalf("list B: %d", listRecB.Code)
	}
	if strings.Contains(listRecB.Body.String(), consentIDA) {
		t.Fatal("User B should not see User A's consent ID in listing")
	}

	// 5. User B attempts to access User A's consent record -> 404 Not Found
	crossGetReq := httptest.NewRequest(http.MethodGet, "/api/v1/consents/"+consentIDA, nil)
	crossGetReq.Header.Set("Authorization", "Bearer "+tokenB)
	crossGetRec := httptest.NewRecorder()
	router.ServeHTTP(crossGetRec, crossGetReq)
	if crossGetRec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 when User B gets User A's consent, got %d", crossGetRec.Code)
	}

	// 6. User B attempts to revoke User A's consent record -> 404 Not Found
	crossRevReq := httptest.NewRequest(http.MethodPost, "/api/v1/consents/"+consentIDA+"/revoke", nil)
	crossRevReq.Header.Set("Authorization", "Bearer "+tokenB)
	crossRevRec := httptest.NewRecorder()
	router.ServeHTTP(crossRevRec, crossRevReq)
	if crossRevRec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 when User B revokes User A's consent, got %d", crossRevRec.Code)
	}

	// 7. User A revokes own consent -> 200 OK
	revReq := httptest.NewRequest(http.MethodPost, "/api/v1/consents/"+consentIDA+"/revoke", nil)
	revReq.Header.Set("Authorization", "Bearer "+tokenA)
	revRec := httptest.NewRecorder()
	router.ServeHTTP(revRec, revReq)
	if revRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for revoking consent, got %d; body: %s", revRec.Code, revRec.Body.String())
	}

	// 8. Repeated revocation by User A is idempotent -> 200 OK
	repeatRevReq := httptest.NewRequest(http.MethodPost, "/api/v1/consents/"+consentIDA+"/revoke", nil)
	repeatRevReq.Header.Set("Authorization", "Bearer "+tokenA)
	repeatRevRec := httptest.NewRecorder()
	router.ServeHTTP(repeatRevRec, repeatRevReq)
	if repeatRevRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for repeated revocation, got %d", repeatRevRec.Code)
	}

	// 9. User A retrieves audit logs -> 200 OK with history
	auditReq := httptest.NewRequest(http.MethodGet, "/api/v1/consents/"+consentIDA+"/audit", nil)
	auditReq.Header.Set("Authorization", "Bearer "+tokenA)
	auditRec := httptest.NewRecorder()
	router.ServeHTTP(auditRec, auditReq)
	if auditRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for audit logs, got %d; body: %s", auditRec.Code, auditRec.Body.String())
	}
	if !strings.Contains(auditRec.Body.String(), "granted") || !strings.Contains(auditRec.Body.String(), "revoked") {
		t.Errorf("expected audit logs to contain 'granted' and 'revoked', got: %s", auditRec.Body.String())
	}
}
