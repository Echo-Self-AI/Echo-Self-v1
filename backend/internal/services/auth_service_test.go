package services

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

func setupLogoutTest(t *testing.T) (*AuthService, *gorm.DB, *models.User) {
	t.Helper()

	db := database.SetupTestDB(t)

	user := &models.User{
		ID:           uuid.New(),
		Email:        "logout-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Logout Test",
	}

	if err := db.Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	tokens, err := NewTokenService(
		"test-secret-that-is-at-least-32-bytes-long",
	)
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	service := NewAuthService(
		repositories.NewUserRepository(db),
		tokens,
		repositories.NewRefreshSessionRepository(db),
	)

	return service, db, user
}

func newTestRefreshToken() string {
	return uuid.NewString() + uuid.NewString()[:28]
}

func TestAuthServiceLogoutRevokesRefreshSession(t *testing.T) {
	service, db, user := setupLogoutTest(t)
	ctx := context.Background()
	now := time.Now()

	rawToken := newTestRefreshToken()

	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: hashRefreshToken(rawToken),
		ExpiresAt: now.Add(24 * time.Hour),
	}

	if err := db.Create(session).Error; err != nil {
		t.Fatalf("create refresh session: %v", err)
	}

	if err := service.Logout(ctx, rawToken); err != nil {
		t.Fatalf("logout: %v", err)
	}

	var saved models.RefreshSession
	if err := db.First(&saved, "id = ?", session.ID).Error; err != nil {
		t.Fatalf("load refresh session: %v", err)
	}

	if saved.RevokedAt == nil {
		t.Fatal("expected logout to revoke the refresh session")
	}
}

func TestAuthServiceLogoutIgnoresMalformedTokens(t *testing.T) {
	service, _, _ := setupLogoutTest(t)

	tests := []struct {
		name  string
		token string
	}{
		{name: "empty token", token: ""},
		{name: "short token", token: "short"},
		{name: "long token", token: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdefx"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := service.Logout(context.Background(), tt.token); err != nil {
				t.Fatalf("expected malformed token to be ignored, got: %v", err)
			}
		})
	}
}

func TestAuthServiceLogoutCanBeRepeated(t *testing.T) {
	service, db, user := setupLogoutTest(t)
	ctx := context.Background()

	rawToken := newTestRefreshToken()

	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: hashRefreshToken(rawToken),
		ExpiresAt: time.Now().Add(24 * time.Hour),
	}

	if err := db.Create(session).Error; err != nil {
		t.Fatalf("create refresh session: %v", err)
	}

	for i := 0; i < 2; i++ {
		if err := service.Logout(ctx, rawToken); err != nil {
			t.Fatalf("logout attempt %d: %v", i+1, err)
		}
	}

	var saved models.RefreshSession
	if err := db.First(&saved, "id = ?", session.ID).Error; err != nil {
		t.Fatalf("load refresh session: %v", err)
	}

	if saved.RevokedAt == nil {
		t.Fatal("expected refresh session to remain revoked")
	}
}

func TestAuthServiceLogoutPropagatesRepositoryErrors(t *testing.T) {
	service, _, _ := setupLogoutTest(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	rawToken := newTestRefreshToken()

	err := service.Logout(ctx, rawToken)
	if err == nil {
		t.Fatal("expected cancelled context to produce a repository error")
	}

	if errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("did not expect record-not-found error: %v", err)
	}
}
