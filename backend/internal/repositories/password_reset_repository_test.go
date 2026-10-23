package repositories

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/google/uuid"
)

func TestPasswordResetUpdatesPasswordAndRevokesSessions(t *testing.T) {
	db := database.SetupTestDB(t)

	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "reset-" + uuid.NewString() + "@example.com",
		PasswordHash: "old-hash",
		DisplayName:  "Reset Test",
	}

	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "session-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}

	if err := db.WithContext(ctx).Create(session).Error; err != nil {
		t.Fatalf("create refresh session: %v", err)
	}

	reset := &models.PasswordReset{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "reset-" + uuid.NewString(),
		ExpiresAt: now.Add(30 * time.Minute),
	}

	if err := db.WithContext(ctx).Create(reset).Error; err != nil {
		t.Fatalf("create reset token: %v", err)
	}

	repository := NewPasswordResetRepository(db)

	if err := repository.ResetPassword(
		ctx,
		reset.TokenHash,
		now,
		"new-hash",
	); err != nil {
		t.Fatalf("reset password: %v", err)
	}

	var updatedUser models.User
	if err := db.First(&updatedUser, "id = ?", user.ID).Error; err != nil {
		t.Fatalf("load updated user: %v", err)
	}
	if updatedUser.PasswordHash != "new-hash" {
		t.Fatal("password hash was not updated")
	}

	var updatedReset models.PasswordReset
	if err := db.First(&updatedReset, "id = ?", reset.ID).Error; err != nil {
		t.Fatalf("load reset token: %v", err)
	}
	if updatedReset.UsedAt == nil {
		t.Fatal("reset token was not consumed")
	}

	var updatedSession models.RefreshSession
	if err := db.First(&updatedSession, "id = ?", session.ID).Error; err != nil {
		t.Fatalf("load refresh session: %v", err)
	}
	if updatedSession.RevokedAt == nil {
		t.Fatal("refresh session was not revoked")
	}

	err := repository.ResetPassword(ctx, reset.TokenHash, time.Now(), "another-hash")
	if !errors.Is(err, ErrInvalidPasswordReset) {
		t.Fatalf("expected token reuse to be rejected, got %v", err)
	}
}

func TestPasswordResetRejectsExpiredToken(t *testing.T) {
	db := database.SetupTestDB(t)

	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "reset-expired-" + uuid.NewString() + "@example.com",
		PasswordHash: "current-hash",
		DisplayName:  "Expired Reset Test",
	}

	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	reset := &models.PasswordReset{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "reset-exp-" + uuid.NewString(),
		ExpiresAt: now.Add(-10 * time.Minute), // expired in past
	}

	if err := db.WithContext(ctx).Create(reset).Error; err != nil {
		t.Fatalf("create reset token: %v", err)
	}

	repository := NewPasswordResetRepository(db)

	err := repository.ResetPassword(ctx, reset.TokenHash, now, "new-hash")
	if !errors.Is(err, ErrInvalidPasswordReset) {
		t.Fatalf("expected expired reset token to be rejected, got %v", err)
	}
}
