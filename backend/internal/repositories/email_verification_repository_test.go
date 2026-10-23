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

func TestEmailVerificationConsumesTokenOnce(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:              uuid.New(),
		Email:           "verify-" + uuid.NewString() + "@example.com",
		PasswordHash:    "test-hash",
		DisplayName:     "Verification Test",
		IsEmailVerified: false,
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	verification := &models.EmailVerification{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "verify-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}
	if err := db.WithContext(ctx).Create(verification).Error; err != nil {
		t.Fatalf("create verification token: %v", err)
	}

	repository := NewEmailVerificationRepository(db)

	if err := repository.VerifyAndConsume(
		ctx,
		verification.TokenHash,
		now,
	); err != nil {
		t.Fatalf("verify email: %v", err)
	}

	var updatedUser models.User
	if err := db.First(&updatedUser, "id = ?", user.ID).Error; err != nil {
		t.Fatalf("load user: %v", err)
	}
	if !updatedUser.IsEmailVerified {
		t.Fatal("user email was not marked verified")
	}

	var updatedToken models.EmailVerification
	if err := db.First(&updatedToken, "id = ?", verification.ID).Error; err != nil {
		t.Fatalf("load verification token: %v", err)
	}
	if updatedToken.UsedAt == nil {
		t.Fatal("verification token was not consumed")
	}

	err := repository.VerifyAndConsume(ctx, verification.TokenHash, time.Now())
	if !errors.Is(err, ErrInvalidVerificationRecord) {
		t.Fatalf("expected token reuse to be rejected, got %v", err)
	}
}

func TestEmailVerificationRejectsExpiredToken(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:              uuid.New(),
		Email:           "verify-exp-" + uuid.NewString() + "@example.com",
		PasswordHash:    "test-hash",
		DisplayName:     "Verification Exp Test",
		IsEmailVerified: false,
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	verification := &models.EmailVerification{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "verify-exp-" + uuid.NewString(),
		ExpiresAt: now.Add(-1 * time.Hour), // expired in past
	}
	if err := db.WithContext(ctx).Create(verification).Error; err != nil {
		t.Fatalf("create verification token: %v", err)
	}

	repository := NewEmailVerificationRepository(db)

	err := repository.VerifyAndConsume(ctx, verification.TokenHash, now)
	if !errors.Is(err, ErrInvalidVerificationRecord) {
		t.Fatalf("expected expired verification token to be rejected, got %v", err)
	}
}
