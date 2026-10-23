package services

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
)

func TestUserServiceGetProfile(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()

	userRepo := repositories.NewUserRepository(db)
	userService := NewUserService(userRepo)

	user := &models.User{
		ID:              uuid.New(),
		Email:           "getprofile-" + uuid.NewString() + "@example.com",
		PasswordHash:    "secret-hash",
		DisplayName:     "Original Name",
		IsEmailVerified: true,
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	// 1. Success case
	profile, err := userService.GetProfile(ctx, user.ID.String())
	if err != nil {
		t.Fatalf("get profile: %v", err)
	}
	if profile.ID != user.ID.String() {
		t.Errorf("expected ID %s, got %s", user.ID.String(), profile.ID)
	}
	if profile.Email != user.Email {
		t.Errorf("expected email %s, got %s", user.Email, profile.Email)
	}
	if profile.DisplayName != user.DisplayName {
		t.Errorf("expected display name %s, got %s", user.DisplayName, profile.DisplayName)
	}
	if !profile.IsEmailVerified {
		t.Error("expected IsEmailVerified to be true")
	}

	// 2. Non-existent user
	_, err = userService.GetProfile(ctx, uuid.NewString())
	if !errors.Is(err, ErrUserNotFound) {
		t.Fatalf("expected ErrUserNotFound, got %v", err)
	}

	// 3. Invalid UUID
	_, err = userService.GetProfile(ctx, "invalid-uuid")
	if !errors.Is(err, ErrUserNotFound) {
		t.Fatalf("expected ErrUserNotFound for invalid UUID, got %v", err)
	}
}

func TestUserServiceUpdateProfile(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()

	userRepo := repositories.NewUserRepository(db)
	userService := NewUserService(userRepo)

	user := &models.User{
		ID:           uuid.New(),
		Email:        "updateprofile-" + uuid.NewString() + "@example.com",
		PasswordHash: "secret-hash",
		DisplayName:  "Before Update",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	// 1. Successful update
	updated, err := userService.UpdateProfile(ctx, user.ID.String(), UpdateProfileInput{
		DisplayName: "  Updated Display Name  ",
	})
	if err != nil {
		t.Fatalf("update profile: %v", err)
	}
	if updated.DisplayName != "Updated Display Name" {
		t.Errorf("expected trimmed name 'Updated Display Name', got '%s'", updated.DisplayName)
	}

	// 2. Reject empty name
	_, err = userService.UpdateProfile(ctx, user.ID.String(), UpdateProfileInput{
		DisplayName: "    ",
	})
	if !errors.Is(err, ErrInvalidProfile) {
		t.Fatalf("expected ErrInvalidProfile for empty name, got %v", err)
	}

	// 3. Reject name exceeding 100 runes
	_, err = userService.UpdateProfile(ctx, user.ID.String(), UpdateProfileInput{
		DisplayName: strings.Repeat("a", 101),
	})
	if !errors.Is(err, ErrInvalidProfile) {
		t.Fatalf("expected ErrInvalidProfile for >100 runes, got %v", err)
	}

	// 4. Reject non-existent user
	_, err = userService.UpdateProfile(ctx, uuid.NewString(), UpdateProfileInput{
		DisplayName: "Valid Name",
	})
	if !errors.Is(err, ErrUserNotFound) {
		t.Fatalf("expected ErrUserNotFound, got %v", err)
	}
}

func TestUserServiceDeleteAccount(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	userRepo := repositories.NewUserRepository(db)
	userService := NewUserService(userRepo)

	user := &models.User{
		ID:           uuid.New(),
		Email:        "deleteaccount-" + uuid.NewString() + "@example.com",
		PasswordHash: "secret-hash",
		DisplayName:  "Delete Me",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	// Create associated active refresh session
	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "del-sess-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}
	if err := db.WithContext(ctx).Create(session).Error; err != nil {
		t.Fatalf("create refresh session: %v", err)
	}

	// Create associated pending email verification
	verification := &models.EmailVerification{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "del-verif-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}
	if err := db.WithContext(ctx).Create(verification).Error; err != nil {
		t.Fatalf("create verification: %v", err)
	}

	// Create associated pending password reset
	reset := &models.PasswordReset{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "del-reset-" + uuid.NewString(),
		ExpiresAt: now.Add(30 * time.Minute),
	}
	if err := db.WithContext(ctx).Create(reset).Error; err != nil {
		t.Fatalf("create password reset: %v", err)
	}

	// Delete account
	if err := userService.DeleteAccount(ctx, user.ID.String()); err != nil {
		t.Fatalf("delete account: %v", err)
	}

	// Verify user is soft-deleted
	var fetched models.User
	if err := db.First(&fetched, "id = ?", user.ID).Error; err == nil {
		t.Fatal("expected user to not be found by normal query after soft delete")
	}

	// Verify user exists in Unscoped with DeletedAt set
	var unscoped models.User
	if err := db.Unscoped().First(&unscoped, "id = ?", user.ID).Error; err != nil {
		t.Fatalf("load unscoped user: %v", err)
	}
	if !unscoped.DeletedAt.Valid {
		t.Fatal("expected DeletedAt to be valid")
	}

	// Verify refresh session revoked
	var savedSession models.RefreshSession
	if err := db.First(&savedSession, "id = ?", session.ID).Error; err != nil {
		t.Fatalf("load session: %v", err)
	}
	if savedSession.RevokedAt == nil {
		t.Fatal("expected refresh session to be revoked")
	}

	// Verify email verification consumed
	var savedVerification models.EmailVerification
	if err := db.First(&savedVerification, "id = ?", verification.ID).Error; err != nil {
		t.Fatalf("load verification: %v", err)
	}
	if savedVerification.UsedAt == nil {
		t.Fatal("expected email verification to be invalidated")
	}

	// Verify password reset consumed
	var savedReset models.PasswordReset
	if err := db.First(&savedReset, "id = ?", reset.ID).Error; err != nil {
		t.Fatalf("load reset: %v", err)
	}
	if savedReset.UsedAt == nil {
		t.Fatal("expected password reset to be invalidated")
	}

	// Subsequent delete call returns ErrUserNotFound
	if err := userService.DeleteAccount(ctx, user.ID.String()); !errors.Is(err, ErrUserNotFound) {
		t.Fatalf("expected ErrUserNotFound on already deleted account, got %v", err)
	}
}
