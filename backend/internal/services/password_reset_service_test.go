package services

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestResetPasswordRejectsInvalidTokenLength(t *testing.T) {
	service := &PasswordResetService{}

	err := service.ResetPassword(
		context.Background(),
		"short-token",
		"NewSecurePass123!",
	)

	if !errors.Is(err, ErrInvalidPasswordResetToken) {
		t.Fatalf("expected invalid token error, got %v", err)
	}
}

func TestResetPasswordRejectsShortPassword(t *testing.T) {
	service := &PasswordResetService{}

	err := service.ResetPassword(
		context.Background(),
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"short",
	)

	if !errors.Is(err, ErrInvalidNewPassword) {
		t.Fatalf("expected invalid password error, got %v", err)
	}
}

func TestResetPasswordRejectsPasswordOver72Bytes(t *testing.T) {
	service := &PasswordResetService{}

	longPassword := strings.Repeat("a", 73)

	if len(longPassword) != 73 {
		t.Fatalf("test setup error: expected 73 bytes, got %d", len(longPassword))
	}

	err := service.ResetPassword(
		context.Background(),
		strings.Repeat("a", 64),
		longPassword,
	)

	if !errors.Is(err, ErrInvalidNewPassword) {
		t.Fatalf("expected invalid password error, got %v", err)
	}
}
