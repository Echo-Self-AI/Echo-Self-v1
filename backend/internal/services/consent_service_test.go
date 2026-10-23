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
)

func TestConsentServiceGrantAndValidation(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "consent-service-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Consent Service Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	repo := repositories.NewConsentRepository(db)
	service := NewConsentService(repo)
	meta := RequestMetadata{IPAddress: "192.0.2.1", UserAgent: "TestBrowser"}

	// 1. Success case
	result, err := service.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   PurposeVoiceCloning,
		PolicyVersion: "v1.0",
	}, meta)
	if err != nil {
		t.Fatalf("grant consent: %v", err)
	}
	if result.ConsentType != PurposeVoiceCloning {
		t.Errorf("expected purpose %s, got %s", PurposeVoiceCloning, result.ConsentType)
	}
	if result.Status != models.ConsentStatusGranted {
		t.Errorf("expected status 'granted', got '%s'", result.Status)
	}

	// 2. Reject unsupported purpose
	_, err = service.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   "unsupported_purpose",
		PolicyVersion: "v1.0",
	}, meta)
	if !errors.Is(err, ErrInvalidConsentPurpose) {
		t.Fatalf("expected ErrInvalidConsentPurpose, got %v", err)
	}

	// 3. Reject unsupported policy version
	_, err = service.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   PurposeFaceSynthesis,
		PolicyVersion: "v999.0",
	}, meta)
	if !errors.Is(err, ErrInvalidPolicyVersion) {
		t.Fatalf("expected ErrInvalidPolicyVersion, got %v", err)
	}

	// 4. Reject past expiration
	past := time.Now().Add(-1 * time.Hour)
	_, err = service.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   PurposeFaceSynthesis,
		PolicyVersion: "v1.0",
		ExpiresAt:     &past,
	}, meta)
	if err == nil {
		t.Fatal("expected error for past expiration date")
	}

	// 5. Reject unauthenticated/invalid user ID
	_, err = service.GrantConsent(ctx, "invalid-uuid", GrantConsentInput{
		ConsentType:   PurposeFaceSynthesis,
		PolicyVersion: "v1.0",
	}, meta)
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("expected ErrUnauthorized, got %v", err)
	}
}

func TestConsentServiceCheckConsentAuthorizationBoundary(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "consent-check-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Check Consent Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	repo := repositories.NewConsentRepository(db)
	service := NewConsentService(repo)
	meta := RequestMetadata{IPAddress: "127.0.0.1", UserAgent: "TestRunner"}

	// 1. Missing consent fails closed with ErrConsentNotGranted
	err := service.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          user.ID,
		ConsentType:     PurposeVoiceCloning,
		RequiredVersion: "v1.0",
	})
	if !errors.Is(err, ErrConsentNotGranted) {
		t.Fatalf("expected ErrConsentNotGranted for missing consent, got %v", err)
	}

	// 2. Grant consent
	grant, err := service.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   PurposeVoiceCloning,
		PolicyVersion: "v1.0",
	}, meta)
	if err != nil {
		t.Fatalf("grant consent: %v", err)
	}

	// 3. Valid active consent passes
	err = service.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          user.ID,
		ConsentType:     PurposeVoiceCloning,
		RequiredVersion: "v1.0",
	})
	if err != nil {
		t.Fatalf("expected CheckConsent to succeed, got %v", err)
	}

	// 4. Outdated required policy version fails closed
	err = service.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          user.ID,
		ConsentType:     PurposeVoiceCloning,
		RequiredVersion: "v2.0",
	})
	if !errors.Is(err, ErrConsentPolicyOutdated) {
		t.Fatalf("expected ErrConsentPolicyOutdated, got %v", err)
	}

	// 5. Consent for a different purpose fails closed
	err = service.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          user.ID,
		ConsentType:     PurposeFaceSynthesis,
		RequiredVersion: "v1.0",
	})
	if !errors.Is(err, ErrConsentNotGranted) {
		t.Fatalf("expected ErrConsentNotGranted for different purpose, got %v", err)
	}

	// 6. Revoking consent makes CheckConsent fail closed immediately
	if err := service.RevokeConsent(ctx, user.ID.String(), grant.ID, meta); err != nil {
		t.Fatalf("revoke consent: %v", err)
	}

	err = service.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          user.ID,
		ConsentType:     PurposeVoiceCloning,
		RequiredVersion: "v1.0",
	})
	if !errors.Is(err, ErrConsentNotGranted) {
		t.Fatalf("expected ErrConsentNotGranted after revocation, got %v", err)
	}

	// 7. Re-granting after revocation is allowed and passes check
	_, err = service.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   PurposeVoiceCloning,
		PolicyVersion: "v1.0",
	}, meta)
	if err != nil {
		t.Fatalf("re-grant consent: %v", err)
	}

	err = service.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          user.ID,
		ConsentType:     PurposeVoiceCloning,
		RequiredVersion: "v1.0",
	})
	if err != nil {
		t.Fatalf("expected CheckConsent to succeed after re-grant, got %v", err)
	}
}
