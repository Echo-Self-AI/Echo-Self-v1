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

func TestConsentRepositoryCreateAndFind(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "consent-repo-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Consent Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	repo := NewConsentRepository(db)

	consent := &models.ConsentRecord{
		ID:            uuid.New(),
		UserID:        user.ID,
		ConsentType:   "voice_cloning",
		Status:        models.ConsentStatusGranted,
		PolicyVersion: "v1.0",
		GrantedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	audit := &models.ConsentAuditLog{
		ID:        uuid.New(),
		Action:    models.ConsentStatusGranted,
		IPAddress: "127.0.0.1",
		UserAgent: "TestAgent",
		CreatedAt: now,
	}

	if err := repo.CreateConsent(ctx, consent, audit); err != nil {
		t.Fatalf("create consent: %v", err)
	}

	// 1. Find by ID and Owner
	found, err := repo.FindByIDAndUserID(ctx, consent.ID, user.ID)
	if err != nil {
		t.Fatalf("find consent by ID and owner: %v", err)
	}
	if found == nil || found.ID != consent.ID {
		t.Fatal("expected consent record to be found")
	}

	// 2. Audit logs exist
	logs, err := repo.ListAuditLogsByConsentID(ctx, consent.ID, user.ID)
	if err != nil {
		t.Fatalf("list audit logs: %v", err)
	}
	if len(logs) != 1 {
		t.Fatalf("expected 1 audit log, got %d", len(logs))
	}
	if logs[0].Action != models.ConsentStatusGranted {
		t.Errorf("expected audit action 'granted', got '%s'", logs[0].Action)
	}
}

func TestConsentRepositoryPreventDuplicateActive(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "consent-dup-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Duplicate Consent Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	repo := NewConsentRepository(db)

	consent1 := &models.ConsentRecord{
		ID:            uuid.New(),
		UserID:        user.ID,
		ConsentType:   "face_synthesis",
		Status:        models.ConsentStatusGranted,
		PolicyVersion: "v1.0",
		GrantedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	audit1 := &models.ConsentAuditLog{
		ID:        uuid.New(),
		Action:    models.ConsentStatusGranted,
		CreatedAt: now,
	}

	if err := repo.CreateConsent(ctx, consent1, audit1); err != nil {
		t.Fatalf("create consent 1: %v", err)
	}

	// Attempting a second active grant for the same type fails
	consent2 := &models.ConsentRecord{
		ID:            uuid.New(),
		UserID:        user.ID,
		ConsentType:   "face_synthesis",
		Status:        models.ConsentStatusGranted,
		PolicyVersion: "v1.0",
		GrantedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	audit2 := &models.ConsentAuditLog{
		ID:        uuid.New(),
		Action:    models.ConsentStatusGranted,
		CreatedAt: now,
	}

	err := repo.CreateConsent(ctx, consent2, audit2)
	if !errors.Is(err, ErrConsentAlreadyActive) {
		t.Fatalf("expected ErrConsentAlreadyActive, got %v", err)
	}
}

func TestConsentRepositoryRevokeAndIdempotency(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "consent-revoke-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Revoke Consent Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	repo := NewConsentRepository(db)

	consent := &models.ConsentRecord{
		ID:            uuid.New(),
		UserID:        user.ID,
		ConsentType:   "voice_data_processing",
		Status:        models.ConsentStatusGranted,
		PolicyVersion: "v1.0",
		GrantedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	grantAudit := &models.ConsentAuditLog{
		ID:        uuid.New(),
		Action:    models.ConsentStatusGranted,
		CreatedAt: now,
	}
	if err := repo.CreateConsent(ctx, consent, grantAudit); err != nil {
		t.Fatalf("create consent: %v", err)
	}

	// Revoke
	revokeTime := now.Add(1 * time.Minute)
	revokeAudit := &models.ConsentAuditLog{
		ID:        uuid.New(),
		Action:    models.ConsentStatusRevoked,
		CreatedAt: revokeTime,
	}
	if err := repo.RevokeConsent(ctx, consent.ID, user.ID, revokeTime, revokeAudit); err != nil {
		t.Fatalf("revoke consent: %v", err)
	}

	// Verify revoked state
	revoked, err := repo.FindByIDAndUserID(ctx, consent.ID, user.ID)
	if err != nil || revoked == nil {
		t.Fatalf("load revoked consent: %v", err)
	}
	if revoked.Status != models.ConsentStatusRevoked {
		t.Errorf("expected status 'revoked', got '%s'", revoked.Status)
	}
	if revoked.RevokedAt == nil {
		t.Fatal("expected RevokedAt to be set")
	}

	// Verify audit logs have 2 records (granted, revoked)
	logs, err := repo.ListAuditLogsByConsentID(ctx, consent.ID, user.ID)
	if err != nil {
		t.Fatalf("list audit logs: %v", err)
	}
	if len(logs) != 2 {
		t.Fatalf("expected 2 audit logs, got %d", len(logs))
	}
	if logs[1].Action != models.ConsentStatusRevoked {
		t.Errorf("expected second audit log action 'revoked', got '%s'", logs[1].Action)
	}

	// Repeated revocation is safe and idempotent
	repeatAudit := &models.ConsentAuditLog{ID: uuid.New(), CreatedAt: revokeTime}
	if err := repo.RevokeConsent(ctx, consent.ID, user.ID, revokeTime, repeatAudit); err != nil {
		t.Fatalf("repeat revocation failed: %v", err)
	}
}

func TestConsentRepositoryOwnershipIsolation(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	userA := &models.User{
		ID:           uuid.New(),
		Email:        "user-a-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "User A",
	}
	userB := &models.User{
		ID:           uuid.New(),
		Email:        "user-b-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "User B",
	}
	if err := db.WithContext(ctx).Create(userA).Error; err != nil {
		t.Fatalf("create user A: %v", err)
	}
	if err := db.WithContext(ctx).Create(userB).Error; err != nil {
		t.Fatalf("create user B: %v", err)
	}

	repo := NewConsentRepository(db)

	consentA := &models.ConsentRecord{
		ID:            uuid.New(),
		UserID:        userA.ID,
		ConsentType:   "voice_cloning",
		Status:        models.ConsentStatusGranted,
		PolicyVersion: "v1.0",
		GrantedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	auditA := &models.ConsentAuditLog{
		ID:        uuid.New(),
		Action:    models.ConsentStatusGranted,
		CreatedAt: now,
	}
	if err := repo.CreateConsent(ctx, consentA, auditA); err != nil {
		t.Fatalf("create consent for user A: %v", err)
	}

	// User B cannot find User A's consent
	found, err := repo.FindByIDAndUserID(ctx, consentA.ID, userB.ID)
	if err != nil {
		t.Fatalf("error querying with wrong user ID: %v", err)
	}
	if found != nil {
		t.Fatal("expected User B to NOT find User A's consent record")
	}

	// User B cannot revoke User A's consent
	auditB := &models.ConsentAuditLog{ID: uuid.New(), CreatedAt: now}
	err = repo.RevokeConsent(ctx, consentA.ID, userB.ID, now, auditB)
	if !errors.Is(err, ErrConsentNotFound) {
		t.Fatalf("expected ErrConsentNotFound when User B tries to revoke User A's consent, got %v", err)
	}
}
