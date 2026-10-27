package repositories

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

func TestRefreshSessionRotationRejectsReuse(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "refresh-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Refresh Test",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	oldSession := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "old-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}

	if err := db.WithContext(ctx).Create(oldSession).Error; err != nil {
		t.Fatalf("create old session: %v", err)
	}

	replacement := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "new-" + uuid.NewString(),
		ExpiresAt: now.Add(30 * 24 * time.Hour),
	}

	repository := NewRefreshSessionRepository(db)

	if err := repository.Rotate(
		ctx,
		oldSession.TokenHash,
		replacement,
		now,
	); err != nil {
		t.Fatalf("rotate refresh session: %v", err)
	}

	var consumed models.RefreshSession
	if err := db.First(&consumed, "id = ?", oldSession.ID).Error; err != nil {
		t.Fatalf("load old session: %v", err)
	}
	if consumed.RevokedAt == nil {
		t.Fatal("old refresh session was not revoked")
	}

	var savedReplacement models.RefreshSession
	if err := db.First(&savedReplacement, "id = ?", replacement.ID).Error; err != nil {
		t.Fatalf("load replacement session: %v", err)
	}

	err := repository.Rotate(ctx, oldSession.TokenHash, &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "reused-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}, now)

	if !errors.Is(err, ErrTokenReuseDetected) {
		t.Fatalf("expected ErrTokenReuseDetected on reuse, got %v", err)
	}

	// Verify all user sessions were revoked
	var activeCount int64
	db.Model(&models.RefreshSession{}).Where("user_id = ? AND revoked_at IS NULL", user.ID).Count(&activeCount)
	if activeCount != 0 {
		t.Fatalf("expected all sessions revoked on reuse, got %d active", activeCount)
	}
}

func TestRefreshSessionConcurrentRotationAllowsOnlyOneSuccess(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "refresh-race-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Refresh Race Test",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	oldSession := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "race-old-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}
	if err := db.WithContext(ctx).Create(oldSession).Error; err != nil {
		t.Fatalf("create old session: %v", err)
	}

	repository := NewRefreshSessionRepository(db)

	start := make(chan struct{})
	var wg sync.WaitGroup
	errs := make(chan error, 2)

	for i := 0; i < 2; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()
			<-start

			replacement := &models.RefreshSession{
				ID:        uuid.New(),
				UserID:    user.ID,
				TokenHash: "race-new-" + uuid.NewString(),
				ExpiresAt: now.Add(30 * 24 * time.Hour),
			}

			errs <- repository.Rotate(
				ctx,
				oldSession.TokenHash,
				replacement,
				now,
			)
		}()
	}

	close(start)
	wg.Wait()
	close(errs)

	successes := 0
	rejections := 0

	for err := range errs {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, gorm.ErrRecordNotFound):
			rejections++
		default:
			t.Fatalf("unexpected rotation error: %v", err)
		}
	}

	if successes != 1 || rejections != 1 {
		t.Fatalf(
			"expected one successful rotation and one rejection; got %d successes and %d rejections",
			successes,
			rejections,
		)
	}

	var consumed models.RefreshSession
	if err := db.First(&consumed, "id = ?", oldSession.ID).Error; err != nil {
		t.Fatalf("load old session: %v", err)
	}
	if consumed.RevokedAt == nil {
		t.Fatal("old refresh session was not revoked")
	}

	var replacementCount int64
	if err := db.Model(&models.RefreshSession{}).
		Where("user_id = ? AND id <> ?", user.ID, oldSession.ID).
		Count(&replacementCount).Error; err != nil {
		t.Fatalf("count replacement sessions: %v", err)
	}

	if replacementCount != 1 {
		t.Fatalf("expected exactly one replacement session, got %d", replacementCount)
	}
}

func TestRefreshSessionRevokePreventsRotation(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "logout-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Logout Test",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "logout-" + uuid.NewString(),
		ExpiresAt: now.Add(24 * time.Hour),
	}
	if err := db.WithContext(ctx).Create(session).Error; err != nil {
		t.Fatalf("create refresh session: %v", err)
	}

	repository := NewRefreshSessionRepository(db)

	if err := repository.Revoke(ctx, session.TokenHash, now); err != nil {
		t.Fatalf("revoke refresh session: %v", err)
	}

	var revoked models.RefreshSession
	if err := db.First(&revoked, "id = ?", session.ID).Error; err != nil {
		t.Fatalf("load revoked session: %v", err)
	}
	if revoked.RevokedAt == nil {
		t.Fatal("expected refresh session to be revoked")
	}

	err := repository.Rotate(ctx, session.TokenHash, &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "replacement-" + uuid.NewString(),
		ExpiresAt: now.Add(30 * 24 * time.Hour),
	}, now)

	if !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("expected revoked session rotation to be rejected, got %v", err)
	}
}

func TestRefreshSessionRotationRejectsExpired(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()
	now := time.Now()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "expired-session-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Expired Session Test",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "expired-" + uuid.NewString(),
		ExpiresAt: now.Add(-1 * time.Hour), // expired in past
	}
	if err := db.WithContext(ctx).Create(session).Error; err != nil {
		t.Fatalf("create refresh session: %v", err)
	}

	repository := NewRefreshSessionRepository(db)

	err := repository.Rotate(ctx, session.TokenHash, &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: "replacement-" + uuid.NewString(),
		ExpiresAt: now.Add(30 * 24 * time.Hour),
	}, now)

	if !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("expected expired session rotation to be rejected, got %v", err)
	}
}
