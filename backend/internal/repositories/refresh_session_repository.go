package repositories

import (
	"context"
	"errors"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

var (
	ErrTokenReuseDetected = errors.New("refresh token reuse detected")
)

type RefreshSessionRepository struct {
	db *gorm.DB
}

func NewRefreshSessionRepository(db *gorm.DB) *RefreshSessionRepository {
	return &RefreshSessionRepository{db: db}
}

func (r *RefreshSessionRepository) Create(
	ctx context.Context,
	session *models.RefreshSession,
) error {
	return r.db.WithContext(ctx).Create(session).Error
}

func (r *RefreshSessionRepository) Rotate(
	ctx context.Context,
	tokenHash string,
	newSession *models.RefreshSession,
	now time.Time,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var current models.RefreshSession

		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("token_hash = ?", tokenHash).
			First(&current).Error

		if errors.Is(err, gorm.ErrRecordNotFound) {
			return gorm.ErrRecordNotFound
		}
		if err != nil {
			return err
		}

		// Security: Detect refresh token reuse (RFC 6819)
		// If a revoked token is used again, invalidate ALL active sessions for this user to mitigate theft
		if current.RevokedAt != nil {
			_ = tx.Model(&models.RefreshSession{}).
				Where("user_id = ? AND revoked_at IS NULL", current.UserID).
				Update("revoked_at", now).Error
			return ErrTokenReuseDetected
		}

		if !current.ExpiresAt.After(now) {
			return gorm.ErrRecordNotFound
		}

		// consume the old token and create the replacement atomically
		if err := tx.Model(&current).Update("revoked_at", now).Error; err != nil {
			return err
		}

		return tx.Create(newSession).Error
	})
}

func (r *RefreshSessionRepository) Revoke(
	ctx context.Context,
	tokenHash string,
	now time.Time,
) error {
	result := r.db.WithContext(ctx).
		Model(&models.RefreshSession{}).
		Where("token_hash = ? AND revoked_at IS NULL", tokenHash).
		Update("revoked_at", now)

	return result.Error
}

func (r *RefreshSessionRepository) FindByTokenHash(
	ctx context.Context,
	tokenHash string,
) (*models.RefreshSession, error) {
	var session models.RefreshSession

	err := r.db.WithContext(ctx).
		Where("token_hash = ?", tokenHash).
		First(&session).Error

	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}

	if err != nil {
		return nil, err
	}

	return &session, nil
}
