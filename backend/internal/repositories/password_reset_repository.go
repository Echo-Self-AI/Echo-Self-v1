package repositories

import (
	"context"
	"errors"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

var ErrInvalidPasswordReset = errors.New("invalid or expired password reset token")

type PasswordResetRepository struct {
	db *gorm.DB
}

func NewPasswordResetRepository(db *gorm.DB) *PasswordResetRepository {
	return &PasswordResetRepository{db: db}
}

func (r *PasswordResetRepository) Create(
	ctx context.Context,
	reset *models.PasswordReset,
) error {
	return r.db.WithContext(ctx).Create(reset).Error
}

func (r *PasswordResetRepository) ResetPassword(
	ctx context.Context,
	tokenHash string,
	now time.Time,
	newPasswordHash string,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var reset models.PasswordReset

		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("token_hash = ?", tokenHash).
			First(&reset).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrInvalidPasswordReset
		}
		if err != nil {
			return err
		}

		if reset.UsedAt != nil || !reset.ExpiresAt.After(now) {
			return ErrInvalidPasswordReset
		}

		var user models.User
		err = tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			First(&user, "id = ?", reset.UserID).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrInvalidPasswordReset
		}
		if err != nil {
			return err
		}

		if err := tx.Model(&user).
			Update("password_hash", newPasswordHash).Error; err != nil {
			return err
		}

		if err := tx.Model(&reset).
			Update("used_at", now).Error; err != nil {
			return err
		}

		// Revoke all refresh sessions so old sessions cannot refresh tokens.
		if err := tx.Model(&models.RefreshSession{}).
			Where("user_id = ? AND revoked_at IS NULL", user.ID).
			Update("revoked_at", now).Error; err != nil {
			return err
		}

		return nil
	})
}

func (r *PasswordResetRepository) InvalidateOutstanding(
	ctx context.Context,
	userID uuid.UUID,
	now time.Time,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return tx.Model(&models.PasswordReset{}).
			Where("user_id = ? AND used_at IS NULL", userID).
			Update("used_at", now).Error
	})
}
