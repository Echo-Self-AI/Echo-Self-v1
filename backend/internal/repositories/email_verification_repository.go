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

var (
	ErrInvalidVerificationRecord = errors.New("invalid verification record")
	ErrAlreadyVerified           = errors.New("email already verified")
)

type EmailVerificationRepository struct {
	db *gorm.DB
}

func NewEmailVerificationRepository(db *gorm.DB) *EmailVerificationRepository {
	return &EmailVerificationRepository{db: db}
}

func (r *EmailVerificationRepository) Create(
	ctx context.Context,
	verification *models.EmailVerification,
) error {
	return r.db.WithContext(ctx).Create(verification).Error
}

func (r *EmailVerificationRepository) FindByTokenHash(
	ctx context.Context,
	tokenHash string,
) (*models.EmailVerification, error) {
	var verification models.EmailVerification

	err := r.db.WithContext(ctx).
		Where("token_hash = ?", tokenHash).
		First(&verification).Error

	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	return &verification, nil
}

func (r *EmailVerificationRepository) Consume(
	ctx context.Context,
	tokenHash string,
	now time.Time,
) (*models.EmailVerification, error) {
	var verification models.EmailVerification

	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("token_hash = ?", tokenHash).
			First(&verification).Error
		if err != nil {
			return err
		}

		if verification.UsedAt != nil || !verification.ExpiresAt.After(now) {
			return gorm.ErrRecordNotFound
		}

		if err := tx.Model(&verification).Update("used_at", now).Error; err != nil {
			return err
		}

		return nil
	})

	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	return &verification, nil
}

func (r *EmailVerificationRepository) MarkUserVerified(
	ctx context.Context,
	userID uuid.UUID,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var user models.User

		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			First(&user, "id = ?", userID).Error; err != nil {
			return err
		}

		if user.IsEmailVerified {
			return nil
		}

		return tx.Model(&user).
			Update("is_email_verified", true).Error
	})
}

func (r *EmailVerificationRepository) VerifyAndConsume(
	ctx context.Context,
	tokenHash string,
	now time.Time,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var verification models.EmailVerification

		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("token_hash = ?", tokenHash).
			First(&verification).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrInvalidVerificationRecord
		}
		if err != nil {
			return err
		}

		if verification.UsedAt != nil || !verification.ExpiresAt.After(now) {
			return ErrInvalidVerificationRecord
		}

		var user models.User
		err = tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			First(&user, "id = ?", verification.UserID).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrInvalidVerificationRecord
		}
		if err != nil {
			return err
		}

		if user.IsEmailVerified {
			return ErrAlreadyVerified
		}

		if err := tx.Model(&verification).
			Update("used_at", now).Error; err != nil {
			return err
		}

		return tx.Model(&user).
			Update("is_email_verified", true).Error
	})
}
