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

type UserRepository struct {
	db *gorm.DB
}

func NewUserRepository(db *gorm.DB) *UserRepository {
	return &UserRepository{
		db: db,
	}
}

func (r *UserRepository) Create(
	ctx context.Context,
	user *models.User,
) error {
	return r.db.WithContext(ctx).Create(user).Error
}

func (r *UserRepository) FindByEmail(
	ctx context.Context,
	email string,
) (*models.User, error) {
	var user models.User

	err := r.db.WithContext(ctx).
		Where("email = ?", email).
		First(&user).Error

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}

		return nil, err
	}

	return &user, nil
}

func (r *UserRepository) FindByID(
	ctx context.Context,
	id string,
) (*models.User, error) {
	var user models.User

	err := r.db.WithContext(ctx).
		First(&user, "id = ?", id).Error

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}

		return nil, err
	}

	return &user, nil
}

func (r *UserRepository) UpdateProfile(
	ctx context.Context,
	id uuid.UUID,
	displayName string,
) (*models.User, error) {
	var user models.User

	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			First(&user, "id = ?", id).Error; err != nil {
			return err
		}

		user.DisplayName = displayName
		return tx.Model(&user).Update("display_name", displayName).Error
	})

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}

	return &user, nil
}

func (r *UserRepository) DeleteAccount(
	ctx context.Context,
	id uuid.UUID,
	now time.Time,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var user models.User
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			First(&user, "id = ?", id).Error; err != nil {
			return err
		}

		// Soft delete the user
		if err := tx.Delete(&user).Error; err != nil {
			return err
		}

		// Revoke all active refresh sessions
		if err := tx.Model(&models.RefreshSession{}).
			Where("user_id = ? AND revoked_at IS NULL", id).
			Update("revoked_at", now).Error; err != nil {
			return err
		}

		// Invalidate pending email verifications
		if err := tx.Model(&models.EmailVerification{}).
			Where("user_id = ? AND used_at IS NULL", id).
			Update("used_at", now).Error; err != nil {
			return err
		}

		// Invalidate pending password resets
		if err := tx.Model(&models.PasswordReset{}).
			Where("user_id = ? AND used_at IS NULL", id).
			Update("used_at", now).Error; err != nil {
			return err
		}

		// Soft delete media records and mark deletion pending
		if err := tx.Model(&models.Media{}).
			Where("user_id = ? AND deleted_at IS NULL", id).
			Updates(map[string]interface{}{
				"status":     models.MediaStatusDeletionPending,
				"deleted_at": now,
			}).Error; err != nil {
			return err
		}

		return nil
	})
}
