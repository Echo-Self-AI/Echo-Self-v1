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
	ErrMediaNotFound = errors.New("media record not found")
	ErrQuotaExceeded = errors.New("user storage quota exceeded")
)

type MediaRepository struct {
	db *gorm.DB
}

func NewMediaRepository(db *gorm.DB) *MediaRepository {
	return &MediaRepository{db: db}
}

func (r *MediaRepository) Create(ctx context.Context, media *models.Media) error {
	return r.db.WithContext(ctx).Create(media).Error
}

// CreateWithQuotaCheck inserts media metadata within a transaction while locking the user row to serialize concurrent uploads.
func (r *MediaRepository) CreateWithQuotaCheck(
	ctx context.Context,
	media *models.Media,
	maxQuotaBytes int64,
	maxObjects int,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Acquire exclusive lock on the user row to serialize concurrent uploads for this user
		var user models.User
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			First(&user, "id = ?", media.UserID).Error; err != nil {
			return err
		}

		type UsageResult struct {
			TotalBytes  *int64
			ObjectCount int64
		}
		var res UsageResult
		if err := tx.Model(&models.Media{}).
			Where("user_id = ? AND status != ?", media.UserID, models.MediaStatusDeleted).
			Select("COALESCE(SUM(file_size), 0) AS total_bytes, COUNT(id) AS object_count").
			Scan(&res).Error; err != nil {
			return err
		}

		var currentBytes int64
		if res.TotalBytes != nil {
			currentBytes = *res.TotalBytes
		}

		if maxObjects > 0 && res.ObjectCount >= int64(maxObjects) {
			return ErrQuotaExceeded
		}
		if maxQuotaBytes > 0 && currentBytes+media.FileSize > maxQuotaBytes {
			return ErrQuotaExceeded
		}

		return tx.Create(media).Error
	})
}

func (r *MediaRepository) FindByIDAndUserID(ctx context.Context, id uuid.UUID, userID uuid.UUID) (*models.Media, error) {
	var media models.Media
	err := r.db.WithContext(ctx).
		Where("id = ? AND user_id = ?", id, userID).
		First(&media).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &media, nil
}

func (r *MediaRepository) ListByUserID(ctx context.Context, userID uuid.UUID, category string) ([]models.Media, error) {
	var mediaList []models.Media
	query := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("created_at DESC")

	if category != "" {
		query = query.Where("category = ?", category)
	}

	err := query.Find(&mediaList).Error
	if err != nil {
		return nil, err
	}
	return mediaList, nil
}

// GetUserStorageUsage calculates the total stored bytes and active object count for a user.
func (r *MediaRepository) GetUserStorageUsage(ctx context.Context, userID uuid.UUID) (totalBytes int64, objectCount int64, err error) {
	type UsageResult struct {
		TotalBytes  *int64
		ObjectCount int64
	}

	var res UsageResult
	err = r.db.WithContext(ctx).
		Model(&models.Media{}).
		Where("user_id = ? AND status != ?", userID, models.MediaStatusDeleted).
		Select("COALESCE(SUM(file_size), 0) AS total_bytes, COUNT(id) AS object_count").
		Scan(&res).Error
	if err != nil {
		return 0, 0, err
	}

	if res.TotalBytes != nil {
		totalBytes = *res.TotalBytes
	}
	objectCount = res.ObjectCount

	return totalBytes, objectCount, nil
}

func (r *MediaRepository) SoftDelete(ctx context.Context, id uuid.UUID, userID uuid.UUID, now time.Time) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var media models.Media
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("id = ? AND user_id = ?", id, userID).
			First(&media).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return ErrMediaNotFound
			}
			return err
		}

		if err := tx.Model(&media).Updates(map[string]interface{}{
			"status":     models.MediaStatusDeletionPending,
			"deleted_at": now,
		}).Error; err != nil {
			return err
		}

		return nil
	})
}

func (r *MediaRepository) MarkStatus(ctx context.Context, id uuid.UUID, userID uuid.UUID, status string) error {
	result := r.db.WithContext(ctx).
		Model(&models.Media{}).
		Where("id = ? AND user_id = ?", id, userID).
		Update("status", status)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return ErrMediaNotFound
	}
	return nil
}

func (r *MediaRepository) HardDelete(ctx context.Context, id uuid.UUID) error {
	return r.db.WithContext(ctx).
		Unscoped().
		Delete(&models.Media{}, "id = ?", id).Error
}

func (r *MediaRepository) ListPendingDeletions(ctx context.Context, limit int) ([]models.Media, error) {
	var list []models.Media
	err := r.db.WithContext(ctx).
		Unscoped().
		Where("status = ?", models.MediaStatusDeletionPending).
		Order("updated_at ASC").
		Limit(limit).
		Find(&list).Error
	if err != nil {
		return nil, err
	}
	return list, nil
}
