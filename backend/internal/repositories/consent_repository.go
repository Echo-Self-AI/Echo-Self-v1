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
	ErrConsentNotFound      = errors.New("consent record not found")
	ErrConsentAlreadyActive = errors.New("active consent record already exists for this purpose")
)

type ConsentRepository struct {
	db *gorm.DB
}

func NewConsentRepository(db *gorm.DB) *ConsentRepository {
	return &ConsentRepository{db: db}
}

func (r *ConsentRepository) CreateConsent(
	ctx context.Context,
	consent *models.ConsentRecord,
	auditLog *models.ConsentAuditLog,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var existing models.ConsentRecord
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("user_id = ? AND consent_type = ? AND status = ?",
				consent.UserID, consent.ConsentType, models.ConsentStatusGranted).
			First(&existing).Error

		if err == nil {
			return ErrConsentAlreadyActive
		}
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return err
		}

		if err := tx.Create(consent).Error; err != nil {
			return err
		}

		auditLog.ConsentID = consent.ID
		auditLog.UserID = consent.UserID
		auditLog.ConsentType = consent.ConsentType
		auditLog.PolicyVersion = consent.PolicyVersion
		return tx.Create(auditLog).Error
	})
}

func (r *ConsentRepository) RevokeConsent(
	ctx context.Context,
	id uuid.UUID,
	userID uuid.UUID,
	now time.Time,
	auditLog *models.ConsentAuditLog,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var current models.ConsentRecord
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("id = ? AND user_id = ?", id, userID).
			First(&current).Error

		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrConsentNotFound
		}
		if err != nil {
			return err
		}

		if current.Status == models.ConsentStatusRevoked {
			// Idempotent: already revoked
			return nil
		}

		if err := tx.Model(&current).
			Updates(map[string]interface{}{
				"status":     models.ConsentStatusRevoked,
				"revoked_at": now,
				"updated_at": now,
			}).Error; err != nil {
			return err
		}

		auditLog.ConsentID = current.ID
		auditLog.UserID = current.UserID
		auditLog.Action = models.ConsentStatusRevoked
		auditLog.ConsentType = current.ConsentType
		auditLog.PolicyVersion = current.PolicyVersion
		auditLog.CreatedAt = now

		return tx.Create(auditLog).Error
	})
}

func (r *ConsentRepository) FindByIDAndUserID(
	ctx context.Context,
	id uuid.UUID,
	userID uuid.UUID,
) (*models.ConsentRecord, error) {
	var record models.ConsentRecord
	err := r.db.WithContext(ctx).
		Where("id = ? AND user_id = ?", id, userID).
		First(&record).Error

	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	return &record, nil
}

func (r *ConsentRepository) ListByUserID(
	ctx context.Context,
	userID uuid.UUID,
) ([]models.ConsentRecord, error) {
	var records []models.ConsentRecord
	err := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("created_at DESC").
		Find(&records).Error

	return records, err
}

func (r *ConsentRepository) GetActiveConsent(
	ctx context.Context,
	userID uuid.UUID,
	consentType string,
) (*models.ConsentRecord, error) {
	var record models.ConsentRecord
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND consent_type = ? AND status = ?",
			userID, consentType, models.ConsentStatusGranted).
		First(&record).Error

	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	return &record, nil
}

func (r *ConsentRepository) ListAuditLogsByConsentID(
	ctx context.Context,
	consentID uuid.UUID,
	userID uuid.UUID,
) ([]models.ConsentAuditLog, error) {
	var logs []models.ConsentAuditLog
	err := r.db.WithContext(ctx).
		Where("consent_id = ? AND user_id = ?", consentID, userID).
		Order("created_at ASC").
		Find(&logs).Error

	return logs, err
}
