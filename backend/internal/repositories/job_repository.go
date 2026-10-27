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
	ErrJobNotFound        = errors.New("job record not found")
	ErrJobAlreadyClaimed  = errors.New("job is already claimed or no longer eligible")
	ErrTerminalState      = errors.New("job is in a terminal state and cannot be modified")
	ErrInvalidLeaseHolder = errors.New("worker does not hold the active lease for this job")
)

type JobRepository struct {
	db *gorm.DB
}

func NewJobRepository(db *gorm.DB) *JobRepository {
	return &JobRepository{db: db}
}

// CreateWithOutbox atomically persists the job record and its pending outbox event.
func (r *JobRepository) CreateWithOutbox(
	ctx context.Context,
	job *models.Job,
	outbox *models.JobOutbox,
) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(job).Error; err != nil {
			return err
		}

		outbox.JobID = job.ID
		if err := tx.Create(outbox).Error; err != nil {
			return err
		}

		return nil
	})
}

func (r *JobRepository) FindByIDAndUserID(
	ctx context.Context,
	id uuid.UUID,
	userID uuid.UUID,
) (*models.Job, error) {
	var job models.Job
	err := r.db.WithContext(ctx).
		Where("id = ? AND user_id = ?", id, userID).
		First(&job).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &job, nil
}

func (r *JobRepository) FindByID(
	ctx context.Context,
	id uuid.UUID,
) (*models.Job, error) {
	var job models.Job
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&job).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &job, nil
}

func (r *JobRepository) ListByUserID(
	ctx context.Context,
	userID uuid.UUID,
	status string,
	limit, offset int,
) ([]models.Job, error) {
	var jobs []models.Job
	query := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("created_at DESC")

	if status != "" {
		query = query.Where("status = ?", status)
	}
	if limit > 0 {
		query = query.Limit(limit)
	}
	if offset > 0 {
		query = query.Offset(offset)
	}

	if err := query.Find(&jobs).Error; err != nil {
		return nil, err
	}
	return jobs, nil
}

// ClaimJob safely claims a job for processing using conditional row-level locking.
func (r *JobRepository) ClaimJob(
	ctx context.Context,
	id uuid.UUID,
	workerID string,
	leaseDuration time.Duration,
) (*models.Job, error) {
	var claimed models.Job
	now := time.Now().UTC()
	lockedUntil := now.Add(leaseDuration)

	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var job models.Job
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("id = ?", id).
			First(&job).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return ErrJobNotFound
			}
			return err
		}

		// Prevent terminal jobs from being claimed
		if models.IsTerminalStatus(job.Status) {
			return ErrTerminalState
		}

		// Check if job is eligible
		canClaim := false
		if job.Status == models.JobStatusQueued {
			canClaim = true
		} else if job.Status == models.JobStatusRetryWait {
			if job.RetryAt == nil || !job.RetryAt.After(now) {
				canClaim = true
			}
		} else if job.Status == models.JobStatusProcessing {
			// Lease expired -> worker crashed or hung
			if job.LockedUntil != nil && job.LockedUntil.Before(now) {
				canClaim = true
			}
		}

		if !canClaim {
			return ErrJobAlreadyClaimed
		}

		job.Status = models.JobStatusProcessing
		job.LockedBy = &workerID
		job.LockedUntil = &lockedUntil
		if job.StartedAt == nil {
			job.StartedAt = &now
		}
		job.AttemptCount++
		job.UpdatedAt = now

		if err := tx.Save(&job).Error; err != nil {
			return err
		}

		claimed = job
		return nil
	})

	if err != nil {
		return nil, err
	}
	return &claimed, nil
}

// RenewLease extends the worker lease while processing continues.
func (r *JobRepository) RenewLease(
	ctx context.Context,
	id uuid.UUID,
	workerID string,
	leaseDuration time.Duration,
) error {
	now := time.Now().UTC()
	newExpiry := now.Add(leaseDuration)

	result := r.db.WithContext(ctx).
		Model(&models.Job{}).
		Where("id = ? AND locked_by = ? AND status = ?", id, workerID, models.JobStatusProcessing).
		Updates(map[string]any{
			"locked_until": newExpiry,
			"updated_at":   now,
		})
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return ErrInvalidLeaseHolder
	}
	return nil
}

// MarkSucceeded marks the job as successfully completed and persists typed result.
func (r *JobRepository) MarkSucceeded(
	ctx context.Context,
	id uuid.UUID,
	workerID string,
	result models.JSONMap,
) error {
	now := time.Now().UTC()
	res := r.db.WithContext(ctx).
		Model(&models.Job{}).
		Where("id = ? AND locked_by = ? AND status IN (?, ?)", id, workerID, models.JobStatusProcessing, models.JobStatusCancelRequested).
		Updates(map[string]any{
			"status":       models.JobStatusSucceeded,
			"result":       result,
			"completed_at": now,
			"locked_by":    nil,
			"locked_until": nil,
			"updated_at":   now,
		})
	if res.Error != nil {
		return res.Error
	}
	if res.RowsAffected == 0 {
		return ErrInvalidLeaseHolder
	}
	return nil
}

// MarkFailed records a terminal failure on the job.
func (r *JobRepository) MarkFailed(
	ctx context.Context,
	id uuid.UUID,
	workerID string,
	errCode, errMsg string,
) error {
	now := time.Now().UTC()
	res := r.db.WithContext(ctx).
		Model(&models.Job{}).
		Where("id = ? AND locked_by = ?", id, workerID).
		Updates(map[string]any{
			"status":        models.JobStatusFailed,
			"error_code":    errCode,
			"error_message": errMsg,
			"completed_at":  now,
			"locked_by":     nil,
			"locked_until":  nil,
			"updated_at":    now,
		})
	if res.Error != nil {
		return res.Error
	}
	if res.RowsAffected == 0 {
		return ErrInvalidLeaseHolder
	}
	return nil
}

// ScheduleRetry puts the job into retry_wait state with an exponential backoff timestamp.
func (r *JobRepository) ScheduleRetry(
	ctx context.Context,
	id uuid.UUID,
	workerID string,
	retryAt time.Time,
	errCode, errMsg string,
) error {
	now := time.Now().UTC()
	res := r.db.WithContext(ctx).
		Model(&models.Job{}).
		Where("id = ? AND locked_by = ?", id, workerID).
		Updates(map[string]any{
			"status":        models.JobStatusRetryWait,
			"retry_at":      retryAt,
			"error_code":    errCode,
			"error_message": errMsg,
			"locked_by":     nil,
			"locked_until":  nil,
			"updated_at":    now,
		})
	if res.Error != nil {
		return res.Error
	}
	if res.RowsAffected == 0 {
		return ErrInvalidLeaseHolder
	}
	return nil
}

// RequestCancellation safely transitions queued/running jobs towards cancellation.
func (r *JobRepository) RequestCancellation(
	ctx context.Context,
	id uuid.UUID,
	userID uuid.UUID,
	now time.Time,
) (*models.Job, error) {
	var updated models.Job
	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var job models.Job
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("id = ? AND user_id = ?", id, userID).
			First(&job).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return ErrJobNotFound
			}
			return err
		}

		// Idempotent: already terminal
		if models.IsTerminalStatus(job.Status) {
			updated = job
			return nil
		}

		if job.Status == models.JobStatusQueued || job.Status == models.JobStatusRetryWait {
			// Not yet claimed by any active worker: cancel immediately
			job.Status = models.JobStatusCancelled
			job.CompletedAt = &now
			job.CancelRequestedAt = &now
		} else if job.Status == models.JobStatusProcessing {
			// Actively running: mark cancel_requested so worker aborts at checkpoint
			job.Status = models.JobStatusCancelRequested
			job.CancelRequestedAt = &now
		}

		job.UpdatedAt = now
		if err := tx.Save(&job).Error; err != nil {
			return err
		}

		updated = job
		return nil
	})

	if err != nil {
		return nil, err
	}
	return &updated, nil
}

// AcknowledgeCancellation transitions cancel_requested jobs to cancelled upon worker exit.
func (r *JobRepository) AcknowledgeCancellation(
	ctx context.Context,
	id uuid.UUID,
	workerID string,
) error {
	now := time.Now().UTC()
	return r.db.WithContext(ctx).
		Model(&models.Job{}).
		Where("id = ? AND locked_by = ?", id, workerID).
		Updates(map[string]any{
			"status":       models.JobStatusCancelled,
			"completed_at": now,
			"locked_by":    nil,
			"locked_until": nil,
			"updated_at":   now,
		}).Error
}

// GetPendingOutbox retrieves pending outbox events for dispatch to Redis.
func (r *JobRepository) GetPendingOutbox(
	ctx context.Context,
	limit int,
) ([]models.JobOutbox, error) {
	var list []models.JobOutbox
	err := r.db.WithContext(ctx).
		Where("status = ?", models.OutboxStatusPending).
		Order("created_at ASC").
		Limit(limit).
		Find(&list).Error
	if err != nil {
		return nil, err
	}
	return list, nil
}

func (r *JobRepository) MarkOutboxPublished(
	ctx context.Context,
	id uuid.UUID,
	publishedAt time.Time,
) error {
	return r.db.WithContext(ctx).
		Model(&models.JobOutbox{}).
		Where("id = ?", id).
		Updates(map[string]any{
			"status":       models.OutboxStatusPublished,
			"published_at": publishedAt,
			"updated_at":   publishedAt,
		}).Error
}

func (r *JobRepository) MarkOutboxFailed(
	ctx context.Context,
	id uuid.UUID,
) error {
	now := time.Now().UTC()
	return r.db.WithContext(ctx).
		Model(&models.JobOutbox{}).
		Where("id = ?", id).
		Updates(map[string]any{
			"status":     models.OutboxStatusFailed,
			"updated_at": now,
		}).Error
}

// RecoverStaleJobs finds jobs whose workers died or timed out and resets or fails them.
func (r *JobRepository) RecoverStaleJobs(
	ctx context.Context,
	limit int,
) (int64, error) {
	now := time.Now().UTC()
	var recovered int64

	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var staleJobs []models.Job
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
			Where("status = ? AND locked_until < ?", models.JobStatusProcessing, now).
			Limit(limit).
			Find(&staleJobs).Error
		if err != nil {
			return err
		}

		for _, job := range staleJobs {
			if job.AttemptCount >= job.MaxAttempts {
				errCode := "LEASE_EXPIRED_MAX_ATTEMPTS"
				errMsg := "worker lease expired and maximum attempt count reached"
				if err := tx.Model(&job).Updates(map[string]any{
					"status":        models.JobStatusFailed,
					"error_code":    errCode,
					"error_message": errMsg,
					"completed_at":  now,
					"locked_by":     nil,
					"locked_until":  nil,
					"updated_at":    now,
				}).Error; err != nil {
					return err
				}
			} else {
				// Eligible for retry
				if err := tx.Model(&job).Updates(map[string]any{
					"status":       models.JobStatusRetryWait,
					"retry_at":     now,
					"locked_by":    nil,
					"locked_until": nil,
					"updated_at":   now,
				}).Error; err != nil {
					return err
				}
			}
			recovered++
		}

		return nil
	})

	if err != nil {
		return 0, err
	}
	return recovered, nil
}
