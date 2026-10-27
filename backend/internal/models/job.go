package models

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

const (
	JobStatusQueued          = "queued"
	JobStatusProcessing      = "processing"
	JobStatusRetryWait       = "retry_wait"
	JobStatusSucceeded       = "succeeded"
	JobStatusFailed          = "failed"
	JobStatusCancelRequested = "cancel_requested"
	JobStatusCancelled       = "cancelled"

	OutboxStatusPending   = "pending"
	OutboxStatusPublished = "published"
	OutboxStatusFailed    = "failed"

	OutboxEventJobQueued          = "job_queued"
	OutboxEventJobCancelRequested = "job_cancel_requested"
)

// JSONMap is a convenience type for JSONB columns in PostgreSQL.
type JSONMap map[string]any

func (j JSONMap) Value() (driver.Value, error) {
	if j == nil {
		return "{}", nil
	}
	return json.Marshal(j)
}

func (j *JSONMap) Scan(value any) error {
	if value == nil {
		*j = make(JSONMap)
		return nil
	}

	var bytes []byte
	switch v := value.(type) {
	case []byte:
		bytes = v
	case string:
		bytes = []byte(v)
	default:
		return errors.New("type assertion to []byte or string failed")
	}

	if len(bytes) == 0 {
		*j = make(JSONMap)
		return nil
	}

	return json.Unmarshal(bytes, j)
}

// IsTerminalStatus returns true if the job status cannot be changed further.
func IsTerminalStatus(status string) bool {
	switch status {
	case JobStatusSucceeded, JobStatusFailed, JobStatusCancelled:
		return true
	default:
		return false
	}
}

// IsValidStatusTransition validates state transitions for background jobs.
func IsValidStatusTransition(current, next string) bool {
	if current == next {
		return true
	}

	if IsTerminalStatus(current) {
		return false
	}

	switch current {
	case JobStatusQueued:
		return next == JobStatusProcessing ||
			next == JobStatusCancelRequested ||
			next == JobStatusCancelled

	case JobStatusProcessing:
		return next == JobStatusSucceeded ||
			next == JobStatusFailed ||
			next == JobStatusRetryWait ||
			next == JobStatusCancelRequested ||
			next == JobStatusCancelled

	case JobStatusRetryWait:
		return next == JobStatusProcessing ||
			next == JobStatusQueued ||
			next == JobStatusCancelRequested ||
			next == JobStatusCancelled

	case JobStatusCancelRequested:
		return next == JobStatusCancelled ||
			next == JobStatusFailed ||
			next == JobStatusSucceeded

	default:
		return false
	}
}

type Job struct {
	ID                uuid.UUID  `gorm:"type:uuid;primaryKey" json:"id"`
	UserID            uuid.UUID  `gorm:"type:uuid;not null;index" json:"user_id"`
	JobType           string     `gorm:"type:varchar(50);not null" json:"job_type"`
	Status            string     `gorm:"type:varchar(30);not null;default:'queued'" json:"status"`
	InputMediaID      *uuid.UUID `gorm:"type:uuid" json:"input_media_id,omitempty"`
	Payload           JSONMap    `gorm:"type:jsonb;not null;default:'{}'" json:"payload"`
	Result            JSONMap    `gorm:"type:jsonb" json:"result,omitempty"`
	ErrorCode         *string    `gorm:"type:varchar(100)" json:"error_code,omitempty"`
	ErrorMessage      *string    `gorm:"type:text" json:"error_message,omitempty"`
	AttemptCount      int        `gorm:"not null;default:0" json:"attempt_count"`
	MaxAttempts       int        `gorm:"not null;default:3" json:"max_attempts"`
	LockedBy          *string    `gorm:"type:varchar(100)" json:"-"`
	LockedUntil       *time.Time `json:"-"`
	RetryAt           *time.Time `json:"retry_at,omitempty"`
	CancelRequestedAt *time.Time `json:"cancel_requested_at,omitempty"`
	StartedAt         *time.Time `json:"started_at,omitempty"`
	CompletedAt       *time.Time `json:"completed_at,omitempty"`
	CreatedAt         time.Time  `json:"created_at"`
	UpdatedAt         time.Time  `json:"updated_at"`
}

func (Job) TableName() string {
	return "jobs"
}

func (j *Job) BeforeCreate(tx *gorm.DB) error {
	if j.ID == uuid.Nil {
		j.ID = uuid.New()
	}
	if j.Payload == nil {
		j.Payload = make(JSONMap)
	}
	return nil
}

type JobOutbox struct {
	ID          uuid.UUID  `gorm:"type:uuid;primaryKey" json:"id"`
	JobID       uuid.UUID  `gorm:"type:uuid;not null;index" json:"job_id"`
	EventType   string     `gorm:"type:varchar(50);not null" json:"event_type"`
	Payload     JSONMap    `gorm:"type:jsonb;not null;default:'{}'" json:"payload"`
	Status      string     `gorm:"type:varchar(30);not null;default:'pending'" json:"status"`
	PublishedAt *time.Time `json:"published_at,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

func (JobOutbox) TableName() string {
	return "job_outbox"
}

func (o *JobOutbox) BeforeCreate(tx *gorm.DB) error {
	if o.ID == uuid.Nil {
		o.ID = uuid.New()
	}
	if o.Payload == nil {
		o.Payload = make(JSONMap)
	}
	return nil
}
