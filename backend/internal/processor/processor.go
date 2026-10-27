package processor

import (
	"context"
	"errors"

	"github.com/google/uuid"
)

var (
	ErrUnsupportedJobType = errors.New("processor: unsupported job type")
	ErrInvalidRequest     = errors.New("processor: invalid job request")
	ErrTransientFailure   = errors.New("processor: transient failure; retryable")
	ErrPermanentFailure   = errors.New("processor: permanent processing failure")
)

// ProcessRequest encapsulates the authorized inputs needed by a processor.
type ProcessRequest struct {
	JobID   uuid.UUID      `json:"job_id"`
	JobType string         `json:"job_type"`
	UserID  uuid.UUID      `json:"user_id"`
	MediaID *uuid.UUID     `json:"media_id,omitempty"`
	Payload map[string]any `json:"payload,omitempty"`
}

// ProcessResult defines the typed output returned upon successful processing.
type ProcessResult struct {
	Output   map[string]any `json:"output"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Processor defines the interface for job execution.
// A future FastAPI microservice adapter will implement this interface over HTTP,
// cleanly decoupling queue orchestration from AI inference.
type Processor interface {
	Process(ctx context.Context, req ProcessRequest) (*ProcessResult, error)
	SupportedJobTypes() []string
}
