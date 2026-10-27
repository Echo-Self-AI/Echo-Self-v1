package services

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/google/uuid"
)

var (
	ErrJobNotFound        = errors.New("job not found")
	ErrUnsupportedJobType = errors.New("unsupported or disallowed job type")
	ErrInvalidJobPayload  = errors.New("invalid job payload or options")
)

type SubmitJobInput struct {
	JobType string         `json:"job_type"`
	MediaID *string        `json:"media_id,omitempty"`
	Payload map[string]any `json:"payload,omitempty"`
}

type JobResult struct {
	ID                string         `json:"id"`
	UserID            string         `json:"user_id"`
	JobType           string         `json:"job_type"`
	Status            string         `json:"status"`
	InputMediaID      *string        `json:"input_media_id,omitempty"`
	Payload           map[string]any `json:"payload,omitempty"`
	Result            map[string]any `json:"result,omitempty"`
	ErrorCode         *string        `json:"error_code,omitempty"`
	ErrorMessage      *string        `json:"error_message,omitempty"`
	AttemptCount      int            `json:"attempt_count"`
	MaxAttempts       int            `json:"max_attempts"`
	CancelRequestedAt *time.Time     `json:"cancel_requested_at,omitempty"`
	StartedAt         *time.Time     `json:"started_at,omitempty"`
	CompletedAt       *time.Time     `json:"completed_at,omitempty"`
	CreatedAt         time.Time      `json:"created_at"`
	UpdatedAt         time.Time      `json:"updated_at"`
}

type JobRepository interface {
	CreateWithOutbox(ctx context.Context, job *models.Job, outbox *models.JobOutbox) error
	FindByIDAndUserID(ctx context.Context, id, userID uuid.UUID) (*models.Job, error)
	FindByID(ctx context.Context, id uuid.UUID) (*models.Job, error)
	ListByUserID(ctx context.Context, userID uuid.UUID, status string, limit, offset int) ([]models.Job, error)
	RequestCancellation(ctx context.Context, id, userID uuid.UUID, now time.Time) (*models.Job, error)
}

type JobService struct {
	repo         JobRepository
	mediaService *MediaService
	proc         processor.Processor
	onEnqueue    func(ctx context.Context, jobID uuid.UUID) error
}

func NewJobService(
	repo JobRepository,
	mediaService *MediaService,
	proc processor.Processor,
	onEnqueue func(ctx context.Context, jobID uuid.UUID) error,
) *JobService {
	return &JobService{
		repo:         repo,
		mediaService: mediaService,
		proc:         proc,
		onEnqueue:    onEnqueue,
	}
}

func toJobResult(j *models.Job) JobResult {
	var mediaIDStr *string
	if j.InputMediaID != nil {
		s := j.InputMediaID.String()
		mediaIDStr = &s
	}

	return JobResult{
		ID:                j.ID.String(),
		UserID:            j.UserID.String(),
		JobType:           j.JobType,
		Status:            j.Status,
		InputMediaID:      mediaIDStr,
		Payload:           j.Payload,
		Result:            j.Result,
		ErrorCode:         j.ErrorCode,
		ErrorMessage:      j.ErrorMessage,
		AttemptCount:      j.AttemptCount,
		MaxAttempts:       j.MaxAttempts,
		CancelRequestedAt: j.CancelRequestedAt,
		StartedAt:         j.StartedAt,
		CompletedAt:       j.CompletedAt,
		CreatedAt:         j.CreatedAt,
		UpdatedAt:         j.UpdatedAt,
	}
}

func (s *JobService) SubmitJob(
	ctx context.Context,
	userIDStr string,
	input SubmitJobInput,
) (*JobResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	jobType := strings.TrimSpace(input.JobType)
	var supported bool
	for _, t := range s.proc.SupportedJobTypes() {
		if t == jobType {
			supported = true
			break
		}
	}
	if !supported {
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedJobType, jobType)
	}

	var mediaUUID *uuid.UUID
	if input.MediaID != nil && strings.TrimSpace(*input.MediaID) != "" {
		mID, err := uuid.Parse(*input.MediaID)
		if err != nil {
			return nil, ErrMediaNotFound
		}

		// Initial ownership & consent validation prior to submission
		if s.mediaService != nil {
			purpose := resolveJobConsentPurpose(jobType)
			if err := s.mediaService.VerifyProcessingConsent(ctx, userIDStr, mID.String(), purpose); err != nil {
				if errors.Is(err, ErrMediaNotFound) {
					return nil, ErrMediaNotFound
				}
				if errors.Is(err, ErrConsentNotGranted) || errors.Is(err, ErrConsentRevokedOrExpired) {
					return nil, fmt.Errorf("consent check failed: %w", err)
				}
				return nil, fmt.Errorf("verify consent: %w", err)
			}
		}
		mediaUUID = &mID
	}

	now := time.Now().UTC()
	jobID := uuid.New()
	job := &models.Job{
		ID:           jobID,
		UserID:       userID,
		JobType:      jobType,
		Status:       models.JobStatusQueued,
		InputMediaID: mediaUUID,
		Payload:      input.Payload,
		MaxAttempts:  3,
		CreatedAt:    now,
		UpdatedAt:    now,
	}

	outbox := &models.JobOutbox{
		ID:        uuid.New(),
		JobID:     jobID,
		EventType: models.OutboxEventJobQueued,
		Payload: models.JSONMap{
			"job_type": jobType,
			"user_id":  userID.String(),
		},
		Status:    models.OutboxStatusPending,
		CreatedAt: now,
		UpdatedAt: now,
	}

	// 1. Transactional Outbox: Insert job and outbox event in one atomic commit
	if err := s.repo.CreateWithOutbox(ctx, job, outbox); err != nil {
		return nil, fmt.Errorf("create job with outbox: %w", err)
	}

	// 2. Immediate asynchronous dispatch (best effort; background outbox poller guarantees delivery)
	if s.onEnqueue != nil {
		go func() {
			_ = s.onEnqueue(context.Background(), jobID)
		}()
	}

	result := toJobResult(job)
	return &result, nil
}

func (s *JobService) GetJob(
	ctx context.Context,
	userIDStr string,
	jobIDStr string,
) (*JobResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	jobID, err := uuid.Parse(jobIDStr)
	if err != nil {
		return nil, ErrJobNotFound
	}

	job, err := s.repo.FindByIDAndUserID(ctx, jobID, userID)
	if err != nil {
		return nil, fmt.Errorf("find job: %w", err)
	}
	if job == nil {
		return nil, ErrJobNotFound
	}

	result := toJobResult(job)
	return &result, nil
}

func (s *JobService) ListJobs(
	ctx context.Context,
	userIDStr string,
	status string,
	limit, offset int,
) ([]JobResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	jobs, err := s.repo.ListByUserID(ctx, userID, status, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("list jobs: %w", err)
	}

	results := make([]JobResult, len(jobs))
	for i, j := range jobs {
		results[i] = toJobResult(&j)
	}

	return results, nil
}

func (s *JobService) CancelJob(
	ctx context.Context,
	userIDStr string,
	jobIDStr string,
) (*JobResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	jobID, err := uuid.Parse(jobIDStr)
	if err != nil {
		return nil, ErrJobNotFound
	}

	now := time.Now().UTC()
	updated, err := s.repo.RequestCancellation(ctx, jobID, userID, now)
	if err != nil {
		if errors.Is(err, ErrJobNotFound) {
			return nil, ErrJobNotFound
		}
		return nil, fmt.Errorf("cancel job: %w", err)
	}

	result := toJobResult(updated)
	return &result, nil
}

func resolveJobConsentPurpose(jobType string) string {
	switch jobType {
	case processor.JobTypeMockVoiceEmbedding, processor.JobTypeMockTranscription:
		return PurposeVoiceDataProcessing
	case processor.JobTypeMockFaceLandmarks:
		return PurposeFaceLikenessProcessing
	default:
		return PurposeAIPersonalization
	}
}
