package services

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
	"github.com/google/uuid"
)

type mockJobRepo struct {
	jobs    map[uuid.UUID]*models.Job
	outbox  map[uuid.UUID]*models.JobOutbox
	failErr error
}

func newMockJobRepo() *mockJobRepo {
	return &mockJobRepo{
		jobs:   make(map[uuid.UUID]*models.Job),
		outbox: make(map[uuid.UUID]*models.JobOutbox),
	}
}

func (m *mockJobRepo) CreateWithOutbox(ctx context.Context, job *models.Job, outbox *models.JobOutbox) error {
	if m.failErr != nil {
		return m.failErr
	}
	m.jobs[job.ID] = job
	m.outbox[outbox.ID] = outbox
	return nil
}

func (m *mockJobRepo) FindByIDAndUserID(ctx context.Context, id, userID uuid.UUID) (*models.Job, error) {
	j, ok := m.jobs[id]
	if !ok || j.UserID != userID {
		return nil, nil
	}
	return j, nil
}

func (m *mockJobRepo) FindByID(ctx context.Context, id uuid.UUID) (*models.Job, error) {
	j, ok := m.jobs[id]
	if !ok {
		return nil, nil
	}
	return j, nil
}

func (m *mockJobRepo) ListByUserID(ctx context.Context, userID uuid.UUID, status string, limit, offset int) ([]models.Job, error) {
	var list []models.Job
	for _, j := range m.jobs {
		if j.UserID == userID && (status == "" || j.Status == status) {
			list = append(list, *j)
		}
	}
	return list, nil
}

func (m *mockJobRepo) RequestCancellation(ctx context.Context, id, userID uuid.UUID, now time.Time) (*models.Job, error) {
	j, ok := m.jobs[id]
	if !ok || j.UserID != userID {
		return nil, repositories.ErrJobNotFound
	}
	if !models.IsTerminalStatus(j.Status) {
		j.Status = models.JobStatusCancelled
		j.CancelRequestedAt = &now
		j.CompletedAt = &now
	}
	return j, nil
}

func TestJobService_Submit(t *testing.T) {
	ctx := context.Background()
	mockRepo := newMockJobRepo()
	proc := processor.NewMockProcessor()
	mediaRepo := newMockMediaRepo()
	store := storage.NewMemoryStorage()
	mediaService := NewMediaService(mediaRepo, store, nil, defaultTestMediaConfig())

	var enqueuedJob uuid.UUID
	onEnqueue := func(ctx context.Context, id uuid.UUID) error {
		enqueuedJob = id
		return nil
	}

	service := NewJobService(mockRepo, mediaService, proc, onEnqueue)
	userID := uuid.New().String()

	// 1. Invalid User ID
	_, err := service.SubmitJob(ctx, "invalid-uuid", SubmitJobInput{
		JobType: processor.JobTypeMockVoiceEmbedding,
	})
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("expected ErrUnauthorized, got %v", err)
	}

	// 2. Unsupported job type
	_, err = service.SubmitJob(ctx, userID, SubmitJobInput{
		JobType: "unsupported_ml_model",
	})
	if !errors.Is(err, ErrUnsupportedJobType) {
		t.Fatalf("expected ErrUnsupportedJobType, got %v", err)
	}

	// 3. Non-existent media reference
	fakeMediaID := uuid.New().String()
	_, err = service.SubmitJob(ctx, userID, SubmitJobInput{
		JobType: processor.JobTypeMockVoiceEmbedding,
		MediaID: &fakeMediaID,
	})
	if !errors.Is(err, ErrMediaNotFound) {
		t.Fatalf("expected ErrMediaNotFound, got %v", err)
	}

	// 4. Valid job submission without media
	res, err := service.SubmitJob(ctx, userID, SubmitJobInput{
		JobType: processor.JobTypeMockTranscription,
		Payload: map[string]any{"language": "en-US"},
	})
	if err != nil {
		t.Fatalf("unexpected error submitting job: %v", err)
	}

	if res.Status != models.JobStatusQueued {
		t.Errorf("expected status 'queued', got %q", res.Status)
	}
	if res.JobType != processor.JobTypeMockTranscription {
		t.Errorf("expected job type %q, got %q", processor.JobTypeMockTranscription, res.JobType)
	}
	if res.UserID != userID {
		t.Errorf("expected user id %q, got %q", userID, res.UserID)
	}

	// Wait briefly for asynchronous enqueue hook
	time.Sleep(10 * time.Millisecond)
	if enqueuedJob.String() != res.ID {
		t.Errorf("expected enqueuedJob %s, got %s", res.ID, enqueuedJob.String())
	}

	// 5. Get Job
	fetched, err := service.GetJob(ctx, userID, res.ID)
	if err != nil {
		t.Fatalf("get job failed: %v", err)
	}
	if fetched.ID != res.ID {
		t.Errorf("expected id %s, got %s", res.ID, fetched.ID)
	}

	// Cross-user lookup
	otherUser := uuid.New().String()
	_, err = service.GetJob(ctx, otherUser, res.ID)
	if !errors.Is(err, ErrJobNotFound) {
		t.Fatalf("expected ErrJobNotFound for other user, got %v", err)
	}

	// 6. List Jobs
	list, err := service.ListJobs(ctx, userID, "", 10, 0)
	if err != nil {
		t.Fatalf("list jobs failed: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("expected 1 job, got %d", len(list))
	}

	// 7. Cancel Job
	cancelled, err := service.CancelJob(ctx, userID, res.ID)
	if err != nil {
		t.Fatalf("cancel job failed: %v", err)
	}
	if cancelled.Status != models.JobStatusCancelled {
		t.Errorf("expected status 'cancelled', got %q", cancelled.Status)
	}
}
