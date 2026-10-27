package worker

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/queue"
	"github.com/alicebob/miniredis/v2"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

type workerMockJobRepo struct {
	mu        sync.Mutex
	jobs      map[uuid.UUID]*models.Job
	succeeded map[uuid.UUID]models.JSONMap
	failed    map[uuid.UUID]string
	retries   map[uuid.UUID]time.Time
	cancelled map[uuid.UUID]bool
	renewals  int
}

func newWorkerMockJobRepo() *workerMockJobRepo {
	return &workerMockJobRepo{
		jobs:      make(map[uuid.UUID]*models.Job),
		succeeded: make(map[uuid.UUID]models.JSONMap),
		failed:    make(map[uuid.UUID]string),
		retries:   make(map[uuid.UUID]time.Time),
		cancelled: make(map[uuid.UUID]bool),
	}
}

func (m *workerMockJobRepo) ClaimJob(ctx context.Context, id uuid.UUID, workerID string, leaseDuration time.Duration) (*models.Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	j, ok := m.jobs[id]
	if !ok {
		return nil, errors.New("not found")
	}

	if j.Status == models.JobStatusProcessing && (j.LockedUntil == nil || j.LockedUntil.After(time.Now())) {
		return nil, errors.New("already claimed")
	}

	now := time.Now()
	expiry := now.Add(leaseDuration)
	j.Status = models.JobStatusProcessing
	j.LockedBy = &workerID
	j.LockedUntil = &expiry
	j.AttemptCount++
	return j, nil
}

func (m *workerMockJobRepo) RenewLease(ctx context.Context, id uuid.UUID, workerID string, leaseDuration time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.renewals++
	return nil
}

func (m *workerMockJobRepo) MarkSucceeded(ctx context.Context, id uuid.UUID, workerID string, result models.JSONMap) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.succeeded[id] = result
	if j, ok := m.jobs[id]; ok {
		j.Status = models.JobStatusSucceeded
		j.Result = result
	}
	return nil
}

func (m *workerMockJobRepo) MarkFailed(ctx context.Context, id uuid.UUID, workerID string, errCode, errMsg string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failed[id] = errCode
	if j, ok := m.jobs[id]; ok {
		j.Status = models.JobStatusFailed
		j.ErrorCode = &errCode
	}
	return nil
}

func (m *workerMockJobRepo) ScheduleRetry(ctx context.Context, id uuid.UUID, workerID string, retryAt time.Time, errCode, errMsg string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.retries[id] = retryAt
	if j, ok := m.jobs[id]; ok {
		j.Status = models.JobStatusRetryWait
		j.RetryAt = &retryAt
	}
	return nil
}

func (m *workerMockJobRepo) FindByID(ctx context.Context, id uuid.UUID) (*models.Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.jobs[id], nil
}

func (m *workerMockJobRepo) AcknowledgeCancellation(ctx context.Context, id uuid.UUID, workerID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cancelled[id] = true
	if j, ok := m.jobs[id]; ok {
		j.Status = models.JobStatusCancelled
	}
	return nil
}

func (m *workerMockJobRepo) RecoverStaleJobs(ctx context.Context, limit int) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var count int64
	now := time.Now()
	for _, j := range m.jobs {
		if j.Status == models.JobStatusProcessing && j.LockedUntil != nil && j.LockedUntil.Before(now) {
			j.Status = models.JobStatusRetryWait
			count++
		}
	}
	return count, nil
}

func setupWorkerTest(t *testing.T) (*miniredis.Miniredis, *queue.RedisQueue, *workerMockJobRepo) {
	t.Helper()
	s := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: s.Addr()})
	t.Cleanup(func() {
		_ = rdb.Close()
		s.Close()
	})

	q := queue.NewRedisQueue(rdb, "test:worker:queue")
	repo := newWorkerMockJobRepo()
	return s, q, repo
}

func TestWorker_ExecutionLifecycle(t *testing.T) {
	_, q, repo := setupWorkerTest(t)
	proc := processor.NewMockProcessor()

	w := NewWorker(repo, q, proc, nil, WorkerConfig{
		Concurrency:     1,
		LeaseDuration:   5 * time.Second,
		DrainTimeout:    2 * time.Second,
		MinRetryBackoff: 100 * time.Millisecond,
		MaxRetryBackoff: 500 * time.Millisecond,
	}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	w.Start(ctx)
	defer w.Stop()

	// 1. Success case
	jobID := uuid.New()
	repo.jobs[jobID] = &models.Job{
		ID:          jobID,
		UserID:      uuid.New(),
		JobType:     processor.JobTypeMockTranscription,
		Status:      models.JobStatusQueued,
		MaxAttempts: 3,
	}

	if err := q.Enqueue(ctx, jobID); err != nil {
		t.Fatalf("enqueue failed: %v", err)
	}

	// Wait for worker processing
	time.Sleep(150 * time.Millisecond)

	repo.mu.Lock()
	_, succeeded := repo.succeeded[jobID]
	status := repo.jobs[jobID].Status
	repo.mu.Unlock()

	if !succeeded || status != models.JobStatusSucceeded {
		t.Fatalf("expected job %s to succeed, got status: %s", jobID, status)
	}

	// 2. Transient error -> Retry scheduled
	transientJobID := uuid.New()
	repo.jobs[transientJobID] = &models.Job{
		ID:          transientJobID,
		UserID:      uuid.New(),
		JobType:     processor.JobTypeMockTranscription,
		Status:      models.JobStatusQueued,
		Payload:     models.JSONMap{"simulate_error": "transient"},
		MaxAttempts: 3,
	}

	_ = q.Enqueue(ctx, transientJobID)
	time.Sleep(150 * time.Millisecond)

	repo.mu.Lock()
	_, retried := repo.retries[transientJobID]
	transientStatus := repo.jobs[transientJobID].Status
	repo.mu.Unlock()

	if !retried || transientStatus != models.JobStatusRetryWait {
		t.Fatalf("expected transient job to be in retry_wait, got status %s", transientStatus)
	}

	// 3. Permanent error -> Terminal failure without retry
	permJobID := uuid.New()
	repo.jobs[permJobID] = &models.Job{
		ID:          permJobID,
		UserID:      uuid.New(),
		JobType:     processor.JobTypeMockTranscription,
		Status:      models.JobStatusQueued,
		Payload:     models.JSONMap{"simulate_error": "permanent"},
		MaxAttempts: 3,
	}

	_ = q.Enqueue(ctx, permJobID)
	time.Sleep(150 * time.Millisecond)

	repo.mu.Lock()
	errCode, failed := repo.failed[permJobID]
	permStatus := repo.jobs[permJobID].Status
	repo.mu.Unlock()

	if !failed || permStatus != models.JobStatusFailed {
		t.Fatalf("expected permanent job to be failed, got status %s (errCode: %s)", permStatus, errCode)
	}

	// 4. Cancellation before claim
	cancelJobID := uuid.New()
	now := time.Now()
	repo.jobs[cancelJobID] = &models.Job{
		ID:                cancelJobID,
		UserID:            uuid.New(),
		JobType:           processor.JobTypeMockTranscription,
		Status:            models.JobStatusCancelRequested,
		CancelRequestedAt: &now,
		MaxAttempts:       3,
	}

	_ = q.Enqueue(ctx, cancelJobID)
	time.Sleep(150 * time.Millisecond)

	repo.mu.Lock()
	cancelled := repo.cancelled[cancelJobID]
	cancelStatus := repo.jobs[cancelJobID].Status
	repo.mu.Unlock()

	if !cancelled || cancelStatus != models.JobStatusCancelled {
		t.Fatalf("expected cancelled job to acknowledge cancellation, got status %s", cancelStatus)
	}
}

func TestWorker_BackoffCalculation(t *testing.T) {
	minB := 1 * time.Second
	maxB := 30 * time.Second

	b1 := calculateBackoff(1, minB, maxB)
	if b1 < minB || b1 > 2*time.Second {
		t.Errorf("expected attempt 1 backoff near 1s, got %v", b1)
	}

	b2 := calculateBackoff(2, minB, maxB)
	if b2 < 2*time.Second || b2 > 4*time.Second {
		t.Errorf("expected attempt 2 backoff near 2s, got %v", b2)
	}

	bMax := calculateBackoff(10, minB, maxB)
	if bMax > maxB {
		t.Errorf("backoff exceeded max bounds: %v > %v", bMax, maxB)
	}
}
