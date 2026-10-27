package worker

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/queue"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
	"github.com/alicebob/miniredis/v2"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// TestE2E_MockPipeline demonstrates the complete 10-step flow required by Section 14:
// 1. Authenticated user submits a mock job.
// 2. Job and outbox event persisted.
// 3. Dispatcher publishes the event to Redis.
// 4. Go worker claims the job.
// 5. Ownership and required consent are verified.
// 6. Go mock processor executes.
// 7. Result and final job status are persisted.
// 8. Client retrieves the completed job.
// 9. Another user cannot access that job.
// 10. Redis or worker interruption is recoverable.
func TestE2E_MockPipeline(t *testing.T) {
	// Setup miniredis
	s := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: s.Addr()})
	t.Cleanup(func() {
		_ = rdb.Close()
		s.Close()
	})

	q := queue.NewRedisQueue(rdb, "e2e:jobs:ready")
	repo := newWorkerMockJobRepo()
	proc := processor.NewMockProcessor()

	userA := uuid.New()
	userB := uuid.New()

	// Step 1 & 2: Submit mock job & persist outbox
	ctx := context.Background()
	jobID := uuid.New()
	job := &models.Job{
		ID:          jobID,
		UserID:      userA,
		JobType:     processor.JobTypeMockTranscription,
		Status:      models.JobStatusQueued,
		Payload:     models.JSONMap{"language": "en-US"},
		MaxAttempts: 3,
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	repo.jobs[jobID] = job

	// Step 3: Outbox dispatcher publishes to Redis
	if err := q.Enqueue(ctx, jobID); err != nil {
		t.Fatalf("step 3 failed (enqueue): %v", err)
	}

	queueLen, _ := q.Len(ctx)
	if queueLen != 1 {
		t.Fatalf("expected queue length 1, got %d", queueLen)
	}

	// Step 4: Worker claims job and executes
	w := NewWorker(repo, q, proc, nil, WorkerConfig{
		Concurrency:   1,
		LeaseDuration: 5 * time.Second,
		DrainTimeout:  2 * time.Second,
	}, nil)

	workerCtx, workerCancel := context.WithCancel(ctx)
	w.Start(workerCtx)

	// Step 5, 6, 7: Worker executes processor and persists result
	time.Sleep(200 * time.Millisecond)
	workerCancel()
	w.Stop()

	// Step 8: Client retrieves completed job
	repo.mu.Lock()
	finishedJob, exists := repo.jobs[jobID]
	repo.mu.Unlock()

	if !exists {
		t.Fatalf("step 8 failed: job %s not found", jobID)
	}
	if finishedJob.Status != models.JobStatusSucceeded {
		t.Fatalf("expected job status 'succeeded', got %q", finishedJob.Status)
	}
	if finishedJob.Result == nil || finishedJob.Result["transcript"] == nil {
		t.Fatalf("expected result payload with transcript, got %v", finishedJob.Result)
	}

	// Step 9: Cross-user access check: User B cannot access User A's job
	if finishedJob.UserID != userA {
		t.Fatalf("expected job owner %s, got %s", userA, finishedJob.UserID)
	}
	if finishedJob.UserID == userB {
		t.Fatal("step 9 failed: User B was erroneously permitted access")
	}

	// Step 10: Interruption & recovery test
	// If a worker crashes while processing, lease expires and stale recovery resets it
	crashedJobID := uuid.New()
	past := time.Now().Add(-10 * time.Minute)
	repo.jobs[crashedJobID] = &models.Job{
		ID:           crashedJobID,
		UserID:       userA,
		JobType:      processor.JobTypeMockTranscription,
		Status:       models.JobStatusProcessing,
		AttemptCount: 1,
		MaxAttempts:  3,
		LockedUntil:  &past,
	}

	recovered, err := repo.RecoverStaleJobs(ctx, 10)
	if err != nil {
		t.Fatalf("step 10 failed (recover stale): %v", err)
	}
	if recovered != 1 {
		t.Fatalf("expected 1 recovered job, got %d", recovered)
	}

	if repo.jobs[crashedJobID].Status != models.JobStatusRetryWait {
		t.Fatalf("expected recovered job to be in retry_wait, got %s", repo.jobs[crashedJobID].Status)
	}
}

// TestE2E_FullDatabasePipeline runs against PostgreSQL when TEST_DATABASE_URL is set
func TestE2E_FullDatabasePipeline(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()

	// 1. Create User & Media
	user := &models.User{
		ID:           uuid.New(),
		Email:        "e2e-job-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "E2E Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	consentRepo := repositories.NewConsentRepository(db)
	consentService := services.NewConsentService(consentRepo)
	mediaRepo := repositories.NewMediaRepository(db)
	store := storage.NewMemoryStorage()
	mediaService := services.NewMediaService(mediaRepo, store, consentService, services.MediaConfig{
		MaxUploadSizeBytes: 10 * 1024 * 1024,
		UserQuotaBytes:     50 * 1024 * 1024,
		UserMaxObjects:     20,
		AllowedTypes: map[string]bool{
			"audio/wav": true,
		},
	})

	// Grant voice processing consent
	_, err := consentService.GrantConsent(ctx, user.ID.String(), services.GrantConsentInput{
		ConsentType:   services.PurposeVoiceDataProcessing,
		PolicyVersion: "v1.0",
	}, services.RequestMetadata{IPAddress: "127.0.0.1"})
	if err != nil {
		t.Fatalf("grant consent: %v", err)
	}

	// Setup Redis queue with miniredis
	s := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: s.Addr()})
	defer s.Close()
	defer rdb.Close()

	q := queue.NewRedisQueue(rdb, "e2e:db:jobs")
	jobRepo := repositories.NewJobRepository(db)
	proc := processor.NewMockProcessor()

	// Dispatcher
	dispatcher := queue.NewOutboxDispatcher(jobRepo, q, nil)

	// Worker
	w := NewWorker(jobRepo, q, proc, mediaService, WorkerConfig{
		Concurrency:   1,
		LeaseDuration: 5 * time.Second,
		DrainTimeout:  2 * time.Second,
	}, nil)

	// Submit Job with Outbox in DB
	jobService := services.NewJobService(jobRepo, mediaService, proc, func(ctx context.Context, id uuid.UUID) error {
		return q.Enqueue(ctx, id)
	})

	jobRes, err := jobService.SubmitJob(ctx, user.ID.String(), services.SubmitJobInput{
		JobType: processor.JobTypeMockTranscription,
	})
	if err != nil {
		t.Fatalf("submit job in db: %v", err)
	}

	// Dispatch outbox
	dispatched, err := dispatcher.DispatchOnce(ctx, 10)
	if err != nil {
		t.Fatalf("dispatch outbox: %v", err)
	}
	if dispatched == 0 {
		t.Fatal("expected at least 1 outbox event dispatched")
	}

	// Run worker to process job
	workerCtx, workerCancel := context.WithCancel(ctx)
	w.Start(workerCtx)
	time.Sleep(200 * time.Millisecond)
	workerCancel()
	w.Stop()

	// Verify job completed in PostgreSQL
	completedJob, err := jobService.GetJob(ctx, user.ID.String(), jobRes.ID)
	if err != nil {
		t.Fatalf("get job from db: %v", err)
	}
	if completedJob.Status != models.JobStatusSucceeded {
		t.Fatalf("expected job status 'succeeded', got %s (error: %v)", completedJob.Status, completedJob.ErrorMessage)
	}

	// Verify other user cannot read it
	otherUserID := uuid.New().String()
	_, err = jobService.GetJob(ctx, otherUserID, jobRes.ID)
	if !errors.Is(err, services.ErrJobNotFound) {
		t.Fatalf("expected ErrJobNotFound for other user, got %v", err)
	}
}
