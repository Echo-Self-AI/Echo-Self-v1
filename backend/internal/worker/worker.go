package worker

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/queue"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/google/uuid"
)

type WorkerConfig struct {
	WorkerID            string
	Concurrency         int
	LeaseDuration       time.Duration
	DrainTimeout        time.Duration
	MinRetryBackoff     time.Duration
	MaxRetryBackoff     time.Duration
	StaleRecoveryPeriod time.Duration
}

type JobRepository interface {
	ClaimJob(ctx context.Context, id uuid.UUID, workerID string, leaseDuration time.Duration) (*models.Job, error)
	RenewLease(ctx context.Context, id uuid.UUID, workerID string, leaseDuration time.Duration) error
	MarkSucceeded(ctx context.Context, id uuid.UUID, workerID string, result models.JSONMap) error
	MarkFailed(ctx context.Context, id uuid.UUID, workerID string, errCode, errMsg string) error
	ScheduleRetry(ctx context.Context, id uuid.UUID, workerID string, retryAt time.Time, errCode, errMsg string) error
	FindByID(ctx context.Context, id uuid.UUID) (*models.Job, error)
	AcknowledgeCancellation(ctx context.Context, id uuid.UUID, workerID string) error
	RecoverStaleJobs(ctx context.Context, limit int) (int64, error)
}

type Worker struct {
	repo         JobRepository
	queue        queue.Queue
	processor    processor.Processor
	mediaService *services.MediaService
	cfg          WorkerConfig
	logger       *slog.Logger
	wg           sync.WaitGroup
	stopChan     chan struct{}
	stopOnce     sync.Once
}

func NewWorker(
	repo JobRepository,
	q queue.Queue,
	proc processor.Processor,
	mediaService *services.MediaService,
	cfg WorkerConfig,
	logger *slog.Logger,
) *Worker {
	if cfg.WorkerID == "" {
		cfg.WorkerID = "worker-" + uuid.NewString()[:8]
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 2
	}
	if cfg.LeaseDuration <= 0 {
		cfg.LeaseDuration = 30 * time.Second
	}
	if cfg.DrainTimeout <= 0 {
		cfg.DrainTimeout = 10 * time.Second
	}
	if cfg.MinRetryBackoff <= 0 {
		cfg.MinRetryBackoff = 2 * time.Second
	}
	if cfg.MaxRetryBackoff <= 0 {
		cfg.MaxRetryBackoff = 60 * time.Second
	}
	if cfg.StaleRecoveryPeriod <= 0 {
		cfg.StaleRecoveryPeriod = 15 * time.Second
	}
	if logger == nil {
		logger = slog.Default()
	}

	return &Worker{
		repo:         repo,
		queue:        q,
		processor:    proc,
		mediaService: mediaService,
		cfg:          cfg,
		logger:       logger.With("worker_id", cfg.WorkerID),
		stopChan:     make(chan struct{}),
	}
}

// Start launches worker goroutines and the stale recovery sweeper.
func (w *Worker) Start(ctx context.Context) {
	w.logger.Info("starting background workers",
		"concurrency", w.cfg.Concurrency,
		"lease_duration", w.cfg.LeaseDuration.String(),
	)

	// Launch stale job recovery sweeper
	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		w.runStaleRecovery(ctx)
	}()

	// Launch consumer worker pool
	for i := 0; i < w.cfg.Concurrency; i++ {
		w.wg.Add(1)
		go func(workerIndex int) {
			defer w.wg.Done()
			w.runConsumer(ctx, workerIndex)
		}(i)
	}
}

// Stop initiates graceful shutdown and waits up to drainTimeout.
func (w *Worker) Stop() {
	w.stopOnce.Do(func() {
		w.logger.Info("stopping workers, entering drain period",
			"drain_timeout", w.cfg.DrainTimeout.String(),
		)
		close(w.stopChan)
	})

	c := make(chan struct{})
	go func() {
		defer close(c)
		w.wg.Wait()
	}()

	select {
	case <-c:
		w.logger.Info("all workers stopped cleanly")
	case <-time.After(w.cfg.DrainTimeout):
		w.logger.Warn("drain timeout expired; forcing stop")
	}
}

func (w *Worker) runConsumer(ctx context.Context, index int) {
	workerLog := w.logger.With("consumer_index", index)
	workerLog.Debug("consumer thread started")

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopChan:
			return
		default:
			// Block on Redis queue with 1-second timeout
			jobID, err := w.queue.Dequeue(ctx, 1*time.Second)
			if err != nil {
				if errors.Is(err, queue.ErrQueueEmpty) {
					continue
				}
				if errors.Is(err, context.Canceled) {
					return
				}
				workerLog.Warn("queue dequeue error", "error", err)
				time.Sleep(500 * time.Millisecond)
				continue
			}

			w.processJob(ctx, jobID)
		}
	}
}

func (w *Worker) processJob(ctx context.Context, jobID uuid.UUID) {
	jobLog := w.logger.With("job_id", jobID.String())

	// 1. Conditionally claim the job lease in PostgreSQL
	job, err := w.repo.ClaimJob(ctx, jobID, w.cfg.WorkerID, w.cfg.LeaseDuration)
	if err != nil {
		if errors.Is(err, repositories.ErrJobAlreadyClaimed) ||
			errors.Is(err, repositories.ErrJobNotFound) ||
			errors.Is(err, repositories.ErrTerminalState) {
			jobLog.Debug("job is not claimable, skipping", "reason", err.Error())
			return
		}
		jobLog.Error("failed to claim job in postgres", "error", err)
		return
	}

	jobLog.Info("claimed job for processing",
		"attempt", job.AttemptCount,
		"max_attempts", job.MaxAttempts,
		"job_type", job.JobType,
	)

	// 2. Cancellation Check: Did the user request cancellation while queued or running?
	if job.Status == models.JobStatusCancelRequested || job.CancelRequestedAt != nil {
		jobLog.Info("job cancellation requested, acknowledging cancellation")
		_ = w.repo.AcknowledgeCancellation(ctx, job.ID, w.cfg.WorkerID)
		return
	}

	// 3. Consent Re-Verification: Recheck purpose-specific consent immediately before processing
	if job.InputMediaID != nil && w.mediaService != nil {
		purpose := resolveConsentPurpose(job.JobType)
		if err := w.mediaService.VerifyProcessingConsent(ctx, job.UserID.String(), job.InputMediaID.String(), purpose); err != nil {
			jobLog.Warn("consent verification failed before processing boundary",
				"purpose", purpose,
				"error", err,
			)
			// Permanent failure: Consent not granted or revoked -> do NOT retry!
			_ = w.repo.MarkFailed(ctx, job.ID, w.cfg.WorkerID, "CONSENT_REVOKED", "Consent not granted or revoked for this operation")
			return
		}
	}

	// 4. Start lease renewal heartbeat
	procCtx, procCancel := context.WithCancel(ctx)
	defer procCancel()

	renewDone := make(chan struct{})
	go w.heartbeatLease(procCtx, job.ID, renewDone)
	defer func() {
		close(renewDone)
	}()

	// 5. Execute processing interface
	req := processor.ProcessRequest{
		JobID:   job.ID,
		JobType: job.JobType,
		UserID:  job.UserID,
		MediaID: job.InputMediaID,
		Payload: job.Payload,
	}

	res, err := w.processor.Process(procCtx, req)

	// Check if cancellation arrived during processing
	latest, _ := w.repo.FindByID(ctx, job.ID)
	if latest != nil && (latest.Status == models.JobStatusCancelRequested || latest.CancelRequestedAt != nil) {
		jobLog.Info("cancellation acknowledged during processing completion")
		_ = w.repo.AcknowledgeCancellation(ctx, job.ID, w.cfg.WorkerID)
		return
	}

	// 6. Handle Outcome
	if err == nil {
		// Succeeded: Persist result
		if err := w.repo.MarkSucceeded(ctx, job.ID, w.cfg.WorkerID, res.Output); err != nil {
			jobLog.Error("failed to mark job succeeded in db", "error", err)
		} else {
			jobLog.Info("job succeeded")
		}
		return
	}

	// Error handling
	jobLog.Warn("processor execution failed", "error", err)

	// Permanent failures: do NOT retry
	if errors.Is(err, processor.ErrPermanentFailure) || errors.Is(err, processor.ErrUnsupportedJobType) {
		_ = w.repo.MarkFailed(ctx, job.ID, w.cfg.WorkerID, "PERMANENT_FAILURE", err.Error())
		return
	}

	// Check if attempts exhausted
	if job.AttemptCount >= job.MaxAttempts {
		jobLog.Error("job exceeded max attempts, marking failed", "attempts", job.AttemptCount)
		_ = w.repo.MarkFailed(ctx, job.ID, w.cfg.WorkerID, "MAX_ATTEMPTS_EXCEEDED", "Maximum retry attempts exceeded")
		return
	}

	// Transient error: Schedule retry with exponential backoff and jitter
	backoff := calculateBackoff(job.AttemptCount, w.cfg.MinRetryBackoff, w.cfg.MaxRetryBackoff)
	retryAt := time.Now().UTC().Add(backoff)
	jobLog.Info("scheduling retry for transient error",
		"next_attempt_in", backoff.String(),
		"retry_at", retryAt.Format(time.RFC3339),
	)

	_ = w.repo.ScheduleRetry(ctx, job.ID, w.cfg.WorkerID, retryAt, "TRANSIENT_FAILURE", err.Error())
}

func (w *Worker) heartbeatLease(ctx context.Context, jobID uuid.UUID, done chan struct{}) {
	interval := w.cfg.LeaseDuration / 3
	if interval < 2*time.Second {
		interval = 2 * time.Second
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-done:
			return
		case <-ticker.C:
			if err := w.repo.RenewLease(ctx, jobID, w.cfg.WorkerID, w.cfg.LeaseDuration); err != nil {
				w.logger.Warn("failed to renew job lease", "job_id", jobID, "error", err)
				return
			}
		}
	}
}

func (w *Worker) runStaleRecovery(ctx context.Context) {
	ticker := time.NewTicker(w.cfg.StaleRecoveryPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopChan:
			return
		case <-ticker.C:
			recovered, err := w.repo.RecoverStaleJobs(ctx, 50)
			if err != nil {
				w.logger.Error("stale job recovery error", "error", err)
			} else if recovered > 0 {
				w.logger.Info("recovered stale unfinished jobs", "count", recovered)
			}
		}
	}
}

func resolveConsentPurpose(jobType string) string {
	switch jobType {
	case processor.JobTypeMockVoiceEmbedding, processor.JobTypeMockTranscription:
		return services.PurposeVoiceDataProcessing
	case processor.JobTypeMockFaceLandmarks:
		return services.PurposeFaceLikenessProcessing
	default:
		return services.PurposeAIPersonalization
	}
}

func calculateBackoff(attempt int, minBackoff, maxBackoff time.Duration) time.Duration {
	base := float64(minBackoff) * math.Pow(2, float64(attempt-1))
	if base > float64(maxBackoff) {
		base = float64(maxBackoff)
	}

	// Add 10-25% random jitter
	jitter := base * (0.1 + rand.Float64()*0.15)
	backoff := time.Duration(base + jitter)
	if backoff > maxBackoff {
		backoff = maxBackoff
	}
	return backoff
}
