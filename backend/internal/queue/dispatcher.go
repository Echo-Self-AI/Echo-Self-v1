package queue

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
)

type OutboxDispatcher struct {
	repo   *repositories.JobRepository
	queue  Queue
	logger *slog.Logger
}

func NewOutboxDispatcher(
	repo *repositories.JobRepository,
	q Queue,
	logger *slog.Logger,
) *OutboxDispatcher {
	if logger == nil {
		logger = slog.Default()
	}
	return &OutboxDispatcher{
		repo:   repo,
		queue:  q,
		logger: logger,
	}
}

// DispatchOnce processes one batch of pending outbox events.
func (d *OutboxDispatcher) DispatchOnce(ctx context.Context, batchSize int) (int, error) {
	if batchSize <= 0 {
		batchSize = 50
	}

	events, err := d.repo.GetPendingOutbox(ctx, batchSize)
	if err != nil {
		return 0, fmt.Errorf("fetch pending outbox: %w", err)
	}

	var dispatched int
	for _, event := range events {
		// Idempotently enqueue job ID to Redis
		if err := d.queue.Enqueue(ctx, event.JobID); err != nil {
			d.logger.Error("failed to enqueue outbox event to redis",
				"event_id", event.ID,
				"job_id", event.JobID,
				"error", err,
			)
			// Return early so we retry this event next round without dropping
			return dispatched, fmt.Errorf("enqueue outbox %s: %w", event.ID, err)
		}

		now := time.Now().UTC()
		if err := d.repo.MarkOutboxPublished(ctx, event.ID, now); err != nil {
			d.logger.Error("failed to mark outbox event published",
				"event_id", event.ID,
				"error", err,
			)
			return dispatched, fmt.Errorf("mark published %s: %w", event.ID, err)
		}

		dispatched++
	}

	return dispatched, nil
}

// Start runs the outbox polling loop until context cancellation.
func (d *OutboxDispatcher) Start(ctx context.Context, pollInterval time.Duration, batchSize int) {
	if pollInterval <= 0 {
		pollInterval = 500 * time.Millisecond
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	d.logger.Info("outbox dispatcher started", "poll_interval", pollInterval.String())

	for {
		select {
		case <-ctx.Done():
			d.logger.Info("outbox dispatcher stopping")
			return
		case <-ticker.C:
			if _, err := d.DispatchOnce(ctx, batchSize); err != nil {
				d.logger.Warn("outbox dispatch cycle encountered error", "error", err)
			}
		}
	}
}
