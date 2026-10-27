package queue

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

var (
	ErrQueueEmpty = errors.New("queue: no messages available within timeout")
)

type Queue interface {
	Enqueue(ctx context.Context, jobID uuid.UUID) error
	Dequeue(ctx context.Context, timeout time.Duration) (uuid.UUID, error)
	Len(ctx context.Context) (int64, error)
}

type RedisQueue struct {
	client   *redis.Client
	queueKey string
}

func NewRedisQueue(client *redis.Client, queueKey string) *RedisQueue {
	if queueKey == "" {
		queueKey = "echoself:jobs:ready"
	}
	return &RedisQueue{
		client:   client,
		queueKey: queueKey,
	}
}

// Enqueue pushes a job ID onto the head of the Redis list.
func (q *RedisQueue) Enqueue(ctx context.Context, jobID uuid.UUID) error {
	return q.client.LPush(ctx, q.queueKey, jobID.String()).Err()
}

// Dequeue pops a job ID from the tail of the Redis list with blocking timeout.
func (q *RedisQueue) Dequeue(ctx context.Context, timeout time.Duration) (uuid.UUID, error) {
	if timeout < time.Second {
		timeout = time.Second
	}
	res, err := q.client.BRPop(ctx, timeout, q.queueKey).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return uuid.Nil, ErrQueueEmpty
		}
		return uuid.Nil, fmt.Errorf("dequeue failed: %w", err)
	}

	if len(res) < 2 {
		return uuid.Nil, ErrQueueEmpty
	}

	jobID, err := uuid.Parse(res[1])
	if err != nil {
		return uuid.Nil, fmt.Errorf("invalid uuid in queue: %w", err)
	}

	return jobID, nil
}

// Len returns the current count of elements in the queue.
func (q *RedisQueue) Len(ctx context.Context) (int64, error) {
	return q.client.LLen(ctx, q.queueKey).Result()
}
