package queue

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

func setupTestRedis(t *testing.T) (*miniredis.Miniredis, *redis.Client) {
	t.Helper()
	s := miniredis.RunT(t)

	client := redis.NewClient(&redis.Options{
		Addr: s.Addr(),
	})

	t.Cleanup(func() {
		_ = client.Close()
		s.Close()
	})

	return s, client
}

func TestRedisQueue_EnqueueDequeue(t *testing.T) {
	_, client := setupTestRedis(t)
	ctx := context.Background()
	q := NewRedisQueue(client, "test:jobs")

	jobID := uuid.New()

	// Initially empty
	l, err := q.Len(ctx)
	if err != nil {
		t.Fatalf("queue len error: %v", err)
	}
	if l != 0 {
		t.Fatalf("expected len 0, got %d", l)
	}

	// Dequeue with small timeout on empty queue
	_, err = q.Dequeue(ctx, 50*time.Millisecond)
	if !errors.Is(err, ErrQueueEmpty) {
		t.Fatalf("expected ErrQueueEmpty, got %v", err)
	}

	// Enqueue
	if err := q.Enqueue(ctx, jobID); err != nil {
		t.Fatalf("enqueue error: %v", err)
	}

	l, err = q.Len(ctx)
	if err != nil || l != 1 {
		t.Fatalf("expected len 1, got %d (err: %v)", l, err)
	}

	// Dequeue
	popped, err := q.Dequeue(ctx, 1*time.Second)
	if err != nil {
		t.Fatalf("dequeue error: %v", err)
	}
	if popped != jobID {
		t.Fatalf("expected jobID %s, got %s", jobID, popped)
	}

	// Should be empty now
	l, _ = q.Len(ctx)
	if l != 0 {
		t.Fatalf("expected empty queue after dequeue, got len %d", l)
	}
}

func TestRedisClient_ConnectionError(t *testing.T) {
	// Attempt to connect to an unreachable port
	_, err := NewRedisClient(RedisConfig{
		URL: "redis://127.0.0.1:54321/0",
	})
	if err == nil {
		t.Fatal("expected error connecting to non-existent redis instance")
	}
}
