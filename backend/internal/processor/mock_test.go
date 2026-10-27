package processor

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestMockProcessor_Success(t *testing.T) {
	ctx := context.Background()
	proc := NewMockProcessor()

	types := proc.SupportedJobTypes()
	if len(types) != 3 {
		t.Fatalf("expected 3 supported job types, got %d", len(types))
	}

	for _, jobType := range types {
		t.Run(jobType, func(t *testing.T) {
			jobID := uuid.New()
			req := ProcessRequest{
				JobID:   jobID,
				JobType: jobType,
				UserID:  uuid.New(),
			}

			res, err := proc.Process(ctx, req)
			if err != nil {
				t.Fatalf("unexpected error processing %s: %v", jobType, err)
			}
			if res == nil || res.Output == nil {
				t.Fatal("expected non-nil output")
			}
			if mockExec, ok := res.Metadata["mock_execution"].(bool); !ok || !mockExec {
				t.Fatal("expected mock_execution metadata to be true")
			}

			// Determinism check: processing again with identical JobID must return identical output
			res2, err := proc.Process(ctx, req)
			if err != nil {
				t.Fatalf("error on second run: %v", err)
			}
			if jobType == JobTypeMockTranscription {
				if res.Output["transcript"] != res2.Output["transcript"] {
					t.Fatalf("expected deterministic transcript output")
				}
			}
		})
	}
}

func TestMockProcessor_ErrorSimulation(t *testing.T) {
	ctx := context.Background()
	proc := NewMockProcessor()

	// 1. Unsupported job type
	_, err := proc.Process(ctx, ProcessRequest{
		JobID:   uuid.New(),
		JobType: "unknown_type",
	})
	if !errors.Is(err, ErrUnsupportedJobType) {
		t.Fatalf("expected ErrUnsupportedJobType, got %v", err)
	}

	// 2. Simulated transient error
	_, err = proc.Process(ctx, ProcessRequest{
		JobID:   uuid.New(),
		JobType: JobTypeMockVoiceEmbedding,
		Payload: map[string]any{"simulate_error": "transient"},
	})
	if !errors.Is(err, ErrTransientFailure) {
		t.Fatalf("expected ErrTransientFailure, got %v", err)
	}

	// 3. Simulated permanent error
	_, err = proc.Process(ctx, ProcessRequest{
		JobID:   uuid.New(),
		JobType: JobTypeMockVoiceEmbedding,
		Payload: map[string]any{"simulate_error": "permanent"},
	})
	if !errors.Is(err, ErrPermanentFailure) {
		t.Fatalf("expected ErrPermanentFailure, got %v", err)
	}

	// 4. Context cancellation
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel()
	_, err = proc.Process(cancelCtx, ProcessRequest{
		JobID:   uuid.New(),
		JobType: JobTypeMockVoiceEmbedding,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// 5. Simulated delay cancellation
	delayCtx, delayCancel := context.WithTimeout(ctx, 20*time.Millisecond)
	defer delayCancel()
	_, err = proc.Process(delayCtx, ProcessRequest{
		JobID:   uuid.New(),
		JobType: JobTypeMockVoiceEmbedding,
		Payload: map[string]any{"simulate_delay_ms": float64(200)},
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
}
