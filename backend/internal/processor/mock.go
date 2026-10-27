package processor

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"time"
)

const (
	JobTypeMockVoiceEmbedding = "mock_voice_embedding"
	JobTypeMockFaceLandmarks  = "mock_face_landmarks"
	JobTypeMockTranscription  = "mock_audio_transcription"
)

// MockProcessor provides deterministic test execution without requiring
// external AI dependencies, GPUs, Python runtimes, or paid infrastructure.
type MockProcessor struct{}

func NewMockProcessor() *MockProcessor {
	return &MockProcessor{}
}

func (m *MockProcessor) SupportedJobTypes() []string {
	return []string{
		JobTypeMockVoiceEmbedding,
		JobTypeMockFaceLandmarks,
		JobTypeMockTranscription,
	}
}

func (m *MockProcessor) Process(ctx context.Context, req ProcessRequest) (*ProcessResult, error) {
	// Respect request context cancellation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 1. Verify job type support
	var supported bool
	for _, t := range m.SupportedJobTypes() {
		if t == req.JobType {
			supported = true
			break
		}
	}
	if !supported {
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedJobType, req.JobType)
	}

	// 2. Check for simulated error injection in payload
	if req.Payload != nil {
		if simErr, ok := req.Payload["simulate_error"].(string); ok {
			switch simErr {
			case "transient":
				return nil, ErrTransientFailure
			case "permanent":
				return nil, ErrPermanentFailure
			}
		}

		// Optional simulated processing delay
		if delayMs, ok := req.Payload["simulate_delay_ms"].(float64); ok && delayMs > 0 {
			select {
			case <-time.After(time.Duration(delayMs) * time.Millisecond):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	// 3. Generate deterministic result based on JobID hash
	hasher := sha256.New()
	hasher.Write([]byte(req.JobID.String()))
	hasher.Write([]byte(req.JobType))
	hashBytes := hasher.Sum(nil)

	seed := binary.BigEndian.Uint64(hashBytes[:8])

	output := make(map[string]any)
	switch req.JobType {
	case JobTypeMockVoiceEmbedding:
		// Generate deterministic 4-dimensional mock vector
		embedding := []float64{
			float64(seed%1000) / 1000.0,
			float64((seed>>8)%1000) / 1000.0,
			float64((seed>>16)%1000) / 1000.0,
			float64((seed>>24)%1000) / 1000.0,
		}
		output["embedding"] = embedding
		output["sample_rate_hz"] = 16000
		output["duration_seconds"] = 3.5

	case JobTypeMockFaceLandmarks:
		// Generate deterministic bounding box and landmark points
		output["bounding_box"] = map[string]any{
			"top":    int((seed % 100) + 50),
			"left":   int(((seed >> 8) % 100) + 50),
			"width":  200,
			"height": 200,
		}
		output["landmarks_detected"] = 68
		output["confidence_score"] = 0.985

	case JobTypeMockTranscription:
		output["transcript"] = fmt.Sprintf("EchoSelf deterministic transcript for job %s", req.JobID.String())
		output["word_count"] = 6
		output["language"] = "en-US"
	}

	metadata := map[string]any{
		"mock_execution": true,
		"engine_version": "v1.0",
		"provider":       "echoself_mock_engine",
		"processed_at":   time.Now().UTC().Format(time.RFC3339),
	}

	return &ProcessResult{
		Output:   output,
		Metadata: metadata,
	}, nil
}
