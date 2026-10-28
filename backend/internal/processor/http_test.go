package processor

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestHTTPProcessor_ProcessSuccess(t *testing.T) {
	jobID := uuid.New()
	userID := uuid.New()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/process" {
			t.Errorf("expected path /process, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected method POST, got %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected application/json, got %s", r.Header.Get("Content-Type"))
		}

		var req ProcessRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		if err != nil {
			t.Fatalf("failed decoding request body: %v", err)
		}
		if req.JobID != jobID {
			t.Errorf("expected job ID %s, got %s", jobID, req.JobID)
		}
		if req.JobType != "llm_generate" {
			t.Errorf("expected job type llm_generate, got %s", req.JobType)
		}

		res := ProcessResult{
			Output: map[string]any{
				"generated_text": "EchoSelf is running.",
				"num_tokens":     float64(4),
			},
			Metadata: map[string]any{
				"execution_time_ms": float64(12.5),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(res)
	}))
	defer server.Close()

	proc := NewHTTPProcessor(HTTPProcessorConfig{
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	}, nil)

	ctx := context.Background()
	req := ProcessRequest{
		JobID:   jobID,
		JobType: "llm_generate",
		UserID:  userID,
		Payload: map[string]any{"prompt": "EchoSelf status"},
	}

	result, err := proc.Process(ctx, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if text, ok := result.Output["generated_text"].(string); !ok || text != "EchoSelf is running." {
		t.Errorf("expected 'EchoSelf is running.', got %v", result.Output["generated_text"])
	}
	if tokens, ok := result.Output["num_tokens"].(float64); !ok || tokens != 4 {
		t.Errorf("expected 4 tokens, got %v", result.Output["num_tokens"])
	}
}

func TestHTTPProcessor_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	proc := NewHTTPProcessor(HTTPProcessorConfig{
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := proc.Process(ctx, ProcessRequest{
		JobID:   uuid.New(),
		JobType: "llm_generate",
		UserID:  uuid.New(),
	})
	if err == nil {
		t.Fatal("expected error due to context cancellation, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Errorf("expected context cancellation error, got %v", err)
	}
}

func TestHTTPProcessor_ErrorMappings(t *testing.T) {
	tests := []struct {
		name        string
		statusCode  int
		response    string
		expectedErr error
	}{
		{
			name:        "400 Bad Request maps to ErrInvalidRequest",
			statusCode:  http.StatusBadRequest,
			response:    `{"error": "malformed JSON"}`,
			expectedErr: ErrInvalidRequest,
		},
		{
			name:        "422 Unprocessable Entity maps to ErrPermanentFailure",
			statusCode:  http.StatusUnprocessableEntity,
			response:    `{"error": "consent revoked"}`,
			expectedErr: ErrPermanentFailure,
		},
		{
			name:        "500 Internal Server Error maps to ErrTransientFailure",
			statusCode:  http.StatusInternalServerError,
			response:    `{"error": "cuda out of memory"}`,
			expectedErr: ErrTransientFailure,
		},
		{
			name:        "503 Service Unavailable maps to ErrTransientFailure",
			statusCode:  http.StatusServiceUnavailable,
			response:    `service overloaded`,
			expectedErr: ErrTransientFailure,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.statusCode)
				_, _ = w.Write([]byte(tc.response))
			}))
			defer server.Close()

			proc := NewHTTPProcessor(HTTPProcessorConfig{BaseURL: server.URL}, nil)
			_, err := proc.Process(context.Background(), ProcessRequest{
				JobID:   uuid.New(),
				JobType: "test_job",
				UserID:  uuid.New(),
			})
			if err == nil {
				t.Fatalf("expected error %v, got nil", tc.expectedErr)
			}
			if !errors.Is(err, tc.expectedErr) {
				t.Errorf("expected error %v, got %v", tc.expectedErr, err)
			}
		})
	}
}

func TestHTTPProcessor_HealthAndSupportedJobs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status":  "healthy",
				"version": "0.1.0",
			})
		case "/supported_jobs":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"supported_job_types": []string{"llm_generate", "voice_synthesis"},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	proc := NewHTTPProcessor(HTTPProcessorConfig{BaseURL: server.URL}, nil)

	// Test default supported jobs
	initialJobs := proc.SupportedJobTypes()
	foundLLM := false
	for _, j := range initialJobs {
		if j == "llm_generate" {
			foundLLM = true
			break
		}
	}
	if !foundLLM {
		t.Errorf("expected llm_generate in default supported jobs: %v", initialJobs)
	}

	// Test Health
	health, err := proc.Health(context.Background())
	if err != nil {
		t.Fatalf("unexpected error calling health: %v", err)
	}
	if status, ok := health["status"].(string); !ok || status != "healthy" {
		t.Errorf("expected status 'healthy', got %v", health["status"])
	}

	// Test RefreshSupportedJobTypes
	refreshed, err := proc.RefreshSupportedJobTypes(context.Background())
	if err != nil {
		t.Fatalf("unexpected error refreshing supported jobs: %v", err)
	}
	expected := []string{"llm_generate", "voice_synthesis"}
	if !reflect.DeepEqual(refreshed, expected) {
		t.Errorf("expected %v, got %v", expected, refreshed)
	}
	if !reflect.DeepEqual(proc.SupportedJobTypes(), expected) {
		t.Errorf("expected %v, got %v", expected, proc.SupportedJobTypes())
	}
}

func TestHTTPProcessor_UnreachableHost(t *testing.T) {
	proc := NewHTTPProcessor(HTTPProcessorConfig{
		BaseURL: "http://127.0.0.1:54321", // non-existent port
		Timeout: 200 * time.Millisecond,
	}, nil)

	_, err := proc.Process(context.Background(), ProcessRequest{
		JobID:   uuid.New(),
		JobType: "llm_generate",
		UserID:  uuid.New(),
	})
	if err == nil {
		t.Fatal("expected connection error, got nil")
	}
	if !errors.Is(err, ErrTransientFailure) {
		t.Errorf("expected ErrTransientFailure for connection error, got %v", err)
	}
	if !strings.Contains(err.Error(), "connection to AI service failed") {
		t.Errorf("expected connection error message, got %v", err)
	}
}
