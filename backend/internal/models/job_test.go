package models

import (
	"testing"
)

func TestJobStatusTransitions(t *testing.T) {
	tests := []struct {
		name    string
		from    string
		to      string
		allowed bool
	}{
		// Self transition is no-op / allowed
		{"queued to queued", JobStatusQueued, JobStatusQueued, true},

		// Queued transitions
		{"queued to processing", JobStatusQueued, JobStatusProcessing, true},
		{"queued to cancel_requested", JobStatusQueued, JobStatusCancelRequested, true},
		{"queued to cancelled", JobStatusQueued, JobStatusCancelled, true},
		{"queued to succeeded directly", JobStatusQueued, JobStatusSucceeded, false},
		{"queued to failed directly", JobStatusQueued, JobStatusFailed, false},

		// Processing transitions
		{"processing to succeeded", JobStatusProcessing, JobStatusSucceeded, true},
		{"processing to failed", JobStatusProcessing, JobStatusFailed, true},
		{"processing to retry_wait", JobStatusProcessing, JobStatusRetryWait, true},
		{"processing to cancel_requested", JobStatusProcessing, JobStatusCancelRequested, true},
		{"processing to cancelled", JobStatusProcessing, JobStatusCancelled, true},
		{"processing to queued", JobStatusProcessing, JobStatusQueued, false},

		// RetryWait transitions
		{"retry_wait to processing", JobStatusRetryWait, JobStatusProcessing, true},
		{"retry_wait to queued", JobStatusRetryWait, JobStatusQueued, true},
		{"retry_wait to cancel_requested", JobStatusRetryWait, JobStatusCancelRequested, true},
		{"retry_wait to cancelled", JobStatusRetryWait, JobStatusCancelled, true},
		{"retry_wait to succeeded", JobStatusRetryWait, JobStatusSucceeded, false},

		// CancelRequested transitions
		{"cancel_requested to cancelled", JobStatusCancelRequested, JobStatusCancelled, true},
		{"cancel_requested to failed", JobStatusCancelRequested, JobStatusFailed, true},
		{"cancel_requested to succeeded (race condition)", JobStatusCancelRequested, JobStatusSucceeded, true},
		{"cancel_requested to processing", JobStatusCancelRequested, JobStatusProcessing, false},

		// Terminal states cannot transition to anything
		{"succeeded to processing", JobStatusSucceeded, JobStatusProcessing, false},
		{"succeeded to failed", JobStatusSucceeded, JobStatusFailed, false},
		{"failed to processing", JobStatusFailed, JobStatusProcessing, false},
		{"failed to retry_wait", JobStatusFailed, JobStatusRetryWait, false},
		{"cancelled to queued", JobStatusCancelled, JobStatusQueued, false},
		{"cancelled to processing", JobStatusCancelled, JobStatusProcessing, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsValidStatusTransition(tt.from, tt.to)
			if got != tt.allowed {
				t.Errorf("IsValidStatusTransition(%q, %q) = %v, want %v", tt.from, tt.to, got, tt.allowed)
			}
		})
	}
}

func TestIsTerminalStatus(t *testing.T) {
	terminals := []string{JobStatusSucceeded, JobStatusFailed, JobStatusCancelled}
	for _, s := range terminals {
		if !IsTerminalStatus(s) {
			t.Errorf("expected %s to be terminal", s)
		}
	}

	nonTerminals := []string{JobStatusQueued, JobStatusProcessing, JobStatusRetryWait, JobStatusCancelRequested}
	for _, s := range nonTerminals {
		if IsTerminalStatus(s) {
			t.Errorf("expected %s to NOT be terminal", s)
		}
	}
}
