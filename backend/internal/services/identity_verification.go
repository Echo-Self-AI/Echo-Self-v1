package services

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
)

var (
	ErrIdentityVerificationNotConfigured = errors.New("identity verification provider is not configured")
)

type IdentityVerificationStatus string

const (
	IdentityStatusUnverified IdentityVerificationStatus = "unverified"
	IdentityStatusPending    IdentityVerificationStatus = "pending"
	IdentityStatusVerified   IdentityVerificationStatus = "verified"
	IdentityStatusRejected   IdentityVerificationStatus = "rejected"
)

// IdentityVerificationRecord defines the verification state of a real-world person.
// Crucial privacy rule: An authenticated account or verified email address does NOT constitute
// proof of real-world identity or likeness ownership.
type IdentityVerificationRecord struct {
	UserID      uuid.UUID                  `json:"user_id"`
	Status      IdentityVerificationStatus `json:"status"`
	Provider    string                     `json:"provider"`
	VerifiedAt  *time.Time                 `json:"verified_at,omitempty"`
	ReferenceID string                     `json:"reference_id,omitempty"`
}

// IdentityVerifier defines the interface for external biometric / identity verification providers.
type IdentityVerifier interface {
	GetVerificationStatus(ctx context.Context, userID uuid.UUID) (*IdentityVerificationRecord, error)
}
