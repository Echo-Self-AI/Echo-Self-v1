package services

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
)

var (
	ErrInvalidConsentPurpose   = errors.New("invalid or unsupported consent purpose")
	ErrInvalidPolicyVersion    = errors.New("unsupported consent policy version")
	ErrConsentNotFound         = errors.New("consent record not found")
	ErrConsentAlreadyActive    = errors.New("active consent record already exists for this purpose")
	ErrConsentNotGranted       = errors.New("consent not granted for this purpose")
	ErrConsentRevokedOrExpired = errors.New("consent has been revoked or expired")
	ErrConsentPolicyOutdated   = errors.New("consent policy version is outdated; re-consent required")
	ErrUnauthorized            = errors.New("unauthorized")
)

const (
	PurposeVoiceDataProcessing    = "voice_data_processing"
	PurposeVoiceCloning           = "voice_cloning"
	PurposeFaceLikenessProcessing = "face_likeness_processing"
	PurposeFaceSynthesis          = "face_synthesis"
	PurposeAIPersonalization      = "ai_personalization"

	CurrentPolicyVersion = "v1.0"
)

var supportedPurposes = map[string]bool{
	PurposeVoiceDataProcessing:    true,
	PurposeVoiceCloning:           true,
	PurposeFaceLikenessProcessing: true,
	PurposeFaceSynthesis:          true,
	PurposeAIPersonalization:      true,
}

var supportedPolicyVersions = map[string]bool{
	"v1.0": true,
}

type GrantConsentInput struct {
	ConsentType   string     `json:"consent_type"`
	PolicyVersion string     `json:"policy_version"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
}

type RequestMetadata struct {
	IPAddress string
	UserAgent string
}

type ConsentResult struct {
	ID            string     `json:"id"`
	UserID        string     `json:"user_id"`
	ConsentType   string     `json:"consent_type"`
	Status        string     `json:"status"`
	PolicyVersion string     `json:"policy_version"`
	GrantedAt     time.Time  `json:"granted_at"`
	RevokedAt     *time.Time `json:"revoked_at,omitempty"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
}

type ConsentAuditLogResult struct {
	ID            string    `json:"id"`
	ConsentID     string    `json:"consent_id"`
	UserID        string    `json:"user_id"`
	Action        string    `json:"action"`
	ConsentType   string    `json:"consent_type"`
	PolicyVersion string    `json:"policy_version"`
	IPAddress     string    `json:"ip_address,omitempty"`
	UserAgent     string    `json:"user_agent,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
}

type ConsentCheckRequest struct {
	UserID          uuid.UUID
	ConsentType     string
	RequiredVersion string
}

type ConsentService struct {
	consents *repositories.ConsentRepository
}

func NewConsentService(consents *repositories.ConsentRepository) *ConsentService {
	return &ConsentService{consents: consents}
}

func toConsentResult(r *models.ConsentRecord) ConsentResult {
	return ConsentResult{
		ID:            r.ID.String(),
		UserID:        r.UserID.String(),
		ConsentType:   r.ConsentType,
		Status:        r.Status,
		PolicyVersion: r.PolicyVersion,
		GrantedAt:     r.GrantedAt,
		RevokedAt:     r.RevokedAt,
		ExpiresAt:     r.ExpiresAt,
		CreatedAt:     r.CreatedAt,
		UpdatedAt:     r.UpdatedAt,
	}
}

func (s *ConsentService) GrantConsent(
	ctx context.Context,
	userIDStr string,
	input GrantConsentInput,
	meta RequestMetadata,
) (*ConsentResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	purpose := strings.TrimSpace(input.ConsentType)
	if !supportedPurposes[purpose] {
		return nil, ErrInvalidConsentPurpose
	}

	policyVer := strings.TrimSpace(input.PolicyVersion)
	if !supportedPolicyVersions[policyVer] {
		return nil, ErrInvalidPolicyVersion
	}

	now := time.Now()
	if input.ExpiresAt != nil && !input.ExpiresAt.After(now) {
		return nil, errors.New("expiration date must be in the future")
	}

	record := &models.ConsentRecord{
		ID:            uuid.New(),
		UserID:        userID,
		ConsentType:   purpose,
		Status:        models.ConsentStatusGranted,
		PolicyVersion: policyVer,
		GrantedAt:     now,
		ExpiresAt:     input.ExpiresAt,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	audit := &models.ConsentAuditLog{
		ID:            uuid.New(),
		ConsentID:     record.ID,
		UserID:        userID,
		Action:        models.ConsentStatusGranted,
		ConsentType:   purpose,
		PolicyVersion: policyVer,
		IPAddress:     meta.IPAddress,
		UserAgent:     meta.UserAgent,
		CreatedAt:     now,
	}

	if err := s.consents.CreateConsent(ctx, record, audit); err != nil {
		if errors.Is(err, repositories.ErrConsentAlreadyActive) {
			return nil, ErrConsentAlreadyActive
		}
		return nil, fmt.Errorf("grant consent: %w", err)
	}

	result := toConsentResult(record)
	return &result, nil
}

func (s *ConsentService) RevokeConsent(
	ctx context.Context,
	userIDStr string,
	consentIDStr string,
	meta RequestMetadata,
) error {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return ErrUnauthorized
	}

	consentID, err := uuid.Parse(consentIDStr)
	if err != nil {
		return ErrConsentNotFound
	}

	now := time.Now()
	audit := &models.ConsentAuditLog{
		ID:        uuid.New(),
		IPAddress: meta.IPAddress,
		UserAgent: meta.UserAgent,
	}

	if err := s.consents.RevokeConsent(ctx, consentID, userID, now, audit); err != nil {
		if errors.Is(err, repositories.ErrConsentNotFound) {
			return ErrConsentNotFound
		}
		return fmt.Errorf("revoke consent: %w", err)
	}

	return nil
}

func (s *ConsentService) GetConsent(
	ctx context.Context,
	userIDStr string,
	consentIDStr string,
) (*ConsentResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	consentID, err := uuid.Parse(consentIDStr)
	if err != nil {
		return nil, ErrConsentNotFound
	}

	record, err := s.consents.FindByIDAndUserID(ctx, consentID, userID)
	if err != nil {
		return nil, fmt.Errorf("find consent: %w", err)
	}
	if record == nil {
		return nil, ErrConsentNotFound
	}

	result := toConsentResult(record)
	return &result, nil
}

func (s *ConsentService) ListConsents(
	ctx context.Context,
	userIDStr string,
) ([]ConsentResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	records, err := s.consents.ListByUserID(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("list consents: %w", err)
	}

	results := make([]ConsentResult, len(records))
	for i, r := range records {
		results[i] = toConsentResult(&r)
	}

	return results, nil
}

func (s *ConsentService) GetConsentAuditLogs(
	ctx context.Context,
	userIDStr string,
	consentIDStr string,
) ([]ConsentAuditLogResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	consentID, err := uuid.Parse(consentIDStr)
	if err != nil {
		return nil, ErrConsentNotFound
	}

	// Verify consent exists and belongs to the user
	existing, err := s.consents.FindByIDAndUserID(ctx, consentID, userID)
	if err != nil {
		return nil, fmt.Errorf("find consent: %w", err)
	}
	if existing == nil {
		return nil, ErrConsentNotFound
	}

	logs, err := s.consents.ListAuditLogsByConsentID(ctx, consentID, userID)
	if err != nil {
		return nil, fmt.Errorf("list consent audit logs: %w", err)
	}

	results := make([]ConsentAuditLogResult, len(logs))
	for i, log := range logs {
		results[i] = ConsentAuditLogResult{
			ID:            log.ID.String(),
			ConsentID:     log.ConsentID.String(),
			UserID:        log.UserID.String(),
			Action:        log.Action,
			ConsentType:   log.ConsentType,
			PolicyVersion: log.PolicyVersion,
			IPAddress:     log.IPAddress,
			UserAgent:     log.UserAgent,
			CreatedAt:     log.CreatedAt,
		}
	}

	return results, nil
}

// CheckConsent provides a fail-closed authorization check for AI and sensitive workflows.
// If any condition is unsatisfied or an error occurs, processing must NOT proceed.
func (s *ConsentService) CheckConsent(
	ctx context.Context,
	req ConsentCheckRequest,
) error {
	if req.UserID == uuid.Nil {
		return ErrUnauthorized
	}

	if !supportedPurposes[req.ConsentType] {
		return ErrInvalidConsentPurpose
	}

	record, err := s.consents.GetActiveConsent(ctx, req.UserID, req.ConsentType)
	if err != nil {
		return fmt.Errorf("verify consent: %w", err)
	}
	if record == nil {
		return ErrConsentNotGranted
	}

	if record.Status != models.ConsentStatusGranted {
		return ErrConsentRevokedOrExpired
	}

	if record.ExpiresAt != nil && !record.ExpiresAt.After(time.Now()) {
		return ErrConsentRevokedOrExpired
	}

	if req.RequiredVersion != "" && record.PolicyVersion != req.RequiredVersion {
		return ErrConsentPolicyOutdated
	}

	return nil
}
