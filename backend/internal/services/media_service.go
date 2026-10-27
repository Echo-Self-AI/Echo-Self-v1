package services

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
	"github.com/gabriel-vasile/mimetype"
	"github.com/google/uuid"
)

var (
	ErrMediaNotFound        = errors.New("media not found")
	ErrInvalidCategory      = errors.New("invalid or unsupported media category")
	ErrUnsupportedMediaType = errors.New("unsupported or mismatched media type")
	ErrUploadTooLarge       = errors.New("upload exceeds maximum allowed file size")
	ErrQuotaExceeded        = errors.New("user storage quota or object limit exceeded")
	ErrEmptyFile            = errors.New("file cannot be empty")
	ErrMalformedUpload      = errors.New("malformed upload payload")
	ErrIncompatiblePurpose  = errors.New("requested consent purpose is incompatible with media category")
)

type MediaConfig struct {
	MaxUploadSizeBytes int64
	UserQuotaBytes     int64
	UserMaxObjects     int
	AllowedTypes       map[string]bool
}

type UploadMediaInput struct {
	Category     string
	Filename     string
	DeclaredSize int64
	Reader       io.Reader
}

type MediaResult struct {
	ID               string    `json:"id"`
	UserID           string    `json:"user_id"`
	Category         string    `json:"category"`
	ContentType      string    `json:"content_type"`
	FileSize         int64     `json:"file_size"`
	OriginalFilename string    `json:"original_filename"`
	Status           string    `json:"status"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

type MediaRepository interface {
	Create(ctx context.Context, media *models.Media) error
	CreateWithQuotaCheck(ctx context.Context, media *models.Media, maxQuotaBytes int64, maxObjects int) error
	FindByIDAndUserID(ctx context.Context, id uuid.UUID, userID uuid.UUID) (*models.Media, error)
	ListByUserID(ctx context.Context, userID uuid.UUID, category string) ([]models.Media, error)
	GetUserStorageUsage(ctx context.Context, userID uuid.UUID) (totalBytes int64, objectCount int64, err error)
	SoftDelete(ctx context.Context, id uuid.UUID, userID uuid.UUID, now time.Time) error
	MarkStatus(ctx context.Context, id uuid.UUID, userID uuid.UUID, status string) error
	HardDelete(ctx context.Context, id uuid.UUID) error
	ListPendingDeletions(ctx context.Context, limit int) ([]models.Media, error)
}

type MediaService struct {
	repo     MediaRepository
	storage  storage.Storage
	consents *ConsentService
	cfg      MediaConfig
}

func NewMediaService(
	repo MediaRepository,
	store storage.Storage,
	consents *ConsentService,
	cfg MediaConfig,
) *MediaService {
	return &MediaService{
		repo:     repo,
		storage:  store,
		consents: consents,
		cfg:      cfg,
	}
}

func toMediaResult(m *models.Media) MediaResult {
	return MediaResult{
		ID:               m.ID.String(),
		UserID:           m.UserID.String(),
		Category:         m.Category,
		ContentType:      m.ContentType,
		FileSize:         m.FileSize,
		OriginalFilename: m.OriginalFilename,
		Status:           m.Status,
		CreatedAt:        m.CreatedAt,
		UpdatedAt:        m.UpdatedAt,
	}
}

func sanitizeFilename(name string) string {
	base := filepath.Base(name)
	// Replace control characters and directory separators
	clean := strings.Map(func(r rune) rune {
		if r < 32 || r == '/' || r == '\\' {
			return '_'
		}
		return r
	}, base)
	clean = strings.TrimSpace(clean)
	if clean == "" || clean == "." || clean == ".." {
		return "unnamed_media"
	}
	if len(clean) > 200 {
		clean = clean[:200]
	}
	return clean
}

func (s *MediaService) UploadMedia(
	ctx context.Context,
	userIDStr string,
	input UploadMediaInput,
) (*MediaResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	category := strings.ToLower(strings.TrimSpace(input.Category))
	switch category {
	case models.MediaCategoryVoiceSample, models.MediaCategoryFaceSample, models.MediaCategoryGeneral:
	default:
		return nil, ErrInvalidCategory
	}

	if input.DeclaredSize > s.cfg.MaxUploadSizeBytes {
		return nil, ErrUploadTooLarge
	}

	// Pre-upload quota check
	usedBytes, count, err := s.repo.GetUserStorageUsage(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("check user storage quota: %w", err)
	}

	if count >= int64(s.cfg.UserMaxObjects) {
		return nil, fmt.Errorf("%w: max objects reached (%d)", ErrQuotaExceeded, s.cfg.UserMaxObjects)
	}

	if usedBytes+input.DeclaredSize > s.cfg.UserQuotaBytes {
		return nil, fmt.Errorf("%w: storage quota reached (%d bytes)", ErrQuotaExceeded, s.cfg.UserQuotaBytes)
	}

	// Read magic bytes to reliably sniff content type without trusting client header
	headerBuf := make([]byte, 512)
	n, err := io.ReadFull(input.Reader, headerBuf)
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("%w: %v", ErrMalformedUpload, err)
	}
	if n == 0 {
		return nil, ErrEmptyFile
	}

	headerBytes := headerBuf[:n]
	mime := mimetype.Detect(headerBytes)
	detectedType := mime.String()
	// Strip optional MIME parameters (e.g., charset=utf-8)
	if idx := strings.Index(detectedType, ";"); idx != -1 {
		detectedType = strings.TrimSpace(detectedType[:idx])
	}

	if !s.cfg.AllowedTypes[detectedType] {
		return nil, fmt.Errorf("%w: detected %q not in allowed types", ErrUnsupportedMediaType, detectedType)
	}

	// Validate category-to-content-type constraints
	if category == models.MediaCategoryVoiceSample && !strings.HasPrefix(detectedType, "audio/") {
		return nil, fmt.Errorf("%w: voice samples must be audio files, detected %q", ErrUnsupportedMediaType, detectedType)
	}
	if category == models.MediaCategoryFaceSample && !strings.HasPrefix(detectedType, "image/") {
		return nil, fmt.Errorf("%w: face samples must be image files, detected %q", ErrUnsupportedMediaType, detectedType)
	}

	// Reassemble reader with sniffed header and enforce upload size limit
	combinedReader := io.MultiReader(bytes.NewReader(headerBytes), input.Reader)
	limitedReader := io.LimitReader(combinedReader, s.cfg.MaxUploadSizeBytes+1)

	// Stream buffer to measure exact bytes safely
	var fullBuf bytes.Buffer
	written, err := io.Copy(&fullBuf, limitedReader)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMalformedUpload, err)
	}
	if written > s.cfg.MaxUploadSizeBytes {
		return nil, ErrUploadTooLarge
	}
	if written == 0 {
		return nil, ErrEmptyFile
	}

	// Re-verify quota against actual uploaded size
	if usedBytes+written > s.cfg.UserQuotaBytes {
		return nil, fmt.Errorf("%w: storage quota reached (%d bytes)", ErrQuotaExceeded, s.cfg.UserQuotaBytes)
	}

	cleanFilename := sanitizeFilename(input.Filename)
	mediaID := uuid.New()
	storageKey := fmt.Sprintf("media/%s/%s", userID.String(), mediaID.String())

	// Upload to object storage
	uploadErr := s.storage.Upload(
		ctx,
		storageKey,
		bytes.NewReader(fullBuf.Bytes()),
		written,
		detectedType,
	)
	if uploadErr != nil {
		return nil, fmt.Errorf("storage upload: %w", uploadErr)
	}

	// Persist metadata to database
	now := time.Now()
	media := &models.Media{
		ID:               mediaID,
		UserID:           userID,
		Category:         category,
		StorageKey:       storageKey,
		ContentType:      detectedType,
		FileSize:         written,
		OriginalFilename: cleanFilename,
		Status:           models.MediaStatusReady,
		CreatedAt:        now,
		UpdatedAt:        now,
	}

	if err := s.repo.CreateWithQuotaCheck(ctx, media, s.cfg.UserQuotaBytes, s.cfg.UserMaxObjects); err != nil {
		// Compensating transaction: purge uploaded object if database persistence fails or quota exceeded
		_ = s.storage.Delete(ctx, storageKey)
		if errors.Is(err, repositories.ErrQuotaExceeded) {
			return nil, fmt.Errorf("%w: quota reached during concurrent upload", ErrQuotaExceeded)
		}
		return nil, fmt.Errorf("persist media metadata: %w", err)
	}

	result := toMediaResult(media)
	return &result, nil
}

func (s *MediaService) GetMedia(
	ctx context.Context,
	userIDStr string,
	mediaIDStr string,
) (*MediaResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	mediaID, err := uuid.Parse(mediaIDStr)
	if err != nil {
		return nil, ErrMediaNotFound
	}

	record, err := s.repo.FindByIDAndUserID(ctx, mediaID, userID)
	if err != nil {
		return nil, fmt.Errorf("find media: %w", err)
	}
	if record == nil || record.DeletedAt.Valid || record.Status == models.MediaStatusDeleted {
		return nil, ErrMediaNotFound
	}

	result := toMediaResult(record)
	return &result, nil
}

func (s *MediaService) ListMedia(
	ctx context.Context,
	userIDStr string,
	category string,
) ([]MediaResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUnauthorized
	}

	records, err := s.repo.ListByUserID(ctx, userID, category)
	if err != nil {
		return nil, fmt.Errorf("list media: %w", err)
	}

	results := make([]MediaResult, 0, len(records))
	for _, r := range records {
		if r.DeletedAt.Valid || r.Status == models.MediaStatusDeleted {
			continue
		}
		results = append(results, toMediaResult(&r))
	}

	return results, nil
}

func (s *MediaService) StreamMediaContent(
	ctx context.Context,
	userIDStr string,
	mediaIDStr string,
) (io.ReadCloser, *MediaResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, nil, ErrUnauthorized
	}

	mediaID, err := uuid.Parse(mediaIDStr)
	if err != nil {
		return nil, nil, ErrMediaNotFound
	}

	record, err := s.repo.FindByIDAndUserID(ctx, mediaID, userID)
	if err != nil {
		return nil, nil, fmt.Errorf("find media: %w", err)
	}
	if record == nil || record.DeletedAt.Valid || record.Status == models.MediaStatusDeleted {
		return nil, nil, ErrMediaNotFound
	}

	stream, _, _, err := s.storage.Download(ctx, record.StorageKey)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotFound) {
			return nil, nil, ErrMediaNotFound
		}
		return nil, nil, fmt.Errorf("download media: %w", err)
	}

	res := toMediaResult(record)
	return stream, &res, nil
}

func (s *MediaService) DeleteMedia(
	ctx context.Context,
	userIDStr string,
	mediaIDStr string,
) error {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return ErrUnauthorized
	}

	mediaID, err := uuid.Parse(mediaIDStr)
	if err != nil {
		return ErrMediaNotFound
	}

	record, err := s.repo.FindByIDAndUserID(ctx, mediaID, userID)
	if err != nil {
		return fmt.Errorf("find media: %w", err)
	}
	if record == nil || record.DeletedAt.Valid || record.Status == models.MediaStatusDeleted {
		return ErrMediaNotFound
	}

	now := time.Now()
	// Soft-delete and mark deletion pending
	if err := s.repo.SoftDelete(ctx, mediaID, userID, now); err != nil {
		return fmt.Errorf("soft delete media: %w", err)
	}

	// Attempt deletion from storage
	if err := s.storage.Delete(ctx, record.StorageKey); err != nil {
		// Log or retain as deletion_pending for asynchronous cleanup worker
		return nil
	}

	// If storage deletion succeeded, mark status as deleted
	_ = s.repo.MarkStatus(ctx, mediaID, userID, models.MediaStatusDeleted)
	return nil
}

// ProcessPendingDeletions scans for media marked deletion_pending and cleans them up from storage.
func (s *MediaService) ProcessPendingDeletions(ctx context.Context, batchSize int) (int, error) {
	if batchSize <= 0 {
		batchSize = 50
	}
	pending, err := s.repo.ListPendingDeletions(ctx, batchSize)
	if err != nil {
		return 0, fmt.Errorf("list pending deletions: %w", err)
	}

	var cleaned int
	for _, m := range pending {
		delErr := s.storage.Delete(ctx, m.StorageKey)
		if delErr != nil && !errors.Is(delErr, storage.ErrObjectNotFound) {
			// Failed to delete from storage; retain for next retry cycle
			continue
		}

		if err := s.repo.MarkStatus(ctx, m.ID, m.UserID, models.MediaStatusDeleted); err != nil {
			continue
		}
		cleaned++
	}

	return cleaned, nil
}

// VerifyProcessingConsent validates purpose-specific consent before any future AI processing begins.
// Uploading/storing private media does not grant processing rights.
func (s *MediaService) VerifyProcessingConsent(
	ctx context.Context,
	userIDStr string,
	mediaIDStr string,
	purpose string,
) error {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return ErrUnauthorized
	}

	mediaID, err := uuid.Parse(mediaIDStr)
	if err != nil {
		return ErrMediaNotFound
	}

	record, err := s.repo.FindByIDAndUserID(ctx, mediaID, userID)
	if err != nil {
		return fmt.Errorf("find media: %w", err)
	}
	if record == nil || record.DeletedAt.Valid || record.Status == models.MediaStatusDeleted || record.Status == models.MediaStatusDeletionPending {
		return ErrMediaNotFound
	}

	// Check purpose alignment with media category
	switch purpose {
	case PurposeVoiceCloning, PurposeVoiceDataProcessing:
		if record.Category != models.MediaCategoryVoiceSample {
			return ErrIncompatiblePurpose
		}
	case PurposeFaceSynthesis, PurposeFaceLikenessProcessing:
		if record.Category != models.MediaCategoryFaceSample {
			return ErrIncompatiblePurpose
		}
	}

	if s.consents == nil {
		return errors.New("consent service not configured")
	}

	// Fail-closed purpose-specific verification
	return s.consents.CheckConsent(ctx, ConsentCheckRequest{
		UserID:          userID,
		ConsentType:     purpose,
		RequiredVersion: CurrentPolicyVersion,
	})
}
