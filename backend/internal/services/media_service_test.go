package services

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
	"github.com/google/uuid"
)

// Minimal valid PNG (1x1 transparent)
var samplePNG = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
	0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
	0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
	0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
	0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
}

// Minimal valid WAV header + empty data
var sampleWAV = []byte{
	'R', 'I', 'F', 'F', 0x24, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E',
	'f', 'm', 't', ' ', 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
	0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
	'd', 'a', 't', 'a', 0x00, 0x00, 0x00, 0x00,
}

type mockMediaRepo struct {
	records      map[uuid.UUID]*models.Media
	storageBytes int64
	objectCount  int64
	failCreate   error
}

func newMockMediaRepo() *mockMediaRepo {
	return &mockMediaRepo{records: make(map[uuid.UUID]*models.Media)}
}

func (m *mockMediaRepo) Create(ctx context.Context, media *models.Media) error {
	if m.failCreate != nil {
		return m.failCreate
	}
	m.records[media.ID] = media
	m.storageBytes += media.FileSize
	m.objectCount++
	return nil
}

func (m *mockMediaRepo) FindByIDAndUserID(ctx context.Context, id uuid.UUID, userID uuid.UUID) (*models.Media, error) {
	r, ok := m.records[id]
	if !ok || r.UserID != userID {
		return nil, nil
	}
	return r, nil
}

func (m *mockMediaRepo) ListByUserID(ctx context.Context, userID uuid.UUID, category string) ([]models.Media, error) {
	var list []models.Media
	for _, r := range m.records {
		if r.UserID == userID && (category == "" || r.Category == category) {
			list = append(list, *r)
		}
	}
	return list, nil
}

func (m *mockMediaRepo) GetUserStorageUsage(ctx context.Context, userID uuid.UUID) (int64, int64, error) {
	return m.storageBytes, m.objectCount, nil
}

func (m *mockMediaRepo) SoftDelete(ctx context.Context, id uuid.UUID, userID uuid.UUID, now time.Time) error {
	r, ok := m.records[id]
	if !ok || r.UserID != userID {
		return ErrMediaNotFound
	}
	r.Status = models.MediaStatusDeletionPending
	return nil
}

func (m *mockMediaRepo) MarkStatus(ctx context.Context, id uuid.UUID, userID uuid.UUID, status string) error {
	r, ok := m.records[id]
	if !ok || r.UserID != userID {
		return ErrMediaNotFound
	}
	r.Status = status
	return nil
}

func (m *mockMediaRepo) HardDelete(ctx context.Context, id uuid.UUID) error {
	delete(m.records, id)
	return nil
}

func (m *mockMediaRepo) ListPendingDeletions(ctx context.Context, limit int) ([]models.Media, error) {
	var list []models.Media
	for _, r := range m.records {
		if r.Status == models.MediaStatusDeletionPending {
			list = append(list, *r)
		}
	}
	return list, nil
}

func defaultTestMediaConfig() MediaConfig {
	return MediaConfig{
		MaxUploadSizeBytes: 10 * 1024 * 1024,
		UserQuotaBytes:     50 * 1024 * 1024,
		UserMaxObjects:     10,
		AllowedTypes: map[string]bool{
			"audio/wav":      true,
			"audio/x-wav":    true,
			"audio/vnd.wave": true,
			"audio/mpeg":     true,
			"image/png":      true,
			"image/jpeg":     true,
		},
	}
}

func TestSanitizeFilename(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"../../etc/passwd", "passwd"},
		{"test/path\\to/voice.wav", "voice.wav"},
		{"normal-file.png", "normal-file.png"},
		{"", "unnamed_media"},
		{"..", "unnamed_media"},
		{".", "unnamed_media"},
		{"/root/my audio sample.wav", "my audio sample.wav"},
	}

	for _, tt := range tests {
		got := sanitizeFilename(tt.input)
		if got != tt.expected {
			t.Errorf("sanitizeFilename(%q) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestMediaService_ValidationUnitTests(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStorage()
	mockRepo := newMockMediaRepo()
	service := NewMediaService(mockRepo, store, nil, defaultTestMediaConfig())

	userID := uuid.New().String()

	// 1. Invalid User ID
	_, err := service.UploadMedia(ctx, "invalid-uuid", UploadMediaInput{
		Category: "general",
		Reader:   bytes.NewReader(samplePNG),
	})
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("expected ErrUnauthorized, got %v", err)
	}

	// 2. Invalid Category
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     "invalid_cat",
		Filename:     "file.png",
		DeclaredSize: int64(len(samplePNG)),
		Reader:       bytes.NewReader(samplePNG),
	})
	if !errors.Is(err, ErrInvalidCategory) {
		t.Fatalf("expected ErrInvalidCategory, got %v", err)
	}

	// 3. Declared Size Exceeds Limit
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     "general",
		Filename:     "file.png",
		DeclaredSize: 20 * 1024 * 1024,
		Reader:       bytes.NewReader(samplePNG),
	})
	if !errors.Is(err, ErrUploadTooLarge) {
		t.Fatalf("expected ErrUploadTooLarge, got %v", err)
	}

	// 4. Empty File
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     "general",
		Filename:     "empty.png",
		DeclaredSize: 0,
		Reader:       bytes.NewReader([]byte{}),
	})
	if !errors.Is(err, ErrEmptyFile) {
		t.Fatalf("expected ErrEmptyFile, got %v", err)
	}

	// 5. Unsupported Media Type (text/plain)
	textData := []byte("plain text content which is not allowed")
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     "general",
		Filename:     "test.txt",
		DeclaredSize: int64(len(textData)),
		Reader:       bytes.NewReader(textData),
	})
	if !errors.Is(err, ErrUnsupportedMediaType) {
		t.Fatalf("expected ErrUnsupportedMediaType, got %v", err)
	}

	// 6. Mismatched Category & Content Type (image uploaded as voice_sample)
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     models.MediaCategoryVoiceSample,
		Filename:     "face.png",
		DeclaredSize: int64(len(samplePNG)),
		Reader:       bytes.NewReader(samplePNG),
	})
	if !errors.Is(err, ErrUnsupportedMediaType) {
		t.Fatalf("expected ErrUnsupportedMediaType for image in voice category, got %v", err)
	}

	// 7. Mismatched Category & Content Type (audio uploaded as face_sample)
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     models.MediaCategoryFaceSample,
		Filename:     "voice.wav",
		DeclaredSize: int64(len(sampleWAV)),
		Reader:       bytes.NewReader(sampleWAV),
	})
	if !errors.Is(err, ErrUnsupportedMediaType) {
		t.Fatalf("expected ErrUnsupportedMediaType for audio in face category, got %v", err)
	}
}

func TestMediaService_LifecycleHermetic(t *testing.T) {
	ctx := context.Background()
	userID := uuid.New().String()
	store := storage.NewMemoryStorage()
	mockRepo := newMockMediaRepo()
	service := NewMediaService(mockRepo, store, nil, defaultTestMediaConfig())

	// Upload voice sample
	res, err := service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     models.MediaCategoryVoiceSample,
		Filename:     "sample.wav",
		DeclaredSize: int64(len(sampleWAV)),
		Reader:       bytes.NewReader(sampleWAV),
	})
	if err != nil {
		t.Fatalf("upload voice sample: %v", err)
	}

	if res.OriginalFilename != "sample.wav" {
		t.Errorf("expected original filename 'sample.wav', got %q", res.OriginalFilename)
	}
	if res.Category != models.MediaCategoryVoiceSample {
		t.Errorf("expected category %s, got %s", models.MediaCategoryVoiceSample, res.Category)
	}

	// Retrieve metadata
	fetched, err := service.GetMedia(ctx, userID, res.ID)
	if err != nil {
		t.Fatalf("get media: %v", err)
	}
	if fetched.ID != res.ID {
		t.Errorf("expected ID %s, got %s", res.ID, fetched.ID)
	}

	// Stream content
	stream, meta, err := service.StreamMediaContent(ctx, userID, res.ID)
	if err != nil {
		t.Fatalf("stream media: %v", err)
	}
	defer stream.Close()

	if meta.FileSize != int64(len(sampleWAV)) {
		t.Errorf("expected size %d, got %d", len(sampleWAV), meta.FileSize)
	}

	// List media
	list, err := service.ListMedia(ctx, userID, "")
	if err != nil {
		t.Fatalf("list media: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("expected 1 item, got %d", len(list))
	}

	// Delete media
	if err := service.DeleteMedia(ctx, userID, res.ID); err != nil {
		t.Fatalf("delete media: %v", err)
	}

	// Confirm not found
	_, err = service.GetMedia(ctx, userID, res.ID)
	if !errors.Is(err, ErrMediaNotFound) {
		t.Fatalf("expected ErrMediaNotFound after delete, got %v", err)
	}
}

func TestMediaService_QuotaEnforcement(t *testing.T) {
	ctx := context.Background()
	userID := uuid.New().String()
	store := storage.NewMemoryStorage()
	mockRepo := newMockMediaRepo()

	// Strict config: max 2 objects, max 100 bytes quota
	strictCfg := MediaConfig{
		MaxUploadSizeBytes: 100,
		UserQuotaBytes:     100,
		UserMaxObjects:     2,
		AllowedTypes: map[string]bool{
			"image/png": true,
		},
	}
	service := NewMediaService(mockRepo, store, nil, strictCfg)

	// Upload 1: 67 bytes PNG
	_, err := service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     models.MediaCategoryFaceSample,
		Filename:     "img1.png",
		DeclaredSize: int64(len(samplePNG)),
		Reader:       bytes.NewReader(samplePNG),
	})
	if err != nil {
		t.Fatalf("first upload failed: %v", err)
	}

	// Upload 2: another 67 bytes PNG -> 67 + 67 = 134 > 100 bytes quota!
	_, err = service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     models.MediaCategoryFaceSample,
		Filename:     "img2.png",
		DeclaredSize: int64(len(samplePNG)),
		Reader:       bytes.NewReader(samplePNG),
	})
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Fatalf("expected ErrQuotaExceeded on byte quota limit, got %v", err)
	}
}

func TestMediaService_CompensatingStorageCleanupOnDBFailure(t *testing.T) {
	ctx := context.Background()
	userID := uuid.New().String()
	store := storage.NewMemoryStorage()
	mockRepo := newMockMediaRepo()
	mockRepo.failCreate = errors.New("simulated database transaction failure")

	service := NewMediaService(mockRepo, store, nil, defaultTestMediaConfig())

	_, err := service.UploadMedia(ctx, userID, UploadMediaInput{
		Category:     models.MediaCategoryFaceSample,
		Filename:     "face.png",
		DeclaredSize: int64(len(samplePNG)),
		Reader:       bytes.NewReader(samplePNG),
	})
	if err == nil {
		t.Fatal("expected DB failure error")
	}

	// Ensure no orphaned object left in storage
	if store.Count() > 0 {
		t.Fatalf("expected compensating cleanup to delete uploaded object from storage, found %d objects", store.Count())
	}
}

func TestMediaService_LifecycleWithDB(t *testing.T) {
	db := database.SetupTestDB(t)
	ctx := context.Background()

	user := &models.User{
		ID:           uuid.New(),
		Email:        "media-service-" + uuid.NewString() + "@example.com",
		PasswordHash: "test-hash",
		DisplayName:  "Media Tester",
	}
	if err := db.WithContext(ctx).Create(user).Error; err != nil {
		t.Fatalf("create user: %v", err)
	}

	mediaRepo := repositories.NewMediaRepository(db)
	consentRepo := repositories.NewConsentRepository(db)
	consentService := NewConsentService(consentRepo)
	store := storage.NewMemoryStorage()

	cfg := defaultTestMediaConfig()
	service := NewMediaService(mediaRepo, store, consentService, cfg)

	// 1. Upload valid voice sample
	res, err := service.UploadMedia(ctx, user.ID.String(), UploadMediaInput{
		Category:     models.MediaCategoryVoiceSample,
		Filename:     "../unsafe/my_voice.wav",
		DeclaredSize: int64(len(sampleWAV)),
		Reader:       bytes.NewReader(sampleWAV),
	})
	if err != nil {
		t.Fatalf("upload voice sample: %v", err)
	}

	if res.UserID != user.ID.String() {
		t.Errorf("expected user id %s, got %s", user.ID.String(), res.UserID)
	}
	if res.OriginalFilename != "my_voice.wav" {
		t.Errorf("expected sanitized filename 'my_voice.wav', got %q", res.OriginalFilename)
	}
	if res.Category != models.MediaCategoryVoiceSample {
		t.Errorf("expected category %s, got %s", models.MediaCategoryVoiceSample, res.Category)
	}

	// 2. Retrieve metadata
	fetched, err := service.GetMedia(ctx, user.ID.String(), res.ID)
	if err != nil {
		t.Fatalf("get media: %v", err)
	}
	if fetched.ID != res.ID {
		t.Errorf("expected id %s, got %s", res.ID, fetched.ID)
	}

	// 3. Stream content
	stream, meta, err := service.StreamMediaContent(ctx, user.ID.String(), res.ID)
	if err != nil {
		t.Fatalf("stream media: %v", err)
	}
	defer stream.Close()

	if meta.ID != res.ID {
		t.Errorf("expected stream meta ID %s, got %s", res.ID, meta.ID)
	}
	streamBuf := new(bytes.Buffer)
	if _, err := streamBuf.ReadFrom(stream); err != nil {
		t.Fatalf("read stream content: %v", err)
	}
	if !bytes.Equal(streamBuf.Bytes(), sampleWAV) {
		t.Fatal("streamed bytes do not match uploaded bytes")
	}

	// 4. List media
	list, err := service.ListMedia(ctx, user.ID.String(), "")
	if err != nil {
		t.Fatalf("list media: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("expected 1 media item, got %d", len(list))
	}

	// 5. Consent boundary test:
	// Storing voice sample does NOT grant voice cloning consent!
	err = service.VerifyProcessingConsent(ctx, user.ID.String(), res.ID, PurposeVoiceCloning)
	if !errors.Is(err, ErrConsentNotGranted) {
		t.Fatalf("expected ErrConsentNotGranted before consent granted, got %v", err)
	}

	// Grant purpose-specific consent
	_, err = consentService.GrantConsent(ctx, user.ID.String(), GrantConsentInput{
		ConsentType:   PurposeVoiceCloning,
		PolicyVersion: "v1.0",
	}, RequestMetadata{IPAddress: "127.0.0.1"})
	if err != nil {
		t.Fatalf("grant voice cloning consent: %v", err)
	}

	// Now processing consent should succeed
	if err := service.VerifyProcessingConsent(ctx, user.ID.String(), res.ID, PurposeVoiceCloning); err != nil {
		t.Fatalf("expected consent verification to succeed, got %v", err)
	}

	// Check purpose incompatibility: voice sample cannot be used for face synthesis!
	if err := service.VerifyProcessingConsent(ctx, user.ID.String(), res.ID, PurposeFaceSynthesis); !errors.Is(err, ErrIncompatiblePurpose) {
		t.Fatalf("expected ErrIncompatiblePurpose for face synthesis on voice sample, got %v", err)
	}

	// 6. Delete media
	if err := service.DeleteMedia(ctx, user.ID.String(), res.ID); err != nil {
		t.Fatalf("delete media: %v", err)
	}

	// Verify not found after delete
	_, err = service.GetMedia(ctx, user.ID.String(), res.ID)
	if !errors.Is(err, ErrMediaNotFound) {
		t.Fatalf("expected ErrMediaNotFound after delete, got %v", err)
	}
}
