package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// Minimal valid PNG
var samplePNG = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
	0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
	0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
	0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
	0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
}

// Minimal valid WAV
var sampleWAV = []byte{
	'R', 'I', 'F', 'F', 0x24, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E',
	'f', 'm', 't', ' ', 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
	0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
	'd', 'a', 't', 'a', 0x00, 0x00, 0x00, 0x00,
}

type handlerMockRepo struct {
	records      map[uuid.UUID]*models.Media
	storageBytes int64
	objectCount  int64
}

func newHandlerMockRepo() *handlerMockRepo {
	return &handlerMockRepo{records: make(map[uuid.UUID]*models.Media)}
}

func (m *handlerMockRepo) Create(ctx context.Context, media *models.Media) error {
	m.records[media.ID] = media
	m.storageBytes += media.FileSize
	m.objectCount++
	return nil
}

func (m *handlerMockRepo) FindByIDAndUserID(ctx context.Context, id uuid.UUID, userID uuid.UUID) (*models.Media, error) {
	r, ok := m.records[id]
	if !ok || r.UserID != userID {
		return nil, nil
	}
	return r, nil
}

func (m *handlerMockRepo) ListByUserID(ctx context.Context, userID uuid.UUID, category string) ([]models.Media, error) {
	var list []models.Media
	for _, r := range m.records {
		if r.UserID == userID && (category == "" || r.Category == category) {
			list = append(list, *r)
		}
	}
	return list, nil
}

func (m *handlerMockRepo) GetUserStorageUsage(ctx context.Context, userID uuid.UUID) (int64, int64, error) {
	return m.storageBytes, m.objectCount, nil
}

func (m *handlerMockRepo) SoftDelete(ctx context.Context, id uuid.UUID, userID uuid.UUID, now time.Time) error {
	r, ok := m.records[id]
	if !ok || r.UserID != userID {
		return services.ErrMediaNotFound
	}
	r.Status = models.MediaStatusDeletionPending
	return nil
}

func (m *handlerMockRepo) MarkStatus(ctx context.Context, id uuid.UUID, userID uuid.UUID, status string) error {
	r, ok := m.records[id]
	if !ok || r.UserID != userID {
		return services.ErrMediaNotFound
	}
	r.Status = status
	return nil
}

func (m *handlerMockRepo) HardDelete(ctx context.Context, id uuid.UUID) error {
	delete(m.records, id)
	return nil
}

func (m *handlerMockRepo) ListPendingDeletions(ctx context.Context, limit int) ([]models.Media, error) {
	return nil, nil
}

func setupTestRouter(t *testing.T) (*gin.Engine, *services.TokenService, *handlerMockRepo, *storage.MemoryStorage) {
	t.Helper()
	gin.SetMode(gin.TestMode)

	tokenService, err := services.NewTokenService("32-char-secret-for-jwt-testing-ok!")
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	mockRepo := newHandlerMockRepo()
	store := storage.NewMemoryStorage()
	mediaService := services.NewMediaService(mockRepo, store, nil, services.MediaConfig{
		MaxUploadSizeBytes: 10 * 1024 * 1024,
		UserQuotaBytes:     50 * 1024 * 1024,
		UserMaxObjects:     20,
		AllowedTypes: map[string]bool{
			"audio/wav":      true,
			"audio/x-wav":    true,
			"audio/vnd.wave": true,
			"audio/mpeg":     true,
			"image/png":      true,
			"image/jpeg":     true,
		},
	})
	mediaHandler := NewMediaHandler(mediaService)

	router := gin.New()
	RegisterMediaRoutes(router, mediaHandler, tokenService, MediaRouteConfig{
		MaxUploadSizeBytes:  10 * 1024 * 1024,
		UploadRatePerMinute: 600,
		UploadRateBurst:     100,
	})

	return router, tokenService, mockRepo, store
}

func makeMultipartRequest(t *testing.T, fieldName, fileName string, fileBytes []byte, category string) (*bytes.Buffer, string) {
	t.Helper()
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	if fileName != "" && fileBytes != nil {
		part, err := writer.CreateFormFile(fieldName, fileName)
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		if _, err := part.Write(fileBytes); err != nil {
			t.Fatalf("write form bytes: %v", err)
		}
	}

	if category != "" {
		if err := writer.WriteField("category", category); err != nil {
			t.Fatalf("write form category: %v", err)
		}
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	return body, writer.FormDataContentType()
}

func TestMediaRoutes_Unauthenticated(t *testing.T) {
	router, _, _, _ := setupTestRouter(t)

	endpoints := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/v1/media"},
		{http.MethodGet, "/api/v1/media"},
		{http.MethodGet, "/api/v1/media/" + uuid.NewString()},
		{http.MethodGet, "/api/v1/media/" + uuid.NewString() + "/content"},
		{http.MethodDelete, "/api/v1/media/" + uuid.NewString()},
	}

	for _, ep := range endpoints {
		t.Run(ep.method+" "+ep.path, func(t *testing.T) {
			req := httptest.NewRequest(ep.method, ep.path, nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != http.StatusUnauthorized {
				t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
			}
		})
	}
}

func TestMediaRoutes_LifecycleAndSecurity(t *testing.T) {
	router, tokenService, _, _ := setupTestRouter(t)

	userA := uuid.New()
	userB := uuid.New()

	tokenA, err := tokenService.GenerateAccessToken(userA.String())
	if err != nil {
		t.Fatalf("generate token A: %v", err)
	}
	tokenB, err := tokenService.GenerateAccessToken(userB.String())
	if err != nil {
		t.Fatalf("generate token B: %v", err)
	}

	// 1. Missing file in multipart form
	{
		body, contentType := makeMultipartRequest(t, "", "", nil, "voice_sample")
		req := httptest.NewRequest(http.MethodPost, "/api/v1/media", body)
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected status 400 for missing file, got %d; body: %s", rec.Code, rec.Body.String())
		}
	}

	// 2. Mismatched content type (upload PNG as voice_sample)
	{
		body, contentType := makeMultipartRequest(t, "file", "fake_voice.wav", samplePNG, "voice_sample")
		req := httptest.NewRequest(http.MethodPost, "/api/v1/media", body)
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnsupportedMediaType {
			t.Fatalf("expected status 415 for mismatched MIME, got %d; body: %s", rec.Code, rec.Body.String())
		}
	}

	// 3. Valid Upload (User A uploads voice WAV)
	var mediaID string
	{
		body, contentType := makeMultipartRequest(t, "file", "my_voice.wav", sampleWAV, "voice_sample")
		req := httptest.NewRequest(http.MethodPost, "/api/v1/media", body)
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusCreated {
			t.Fatalf("expected status 201 for valid upload, got %d; body: %s", rec.Code, rec.Body.String())
		}

		var res services.MediaResult
		if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
			t.Fatalf("unmarshal response: %v", err)
		}
		if res.OriginalFilename != "my_voice.wav" {
			t.Errorf("expected original filename 'my_voice.wav', got %q", res.OriginalFilename)
		}
		if res.Category != "voice_sample" {
			t.Errorf("expected category 'voice_sample', got %q", res.Category)
		}
		mediaID = res.ID
	}

	// 4. List Media for User A
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/media", nil)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", rec.Code)
		}
		var list []services.MediaResult
		if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
			t.Fatalf("unmarshal list: %v", err)
		}
		if len(list) != 1 {
			t.Fatalf("expected 1 media item for User A, got %d", len(list))
		}
	}

	// 5. Cross-user isolation: User B lists media -> empty
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/media", nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", rec.Code)
		}
		var list []services.MediaResult
		if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
			t.Fatalf("unmarshal list: %v", err)
		}
		if len(list) != 0 {
			t.Fatalf("expected 0 media items for User B, got %d", len(list))
		}
	}

	// 6. Cross-user isolation: User B tries to GET User A's media -> 404
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/media/"+mediaID, nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected status 404 for cross-user get, got %d", rec.Code)
		}
	}

	// 7. Cross-user isolation: User B tries to stream User A's media content -> 404
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/media/"+mediaID+"/content", nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected status 404 for cross-user stream, got %d", rec.Code)
		}
	}

	// 8. User A streams media content -> 200 OK with correct headers
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/media/"+mediaID+"/content", nil)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", rec.Code)
		}

		if ct := rec.Header().Get("X-Content-Type-Options"); ct != "nosniff" {
			t.Errorf("expected X-Content-Type-Options 'nosniff', got %q", ct)
		}
		if cd := rec.Header().Get("Content-Disposition"); cd != `inline; filename="my_voice.wav"` {
			t.Errorf("expected Content-Disposition inline with filename, got %q", cd)
		}
		if !bytes.Equal(rec.Body.Bytes(), sampleWAV) {
			t.Fatal("streamed body does not match uploaded WAV content")
		}
	}

	// 9. Cross-user isolation: User B tries to DELETE User A's media -> 404
	{
		req := httptest.NewRequest(http.MethodDelete, "/api/v1/media/"+mediaID, nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected status 404 for cross-user delete, got %d", rec.Code)
		}
	}

	// 10. User A deletes media -> 204 No Content
	{
		req := httptest.NewRequest(http.MethodDelete, "/api/v1/media/"+mediaID, nil)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNoContent {
			t.Fatalf("expected status 204 for delete, got %d; body: %s", rec.Code, rec.Body.String())
		}
	}

	// 11. Subsequent GET by User A -> 404
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/media/"+mediaID, nil)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected status 404 after delete, got %d", rec.Code)
		}
	}
}
