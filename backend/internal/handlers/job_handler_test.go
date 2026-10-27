package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"golang.org/x/time/rate"
)

type handlerMockJobRepo struct {
	jobs   map[uuid.UUID]*models.Job
	outbox map[uuid.UUID]*models.JobOutbox
}

func newHandlerMockJobRepo() *handlerMockJobRepo {
	return &handlerMockJobRepo{
		jobs:   make(map[uuid.UUID]*models.Job),
		outbox: make(map[uuid.UUID]*models.JobOutbox),
	}
}

func (m *handlerMockJobRepo) CreateWithOutbox(ctx context.Context, job *models.Job, outbox *models.JobOutbox) error {
	m.jobs[job.ID] = job
	m.outbox[outbox.ID] = outbox
	return nil
}

func (m *handlerMockJobRepo) FindByIDAndUserID(ctx context.Context, id, userID uuid.UUID) (*models.Job, error) {
	j, ok := m.jobs[id]
	if !ok || j.UserID != userID {
		return nil, nil
	}
	return j, nil
}

func (m *handlerMockJobRepo) FindByID(ctx context.Context, id uuid.UUID) (*models.Job, error) {
	j, ok := m.jobs[id]
	if !ok {
		return nil, nil
	}
	return j, nil
}

func (m *handlerMockJobRepo) ListByUserID(ctx context.Context, userID uuid.UUID, status string, limit, offset int) ([]models.Job, error) {
	var list []models.Job
	for _, j := range m.jobs {
		if j.UserID == userID && (status == "" || j.Status == status) {
			list = append(list, *j)
		}
	}
	return list, nil
}

func (m *handlerMockJobRepo) RequestCancellation(ctx context.Context, id, userID uuid.UUID, now time.Time) (*models.Job, error) {
	j, ok := m.jobs[id]
	if !ok || j.UserID != userID {
		return nil, services.ErrJobNotFound
	}
	if !models.IsTerminalStatus(j.Status) {
		j.Status = models.JobStatusCancelled
		j.CancelRequestedAt = &now
		j.CompletedAt = &now
	}
	return j, nil
}

func setupJobTestRouter(t *testing.T) (*gin.Engine, *services.TokenService, *handlerMockJobRepo) {
	t.Helper()
	gin.SetMode(gin.TestMode)

	tokenService, err := services.NewTokenService("32-char-secret-for-jwt-testing-ok!")
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	mockRepo := newHandlerMockJobRepo()
	proc := processor.NewMockProcessor()
	jobService := services.NewJobService(mockRepo, nil, proc, nil)
	jobHandler := NewJobHandler(jobService)

	router := gin.New()
	RegisterJobRoutes(router, jobHandler, tokenService, JobRouteConfig{
		SubmitRatePerSecond: rate.Limit(100),
		SubmitBurst:         50,
	})

	return router, tokenService, mockRepo
}

func TestJobRoutes_Unauthenticated(t *testing.T) {
	router, _, _ := setupJobTestRouter(t)

	endpoints := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/v1/jobs"},
		{http.MethodGet, "/api/v1/jobs"},
		{http.MethodGet, "/api/v1/jobs/" + uuid.NewString()},
		{http.MethodPost, "/api/v1/jobs/" + uuid.NewString() + "/cancel"},
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

func TestJobRoutes_LifecycleAndSecurity(t *testing.T) {
	router, tokenService, _ := setupJobTestRouter(t)

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

	// 1. Submit invalid payload: unsupported job type
	{
		bodyBytes, _ := json.Marshal(map[string]any{
			"job_type": "invalid_model_job",
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/jobs", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for unsupported job type, got %d", rec.Code)
		}
	}

	// 2. Submit valid job for User A
	var jobID string
	{
		bodyBytes, _ := json.Marshal(map[string]any{
			"job_type": processor.JobTypeMockTranscription,
			"payload": map[string]any{
				"language": "en-US",
			},
		})
		req := httptest.NewRequest(http.MethodPost, "/api/v1/jobs", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusCreated {
			t.Fatalf("expected 201 for valid submit, got %d; body: %s", rec.Code, rec.Body.String())
		}

		var res services.JobResult
		if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
			t.Fatalf("unmarshal response: %v", err)
		}
		if res.Status != models.JobStatusQueued {
			t.Errorf("expected status 'queued', got %q", res.Status)
		}
		if res.UserID != userA.String() {
			t.Errorf("expected user id %s, got %s", userA.String(), res.UserID)
		}
		jobID = res.ID
	}

	// 3. User A retrieves job
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/jobs/"+jobID, nil)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rec.Code)
		}
	}

	// 4. Cross-user isolation: User B tries to retrieve User A's job -> 404
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/jobs/"+jobID, nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected 404 for cross-user get, got %d", rec.Code)
		}
	}

	// 5. Cross-user isolation: User B lists jobs -> empty
	{
		req := httptest.NewRequest(http.MethodGet, "/api/v1/jobs", nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rec.Code)
		}
		var list []services.JobResult
		_ = json.Unmarshal(rec.Body.Bytes(), &list)
		if len(list) != 0 {
			t.Fatalf("expected 0 jobs for User B, got %d", len(list))
		}
	}

	// 6. Cross-user isolation: User B tries to cancel User A's job -> 404
	{
		req := httptest.NewRequest(http.MethodPost, "/api/v1/jobs/"+jobID+"/cancel", nil)
		req.Header.Set("Authorization", "Bearer "+tokenB)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected 404 for cross-user cancel, got %d", rec.Code)
		}
	}

	// 7. User A cancels their job -> 200 OK
	{
		req := httptest.NewRequest(http.MethodPost, "/api/v1/jobs/"+jobID+"/cancel", nil)
		req.Header.Set("Authorization", "Bearer "+tokenA)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200 for cancel, got %d", rec.Code)
		}

		var res services.JobResult
		_ = json.Unmarshal(rec.Body.Bytes(), &res)
		if res.Status != models.JobStatusCancelled {
			t.Errorf("expected status 'cancelled', got %q", res.Status)
		}
	}
}
