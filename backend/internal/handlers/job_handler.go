package handlers

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

type JobHandler struct {
	jobService *services.JobService
}

func NewJobHandler(jobService *services.JobService) *JobHandler {
	return &JobHandler{jobService: jobService}
}

func (h *JobHandler) getAuthenticatedUserID(c *gin.Context) (string, bool) {
	userID, exists := c.Get(middleware.UserIDKey)
	if !exists {
		respondError(
			c,
			http.StatusUnauthorized,
			"UNAUTHENTICATED",
			"Authentication is required",
		)
		return "", false
	}

	id, ok := userID.(string)
	if !ok || id == "" {
		respondError(
			c,
			http.StatusUnauthorized,
			"INVALID_AUTH_CONTEXT",
			"Invalid authentication context",
		)
		return "", false
	}

	return id, true
}

type submitJobRequest struct {
	JobType string         `json:"job_type" binding:"required"`
	MediaID *string        `json:"media_id,omitempty"`
	Payload map[string]any `json:"payload,omitempty"`
}

func (h *JobHandler) Submit(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	var req submitJobRequest
	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.jobService.SubmitJob(ctx, userID, services.SubmitJobInput{
		JobType: req.JobType,
		MediaID: req.MediaID,
		Payload: req.Payload,
	})
	if err != nil {
		switch {
		case errors.Is(err, services.ErrUnsupportedJobType):
			respondError(c, http.StatusBadRequest, "UNSUPPORTED_JOB_TYPE", err.Error())
		case errors.Is(err, services.ErrMediaNotFound):
			respondError(c, http.StatusBadRequest, "MEDIA_NOT_FOUND", "Specified input media was not found")
		case errors.Is(err, services.ErrConsentNotGranted), errors.Is(err, services.ErrConsentRevokedOrExpired):
			respondError(c, http.StatusForbidden, "CONSENT_REQUIRED", "Purpose-specific consent must be granted before submitting this job")
		case errors.Is(err, services.ErrUnauthorized):
			respondError(c, http.StatusUnauthorized, "UNAUTHORIZED", "Unauthorized")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not submit background job")
		}
		return
	}

	c.JSON(http.StatusCreated, result)
}

func (h *JobHandler) List(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	status := c.Query("status")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

	if limit <= 0 || limit > 100 {
		limit = 20
	}
	if offset < 0 {
		offset = 0
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	results, err := h.jobService.ListJobs(ctx, userID, status, limit, offset)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			respondRequestTimeout(c)
			return
		}
		respondInternalError(c, "Could not list jobs")
		return
	}

	c.JSON(http.StatusOK, results)
}

func (h *JobHandler) Get(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	jobID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.jobService.GetJob(ctx, userID, jobID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrJobNotFound):
			respondError(c, http.StatusNotFound, "JOB_NOT_FOUND", "Job not found")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not retrieve job")
		}
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *JobHandler) Cancel(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	jobID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.jobService.CancelJob(ctx, userID, jobID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrJobNotFound):
			respondError(c, http.StatusNotFound, "JOB_NOT_FOUND", "Job not found")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not cancel job")
		}
		return
	}

	c.JSON(http.StatusOK, result)
}
