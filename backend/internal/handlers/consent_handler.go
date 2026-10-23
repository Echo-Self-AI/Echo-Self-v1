package handlers

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

type ConsentHandler struct {
	consents *services.ConsentService
}

func NewConsentHandler(consents *services.ConsentService) *ConsentHandler {
	return &ConsentHandler{consents: consents}
}

type grantConsentRequest struct {
	ConsentType   string     `json:"consent_type" binding:"required"`
	PolicyVersion string     `json:"policy_version" binding:"required"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
}

func (h *ConsentHandler) getAuthenticatedUserID(c *gin.Context) (string, bool) {
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

func (h *ConsentHandler) getRequestMetadata(c *gin.Context) services.RequestMetadata {
	return services.RequestMetadata{
		IPAddress: c.ClientIP(),
		UserAgent: c.Request.UserAgent(),
	}
}

func (h *ConsentHandler) Grant(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	var req grantConsentRequest
	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.consents.GrantConsent(ctx, userID, services.GrantConsentInput{
		ConsentType:   req.ConsentType,
		PolicyVersion: req.PolicyVersion,
		ExpiresAt:     req.ExpiresAt,
	}, h.getRequestMetadata(c))

	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidConsentPurpose):
			respondError(c, http.StatusBadRequest,
				"INVALID_CONSENT_PURPOSE", "Invalid or unsupported consent purpose")
		case errors.Is(err, services.ErrInvalidPolicyVersion):
			respondError(c, http.StatusBadRequest,
				"INVALID_POLICY_VERSION", "Unsupported consent policy version")
		case errors.Is(err, services.ErrConsentAlreadyActive):
			respondError(c, http.StatusConflict,
				"CONSENT_ALREADY_ACTIVE", "Active consent record already exists for this purpose")
		case errors.Is(err, services.ErrUnauthorized):
			respondError(c, http.StatusUnauthorized,
				"UNAUTHORIZED", "Unauthorized")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not grant consent")
		}
		return
	}

	c.JSON(http.StatusCreated, result)
}

func (h *ConsentHandler) List(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	results, err := h.consents.ListConsents(ctx, userID)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			respondRequestTimeout(c)
			return
		}
		respondInternalError(c, "Could not list consent records")
		return
	}

	c.JSON(http.StatusOK, results)
}

func (h *ConsentHandler) Get(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	consentID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.consents.GetConsent(ctx, userID, consentID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrConsentNotFound):
			respondError(c, http.StatusNotFound,
				"CONSENT_NOT_FOUND", "Consent record not found")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not retrieve consent record")
		}
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *ConsentHandler) Revoke(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	consentID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	if err := h.consents.RevokeConsent(ctx, userID, consentID, h.getRequestMetadata(c)); err != nil {
		switch {
		case errors.Is(err, services.ErrConsentNotFound):
			respondError(c, http.StatusNotFound,
				"CONSENT_NOT_FOUND", "Consent record not found")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not revoke consent")
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "consent revoked successfully",
	})
}

func (h *ConsentHandler) AuditLogs(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	consentID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	logs, err := h.consents.GetConsentAuditLogs(ctx, userID, consentID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrConsentNotFound):
			respondError(c, http.StatusNotFound,
				"CONSENT_NOT_FOUND", "Consent record not found")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not retrieve consent audit logs")
		}
		return
	}

	c.JSON(http.StatusOK, logs)
}
