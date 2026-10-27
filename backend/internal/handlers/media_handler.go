package handlers

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

type MediaHandler struct {
	mediaService *services.MediaService
}

func NewMediaHandler(mediaService *services.MediaService) *MediaHandler {
	return &MediaHandler{mediaService: mediaService}
}

func (h *MediaHandler) getAuthenticatedUserID(c *gin.Context) (string, bool) {
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

func (h *MediaHandler) Upload(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	fileHeader, err := c.FormFile("file")
	if err != nil {
		respondError(
			c,
			http.StatusBadRequest,
			"MISSING_FILE",
			"A multipart 'file' field is required",
		)
		return
	}

	category := c.PostForm("category")
	if category == "" {
		category = "general"
	}

	file, err := fileHeader.Open()
	if err != nil {
		respondError(
			c,
			http.StatusBadRequest,
			"INVALID_FILE",
			"Could not open uploaded file",
		)
		return
	}
	defer file.Close()

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	result, err := h.mediaService.UploadMedia(ctx, userID, services.UploadMediaInput{
		Category:     category,
		Filename:     fileHeader.Filename,
		DeclaredSize: fileHeader.Size,
		Reader:       file,
	})
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidCategory):
			respondError(c, http.StatusBadRequest, "INVALID_CATEGORY", err.Error())
		case errors.Is(err, services.ErrUnsupportedMediaType):
			respondError(c, http.StatusUnsupportedMediaType, "UNSUPPORTED_MEDIA_TYPE", err.Error())
		case errors.Is(err, services.ErrUploadTooLarge):
			respondError(c, http.StatusRequestEntityTooLarge, "UPLOAD_TOO_LARGE", err.Error())
		case errors.Is(err, services.ErrQuotaExceeded):
			respondError(c, http.StatusForbidden, "STORAGE_QUOTA_EXCEEDED", err.Error())
		case errors.Is(err, services.ErrEmptyFile):
			respondError(c, http.StatusBadRequest, "EMPTY_FILE", err.Error())
		case errors.Is(err, services.ErrMalformedUpload):
			respondError(c, http.StatusBadRequest, "MALFORMED_UPLOAD", err.Error())
		case errors.Is(err, services.ErrUnauthorized):
			respondError(c, http.StatusUnauthorized, "UNAUTHORIZED", "Unauthorized")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, fmt.Sprintf("Upload failed: %v", err))
		}
		return
	}

	c.JSON(http.StatusCreated, result)
}

func (h *MediaHandler) List(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	category := c.Query("category")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	results, err := h.mediaService.ListMedia(ctx, userID, category)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			respondRequestTimeout(c)
			return
		}
		respondInternalError(c, "Could not list media")
		return
	}

	c.JSON(http.StatusOK, results)
}

func (h *MediaHandler) Get(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	mediaID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.mediaService.GetMedia(ctx, userID, mediaID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrMediaNotFound):
			respondError(c, http.StatusNotFound, "MEDIA_NOT_FOUND", "Media not found")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not retrieve media")
		}
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *MediaHandler) Content(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	mediaID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	stream, meta, err := h.mediaService.StreamMediaContent(ctx, userID, mediaID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrMediaNotFound):
			respondError(c, http.StatusNotFound, "MEDIA_NOT_FOUND", "Media not found")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not stream media")
		}
		return
	}
	defer stream.Close()

	c.Header("Content-Type", meta.ContentType)
	c.Header("Content-Length", strconv.FormatInt(meta.FileSize, 10))
	c.Header("Content-Disposition", fmt.Sprintf("inline; filename=%q", meta.OriginalFilename))
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("Cache-Control", "private, no-cache, no-store, must-revalidate")
	c.Status(http.StatusOK)

	_, _ = io.Copy(c.Writer, stream)
}

func (h *MediaHandler) Delete(c *gin.Context) {
	userID, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	mediaID := c.Param("id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	if err := h.mediaService.DeleteMedia(ctx, userID, mediaID); err != nil {
		switch {
		case errors.Is(err, services.ErrMediaNotFound):
			respondError(c, http.StatusNotFound, "MEDIA_NOT_FOUND", "Media not found")
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not delete media")
		}
		return
	}

	c.Status(http.StatusNoContent)
}
