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

type UserHandler struct {
	userService *services.UserService
}

func NewUserHandler(userService *services.UserService) *UserHandler {
	return &UserHandler{userService: userService}
}

type updateProfileRequest struct {
	DisplayName string `json:"display_name" binding:"required"`
}

func (h *UserHandler) getAuthenticatedUserID(c *gin.Context) (string, bool) {
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

func (h *UserHandler) Me(c *gin.Context) {
	id, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	profile, err := h.userService.GetProfile(ctx, id)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrUserNotFound):
			respondError(
				c,
				http.StatusUnauthorized,
				"USER_NOT_FOUND",
				"User no longer exists",
			)
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not retrieve user profile")
		}
		return
	}

	c.JSON(http.StatusOK, profile)
}

func (h *UserHandler) UpdateProfile(c *gin.Context) {
	id, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	var req updateProfileRequest
	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	profile, err := h.userService.UpdateProfile(ctx, id, services.UpdateProfileInput{
		DisplayName: req.DisplayName,
	})
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidProfile):
			respondError(
				c,
				http.StatusBadRequest,
				"INVALID_PROFILE",
				"Display name must be between 1 and 100 characters",
			)
		case errors.Is(err, services.ErrUserNotFound):
			respondError(
				c,
				http.StatusUnauthorized,
				"USER_NOT_FOUND",
				"User no longer exists",
			)
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not update user profile")
		}
		return
	}

	c.JSON(http.StatusOK, profile)
}

func (h *UserHandler) DeleteAccount(c *gin.Context) {
	id, ok := h.getAuthenticatedUserID(c)
	if !ok {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	if err := h.userService.DeleteAccount(ctx, id); err != nil {
		switch {
		case errors.Is(err, services.ErrUserNotFound):
			respondError(
				c,
				http.StatusUnauthorized,
				"USER_NOT_FOUND",
				"User no longer exists",
			)
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not delete account")
		}
		return
	}

	c.Status(http.StatusNoContent)
}
