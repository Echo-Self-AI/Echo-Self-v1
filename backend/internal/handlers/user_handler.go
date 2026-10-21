package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/gin-gonic/gin"
)

type UserHandler struct {
	users *repositories.UserRepository
}

func NewUserHandler(users *repositories.UserRepository) *UserHandler {
	return &UserHandler{users: users}
}

func (h *UserHandler) Me(c *gin.Context) {
	userID, exists := c.Get(middleware.UserIDKey)
	if !exists {
		respondError(
			c,
			http.StatusUnauthorized,
			"UNAUTHENTICATED",
			"Authentication is required",
		)
		return
	}

	id, ok := userID.(string)
	if !ok {
		respondError(
			c,
			http.StatusUnauthorized,
			"INVALID_AUTH_CONTEXT",
			"Invalid authentication context",
		)
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	user, err := h.users.FindByID(ctx, id)
	if err != nil {
		respondInternalError(c, "Could not retrieve user")
		return
	}

	if user == nil {
		respondError(
			c,
			http.StatusUnauthorized,
			"USER_NOT_FOUND",
			"User no longer exists",
		)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":           user.ID,
		"email":        user.Email,
		"display_name": user.DisplayName,
		"created_at":   user.CreatedAt,
	})
}
