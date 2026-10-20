package handlers

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type AuthHandler struct {
	auth              *services.AuthService
	emailVerification *services.EmailVerificationService
	isDevelopment     bool
}

func NewAuthHandler(
	auth *services.AuthService,
	emailVerification *services.EmailVerificationService,
	isDevelopment bool,
) *AuthHandler {
	return &AuthHandler{
		auth:              auth,
		emailVerification: emailVerification,
		isDevelopment:     isDevelopment,
	}
}

type registerRequest struct {
	Email       string `json:"email" binding:"required"`
	Password    string `json:"password" binding:"required"`
	DisplayName string `json:"display_name" binding:"required"`
}

type loginRequest struct {
	Email    string `json:"email" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type refreshRequest struct {
	RefreshToken string `json:"refresh_token" binding:"required"`
}

type logoutRequest struct {
	RefreshToken string `json:"refresh_token" binding:"required"`
}

type confirmEmailVerificationRequest struct {
	Token string `json:"token" binding:"required"`
}

func (h *AuthHandler) Register(c *gin.Context) {
	var req registerRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.auth.Register(ctx, services.RegisterInput{
		Email:       req.Email,
		Password:    req.Password,
		DisplayName: req.DisplayName,
	})
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidRegistration):
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "invalid registration details",
			})
		case errors.Is(err, services.ErrEmailAlreadyExists):
			c.JSON(http.StatusConflict, gin.H{
				"error": "email already registered",
			})
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			c.JSON(http.StatusRequestTimeout, gin.H{
				"error": "request timed out",
			})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "could not register user",
			})
		}
		return
	}

	c.JSON(http.StatusCreated, result)
}

func (h *AuthHandler) Login(c *gin.Context) {
	var req loginRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.auth.Login(ctx, services.LoginInput{
		Email:    req.Email,
		Password: req.Password,
	})
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidCredentials):
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "invalid email or password",
			})
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			c.JSON(http.StatusRequestTimeout, gin.H{
				"error": "request timed out",
			})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "login failed",
			})
		}
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *AuthHandler) Refresh(c *gin.Context) {
	var req refreshRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "refresh_token is required",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.auth.Refresh(ctx, req.RefreshToken)
	if err != nil {
		if errors.Is(err, services.ErrInvalidCredentials) {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "invalid or expired refresh token",
			})
			return
		}

		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "could not refresh session",
		})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *AuthHandler) Logout(c *gin.Context) {
	var req logoutRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "refresh_token is required",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	if err := h.auth.Logout(ctx, req.RefreshToken); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "could not logout",
		})
		return
	}

	// Return the same response even if the token was already revoked
	// or did not exist
	c.Status(http.StatusNoContent)
}

func (h *AuthHandler) RequestEmailVerification(c *gin.Context) {
	userID, err := uuid.Parse(c.GetString("userID"))
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	token, err := h.emailVerification.CreateToken(ctx, userID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrEmailAlreadyVerified):
			c.JSON(http.StatusConflict, gin.H{
				"error": "email already verified",
			})
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			c.JSON(http.StatusRequestTimeout, gin.H{
				"error": "request timed out",
			})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "could not create verification token",
			})
		}
		return
	}

	response := gin.H{
		"message": "verification token created",
	}

	if h.isDevelopment {
		response["verification_token"] = token
	}

	c.JSON(http.StatusCreated, response)
}

func (h *AuthHandler) ConfirmEmailVerification(c *gin.Context) {
	var req confirmEmailVerificationRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	err := h.emailVerification.Verify(ctx, req.Token)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidVerificationToken):
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "invalid or expired verification token",
			})
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			c.JSON(http.StatusRequestTimeout, gin.H{
				"error": "request timed out",
			})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "could not verify email",
			})
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "email verified successfully",
	})
}
