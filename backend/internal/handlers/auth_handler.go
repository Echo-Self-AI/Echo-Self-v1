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
	passwordReset     *services.PasswordResetService
	isDevelopment     bool
}

func NewAuthHandler(
	auth *services.AuthService,
	emailVerification *services.EmailVerificationService,
	passwordReset *services.PasswordResetService,
	isDevelopment bool,
) *AuthHandler {
	return &AuthHandler{
		auth:              auth,
		emailVerification: emailVerification,
		passwordReset:     passwordReset,
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

type passwordResetRequest struct {
	Email string `json:"email" binding:"required,email"`
}

type passwordResetConfirmRequest struct {
	Token       string `json:"token" binding:"required"`
	NewPassword string `json:"new_password" binding:"required"`
}

func (h *AuthHandler) Register(c *gin.Context) {
	var req registerRequest

	if !bindJSON(c, &req) {
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
			respondError(c, http.StatusBadRequest,
				"INVALID_REGISTRATION", "Invalid registration details")
		case errors.Is(err, services.ErrEmailAlreadyExists):
			respondError(c, http.StatusConflict,
				"EMAIL_ALREADY_EXISTS", "Email already registered")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not register user")
		}
		return
	}

	c.JSON(http.StatusCreated, result)
}

func (h *AuthHandler) Login(c *gin.Context) {
	var req loginRequest

	if !bindJSON(c, &req) {
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
			respondError(c, http.StatusUnauthorized,
				"INVALID_CREDENTIALS", "Invalid email or password")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Login failed")
		}
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *AuthHandler) Refresh(c *gin.Context) {
	var req refreshRequest

	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	result, err := h.auth.Refresh(ctx, req.RefreshToken)
	if err != nil {
		if errors.Is(err, services.ErrInvalidCredentials) {
			respondError(c, http.StatusUnauthorized,
				"INVALID_REFRESH_TOKEN", "Invalid or expired refresh token")
			return
		}

		respondInternalError(c, "Could not refresh session")
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *AuthHandler) Logout(c *gin.Context) {
	var req logoutRequest

	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	if err := h.auth.Logout(ctx, req.RefreshToken); err != nil {
		respondInternalError(c, "Could not logout")
		return
	}

	// Return the same response even if the token was already revoked
	// or did not exist
	c.Status(http.StatusNoContent)
}

func (h *AuthHandler) RequestEmailVerification(c *gin.Context) {
	userID, err := uuid.Parse(c.GetString("userID"))
	if err != nil {
		respondError(c, http.StatusUnauthorized, "UNAUTHORIZED", "Unauthorized")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	token, err := h.emailVerification.CreateToken(ctx, userID)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrEmailAlreadyVerified):
			respondError(c, http.StatusConflict,
				"EMAIL_ALREADY_VERIFIED", "Email already verified")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not create verification token")
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

	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	err := h.emailVerification.Verify(ctx, req.Token)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidVerificationToken):
			respondError(c, http.StatusBadRequest,
				"INVALID_VERIFICATION_TOKEN", "Invalid or expired verification token")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not verify email")
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "email verified successfully",
	})
}

func (h *AuthHandler) RequestPasswordReset(c *gin.Context) {
	var req passwordResetRequest

	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	token, err := h.passwordReset.CreateToken(ctx, req.Email)
	if err != nil {
		respondInternalError(c, "Could not process password reset request")
		return
	}

	response := gin.H{
		"message": "if the account exists, password reset instructions will be sent",
	}

	if h.isDevelopment && token != "" {
		response["reset_token"] = token
	}

	c.JSON(http.StatusOK, response)
}

func (h *AuthHandler) ConfirmPasswordReset(c *gin.Context) {
	var req passwordResetConfirmRequest

	if !bindJSON(c, &req) {
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	err := h.passwordReset.ResetPassword(ctx, req.Token, req.NewPassword)
	if err != nil {
		switch {
		case errors.Is(err, services.ErrInvalidPasswordResetToken):
			respondError(c, http.StatusBadRequest,
				"INVALID_PASSWORD_RESET_TOKEN", "Invalid or expired reset token")
		case errors.Is(err, services.ErrInvalidNewPassword):
			respondError(c, http.StatusBadRequest,
				"INVALID_NEW_PASSWORD", "Password must be between 8 and 72 bytes")
		case errors.Is(err, context.DeadlineExceeded),
			errors.Is(err, context.Canceled):
			respondRequestTimeout(c)
		default:
			respondInternalError(c, "Could not reset password")
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "password reset successfully; please log in again",
	})
}
