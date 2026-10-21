package handlers

import (
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

func RegisterAuthRoutes(
	router *gin.Engine,
	authHandler *AuthHandler,
	tokenService *services.TokenService,
) {
	registerAuthRoutesWithLimits(
		router,
		authHandler,
		tokenService,
		defaultAuthRateLimits(),
	)
}

func registerAuthRoutesWithLimits(
	router *gin.Engine,
	authHandler *AuthHandler,
	tokenService *services.TokenService,
	limits authRateLimits,
) {
	authGroup := router.Group("/api/v1/auth")

	authGroup.POST(
		"/register",
		middleware.RateLimitWithConfig(limits.register.requestsPerSecond, limits.register.burst),
		authHandler.Register,
	)

	authGroup.POST(
		"/login",
		middleware.RateLimitWithConfig(limits.login.requestsPerSecond, limits.login.burst),
		authHandler.Login,
	)

	authGroup.POST(
		"/refresh",
		middleware.RateLimitWithConfig(limits.refresh.requestsPerSecond, limits.refresh.burst),
		authHandler.Refresh,
	)

	authGroup.POST("/logout", authHandler.Logout)

	authGroup.POST(
		"/email-verification",
		middleware.RateLimitWithConfig(limits.emailVerification.requestsPerSecond, limits.emailVerification.burst),
		middleware.AuthRequired(tokenService),
		authHandler.RequestEmailVerification,
	)

	authGroup.POST(
		"/email-verification/confirm",
		middleware.RateLimitWithConfig(limits.emailVerificationConfirm.requestsPerSecond, limits.emailVerificationConfirm.burst),
		authHandler.ConfirmEmailVerification,
	)

	authGroup.POST(
		"/password-reset/request",
		middleware.RateLimitWithConfig(limits.passwordResetRequest.requestsPerSecond, limits.passwordResetRequest.burst),
		authHandler.RequestPasswordReset,
	)

	authGroup.POST(
		"/password-reset/confirm",
		middleware.RateLimitWithConfig(limits.passwordResetConfirm.requestsPerSecond, limits.passwordResetConfirm.burst),
		authHandler.ConfirmPasswordReset,
	)
}
