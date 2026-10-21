package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/config"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/handlers"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

func main() {
	cfg := config.Load()
	if cfg.AppEnv == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	db, err := database.Connect(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("database initialization failed: %v", err)
	}

	if err := database.Migrate(db); err != nil {
		log.Fatalf("database migration failed: %v", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		log.Fatalf("database handle failed: %v", err)
	}
	defer sqlDB.Close()

	tokenService, err := services.NewTokenService(cfg.JWTSecret)
	if err != nil {
		log.Fatalf("JWT configuration failed: %v", err)
	}

	userRepository := repositories.NewUserRepository(db)
	refreshSessionRepository := repositories.NewRefreshSessionRepository(db)
	emailVerificationRepository := repositories.NewEmailVerificationRepository(db)
	passwordResetRepository := repositories.NewPasswordResetRepository(db)

	authService := services.NewAuthService(
		userRepository,
		tokenService,
		refreshSessionRepository,
	)

	emailVerificationService :=
		services.NewEmailVerificationService(
			userRepository,
			emailVerificationRepository,
		)

	passwordResetService := services.NewPasswordResetService(
		userRepository,
		passwordResetRepository,
	)

	authHandler := handlers.NewAuthHandler(
		authService,
		emailVerificationService,
		passwordResetService,
		cfg.AppEnv == "development",
	)

	userHandler := handlers.NewUserHandler(userRepository)

	router := gin.Default()
	router.Use(middleware.RateLimit())

	authGroup := router.Group("/api/v1/auth")
	{
		authGroup.POST(
			"/register",
			middleware.RateLimitWithConfig(1.0/60.0, 3),
			authHandler.Register,
		)

		authGroup.POST(
			"/login",
			middleware.RateLimitWithConfig(1.0/60.0, 5),
			authHandler.Login,
		)

		authGroup.POST(
			"/refresh",
			middleware.RateLimitWithConfig(1.0, 10),
			authHandler.Refresh,
		)

		authGroup.POST(
			"/logout",
			authHandler.Logout,
		)

		authGroup.POST(
			"/email-verification",
			middleware.RateLimitWithConfig(1.0/60.0, 3),
			middleware.AuthRequired(tokenService),
			authHandler.RequestEmailVerification,
		)

		authGroup.POST(
			"/email-verification/confirm",
			middleware.RateLimitWithConfig(1.0/60.0, 5),
			authHandler.ConfirmEmailVerification,
		)

		authGroup.POST(
			"/password-reset/request",
			middleware.RateLimitWithConfig(1.0/60.0, 3),
			authHandler.RequestPasswordReset,
		)

		authGroup.POST(
			"/password-reset/confirm",
			middleware.RateLimitWithConfig(1.0/60.0, 5),
			authHandler.ConfirmPasswordReset,
		)
	}

	router.GET("/api/v1/users/me", middleware.AuthRequired(tokenService), userHandler.Me)

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"service": "echoself-api",
			"env":     cfg.AppEnv,
		})
	})

	router.GET("/api/v1/health/db", func(c *gin.Context) {
		sqlDB, err := db.DB()
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "unavailable",
			})
			return
		}

		ctx, cancel := context.WithTimeout(
			c.Request.Context(),
			3*time.Second,
		)
		defer cancel()

		if err := sqlDB.PingContext(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "unavailable",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":   "ok",
			"database": "connected",
		})
	})

	server := &http.Server{
		Addr:              ":" + cfg.AppPort,
		Handler:           router,
		ReadHeaderTimeout: 5 * time.Second,
	}

	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server failed: %v", err)
	}
}
