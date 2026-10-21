package main

import (
	"context"
	"errors"
	"log"
	"log/slog"
	"net/http"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/config"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/handlers"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/server"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

func main() {
	cfg := config.Load()

	if err := cfg.Validate(); err != nil {
		log.Fatalf("configuration validation failed: %v", err)
	}

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

	router := gin.New()

	if err := router.SetTrustedProxies(nil); err != nil {
		log.Fatalf("trusted proxy configuration failed: %v", err)
	}

	logger := slog.New(slog.NewJSONHandler(log.Writer(), nil))

	router.Use(gin.Recovery())
	router.Use(middleware.RequestID())
	router.Use(middleware.RequestLogger(logger))
	router.Use(middleware.SecurityHeaders())
	router.Use(middleware.RequestSizeLimit(middleware.MaxRequestBodySize))
	router.Use(middleware.RateLimit())

	handlers.RegisterAuthRoutes(router, authHandler, tokenService)

	router.GET("/api/v1/users/me", middleware.AuthRequired(tokenService), userHandler.Me)

	healthHandler := handlers.NewHealthHandler(db)

	router.GET("/health", healthHandler.Liveness)
	router.GET("/ready", healthHandler.Readiness)

	// Keep the existing database health endpoint for compatibility.
	router.GET("/api/v1/health/db", healthHandler.Readiness)

	httpServer := &http.Server{
		Addr:              ":" + cfg.AppPort,
		Handler:           router,
		ReadHeaderTimeout: 5 * time.Second,
	}

	serveErr := make(chan error, 1)

	go func() {
		logger.Info("starting HTTP server", "address", httpServer.Addr)

		if err := httpServer.ListenAndServe(); err != nil &&
			!errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
			return
		}

		serveErr <- nil
	}()

	signals := server.NotifyShutdown()

	select {
	case sig := <-signals:
		logger.Info("shutdown signal received", "signal", sig.String())

	case err := <-serveErr:
		if err != nil {
			logger.Error("HTTP server failed", "error", err)
		}

		if closeErr := sqlDB.Close(); closeErr != nil {
			logger.Error("database connection pool close failed", "error", closeErr)
		}

		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx, httpServer); err != nil {
		logger.Error("graceful shutdown failed", "error", err)
	} else {
		logger.Info("HTTP server stopped gracefully")
	}

	if err := sqlDB.Close(); err != nil {
		logger.Error("database connection pool close failed", "error", err)
	} else {
		logger.Info("database connection pool closed")
	}
}
