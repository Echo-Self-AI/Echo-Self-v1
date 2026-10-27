package main

import (
	"context"
	"errors"
	"log"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/config"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/handlers"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/server"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
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

	userService := services.NewUserService(userRepository)
	userHandler := handlers.NewUserHandler(userService)

	consentRepository := repositories.NewConsentRepository(db)
	consentService := services.NewConsentService(consentRepository)
	consentHandler := handlers.NewConsentHandler(consentService)

	logger := slog.New(slog.NewJSONHandler(log.Writer(), nil))

	// Cloudflare R2 / Object Storage Setup
	var storageService storage.Storage
	if cfg.R2Enabled {
		logger.Info("initializing Cloudflare R2 storage", "bucket", cfg.R2BucketName, "region", cfg.R2Region)
		r2Store, err := storage.NewR2Storage(context.Background(), storage.R2Config{
			AccountID:       cfg.R2AccountID,
			AccessKeyID:     cfg.R2AccessKeyID,
			SecretAccessKey: cfg.R2SecretAccessKey,
			BucketName:      cfg.R2BucketName,
			Endpoint:        cfg.R2Endpoint,
			Region:          cfg.R2Region,
		})
		if err != nil {
			log.Fatalf("failed to initialize Cloudflare R2 storage: %v", err)
		}
		storageService = r2Store
	} else {
		logger.Info("cloud storage is not enabled; using in-memory fake storage")
		storageService = storage.NewMemoryStorage()
	}

	allowedTypesMap := make(map[string]bool)
	for _, t := range cfg.MediaAllowedTypes {
		allowedTypesMap[t] = true
	}

	mediaRepository := repositories.NewMediaRepository(db)
	mediaService := services.NewMediaService(mediaRepository, storageService, consentService, services.MediaConfig{
		MaxUploadSizeBytes: cfg.MediaMaxUploadSizeBytes,
		UserQuotaBytes:     cfg.MediaUserQuotaBytes,
		UserMaxObjects:     cfg.MediaUserMaxObjects,
		AllowedTypes:       allowedTypesMap,
	})
	mediaHandler := handlers.NewMediaHandler(mediaService)

	router := gin.New()

	if err := router.SetTrustedProxies(nil); err != nil {
		log.Fatalf("trusted proxy configuration failed: %v", err)
	}

	router.Use(gin.Recovery())
	router.Use(middleware.RequestID())
	router.Use(middleware.RequestLogger(logger))
	router.Use(middleware.SecurityHeaders())
	router.Use(func(c *gin.Context) {
		// Media endpoints enforce their own upload size limit
		if strings.HasPrefix(c.Request.URL.Path, "/api/v1/media") {
			c.Next()
			return
		}
		middleware.RequestSizeLimit(middleware.MaxRequestBodySize)(c)
	})
	router.Use(middleware.RateLimit())

	handlers.RegisterAuthRoutes(router, authHandler, tokenService)

	usersGroup := router.Group("/api/v1/users", middleware.AuthRequired(tokenService))
	{
		usersGroup.GET("/me", userHandler.Me)
		usersGroup.PATCH("/me", userHandler.UpdateProfile)
		usersGroup.DELETE("/me", userHandler.DeleteAccount)
	}

	consentsGroup := router.Group("/api/v1/consents", middleware.AuthRequired(tokenService))
	{
		consentsGroup.POST("", consentHandler.Grant)
		consentsGroup.GET("", consentHandler.List)
		consentsGroup.GET("/:id", consentHandler.Get)
		consentsGroup.POST("/:id/revoke", consentHandler.Revoke)
		consentsGroup.GET("/:id/audit", consentHandler.AuditLogs)
	}

	handlers.RegisterMediaRoutes(router, mediaHandler, tokenService, handlers.MediaRouteConfig{
		MaxUploadSizeBytes:  cfg.MediaMaxUploadSizeBytes,
		UploadRatePerMinute: cfg.MediaUploadRatePerMinute,
		UploadRateBurst:     cfg.MediaUploadRateBurst,
	})

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
