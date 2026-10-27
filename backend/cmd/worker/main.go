package main

import (
	"context"
	"log"
	"log/slog"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/config"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/processor"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/queue"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/server"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/storage"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/worker"
)

func main() {
	cfg := config.Load()

	if err := cfg.Validate(); err != nil {
		log.Fatalf("configuration validation failed: %v", err)
	}

	logger := slog.New(slog.NewJSONHandler(log.Writer(), nil))
	logger.Info("starting EchoSelf background worker daemon")

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

	// Redis connection
	redisClient, err := queue.NewRedisClient(queue.RedisConfig{
		URL:      cfg.RedisURL,
		PoolSize: cfg.RedisPoolSize,
	})
	if err != nil {
		log.Fatalf("redis connection failed: %v", err)
	}
	defer redisClient.Close()

	// Storage setup (memory or R2)
	var storageService storage.Storage
	if cfg.R2Enabled {
		r2Store, err := storage.NewR2Storage(context.Background(), storage.R2Config{
			AccountID:       cfg.R2AccountID,
			AccessKeyID:     cfg.R2AccessKeyID,
			SecretAccessKey: cfg.R2SecretAccessKey,
			BucketName:      cfg.R2BucketName,
			Endpoint:        cfg.R2Endpoint,
			Region:          cfg.R2Region,
		})
		if err != nil {
			log.Fatalf("storage initialization failed: %v", err)
		}
		storageService = r2Store
	} else {
		storageService = storage.NewMemoryStorage()
	}

	// Services & Repositories
	consentRepo := repositories.NewConsentRepository(db)
	consentService := services.NewConsentService(consentRepo)

	allowedTypesMap := make(map[string]bool)
	for _, t := range cfg.MediaAllowedTypes {
		allowedTypesMap[t] = true
	}

	mediaRepo := repositories.NewMediaRepository(db)
	mediaService := services.NewMediaService(mediaRepo, storageService, consentService, services.MediaConfig{
		MaxUploadSizeBytes: cfg.MediaMaxUploadSizeBytes,
		UserQuotaBytes:     cfg.MediaUserQuotaBytes,
		UserMaxObjects:     cfg.MediaUserMaxObjects,
		AllowedTypes:       allowedTypesMap,
	})

	jobRepo := repositories.NewJobRepository(db)
	jobQueue := queue.NewRedisQueue(redisClient, "echoself:jobs:ready")
	mockProcessor := processor.NewMockProcessor()

	// Outbox Dispatcher
	dispatcherCtx, dispatcherCancel := context.WithCancel(context.Background())
	defer dispatcherCancel()

	dispatcher := queue.NewOutboxDispatcher(jobRepo, jobQueue, logger)
	go dispatcher.Start(dispatcherCtx, 500*time.Millisecond, 50)

	userRepo := repositories.NewUserRepository(db)

	// Worker Engine
	workerEngine := worker.NewWorker(
		jobRepo,
		jobQueue,
		mockProcessor,
		mediaService,
		worker.WorkerConfig{
			Concurrency:         cfg.WorkerConcurrency,
			LeaseDuration:       time.Duration(cfg.JobLeaseDurationSeconds) * time.Second,
			DrainTimeout:        time.Duration(cfg.JobDrainTimeoutSeconds) * time.Second,
			MinRetryBackoff:     2 * time.Second,
			MaxRetryBackoff:     60 * time.Second,
			StaleRecoveryPeriod: 10 * time.Second,
		},
		logger,
	)
	workerEngine.SetUserRepository(userRepo)

	workerCtx, workerCancel := context.WithCancel(context.Background())
	defer workerCancel()

	workerEngine.Start(workerCtx)

	// Handle graceful shutdown
	signals := server.NotifyShutdown()
	sig := <-signals
	logger.Info("shutdown signal received", "signal", sig.String())

	// Stop dispatcher and drain workers
	dispatcherCancel()
	workerCancel()
	workerEngine.Stop()

	if closeErr := sqlDB.Close(); closeErr != nil {
		logger.Error("database connection pool close failed", "error", closeErr)
	}

	logger.Info("background worker daemon stopped cleanly")
}
