package config

import (
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	AppEnv      string
	AppPort     string
	DatabaseURL string
	JWTSecret   string

	// Cloudflare R2 Configuration
	R2Enabled         bool
	R2AccountID       string
	R2AccessKeyID     string
	R2SecretAccessKey string
	R2BucketName      string
	R2Endpoint        string
	R2Region          string

	// Media Storage & Free-Tier Cost Controls
	MediaMaxUploadSizeBytes  int64
	MediaUserQuotaBytes      int64
	MediaUserMaxObjects      int
	MediaAllowedTypes        []string
	MediaUploadRatePerMinute int
	MediaUploadRateBurst     int

	// Redis & Background Worker Configuration
	RedisURL                string
	RedisPoolSize           int
	WorkerEnabled           bool
	WorkerConcurrency       int
	JobMaxAttempts          int
	JobLeaseDurationSeconds int
	JobDrainTimeoutSeconds  int
	WorkerHealthPort        string

	// Deployment & Migration
	AutoMigrate     bool
	FrontendBaseURL string

	// Email Delivery (SMTP)
	SMTPEnabled  bool
	SMTPHost     string
	SMTPPort     int
	SMTPUsername string
	SMTPPassword string
	SMTPFrom     string
	SMTPUseTLS   bool
}

func Load() *Config {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found; using environment variables")
	}

	allowedTypesStr := getEnv("MEDIA_ALLOWED_TYPES", "audio/mpeg,audio/wav,audio/ogg,audio/webm,image/jpeg,image/png,image/webp")
	var allowedTypes []string
	for _, t := range strings.Split(allowedTypesStr, ",") {
		trimmed := strings.TrimSpace(t)
		if trimmed != "" {
			allowedTypes = append(allowedTypes, trimmed)
		}
	}

	appEnv := getEnv("APP_ENV", "development")
	defaultAutoMigrate := true
	if appEnv == "production" {
		defaultAutoMigrate = false
	}

	cfg := &Config{
		AppEnv:      appEnv,
		AppPort:     getEnv("APP_PORT", "8080"),
		DatabaseURL: os.Getenv("DATABASE_URL"),
		JWTSecret:   os.Getenv("JWT_SECRET"),

		// Cloudflare R2
		R2Enabled:         getEnvBool("R2_ENABLED", false),
		R2AccountID:       os.Getenv("R2_ACCOUNT_ID"),
		R2AccessKeyID:     os.Getenv("R2_ACCESS_KEY_ID"),
		R2SecretAccessKey: os.Getenv("R2_SECRET_ACCESS_KEY"),
		R2BucketName:      os.Getenv("R2_BUCKET_NAME"),
		R2Endpoint:        os.Getenv("R2_ENDPOINT"),
		R2Region:          getEnv("R2_REGION", "auto"),

		// Cost Controls
		MediaMaxUploadSizeBytes:  getEnvInt64("MEDIA_MAX_UPLOAD_SIZE_BYTES", 10*1024*1024), // 10 MiB
		MediaUserQuotaBytes:      getEnvInt64("MEDIA_USER_QUOTA_BYTES", 50*1024*1024),      // 50 MiB
		MediaUserMaxObjects:      getEnvInt("MEDIA_USER_MAX_OBJECTS", 20),
		MediaAllowedTypes:        allowedTypes,
		MediaUploadRatePerMinute: getEnvInt("MEDIA_UPLOAD_RATE_PER_MINUTE", 10),
		MediaUploadRateBurst:     getEnvInt("MEDIA_UPLOAD_RATE_BURST", 5),

		// Redis & Background Jobs
		RedisURL:                getEnv("REDIS_URL", "redis://localhost:6379/0"),
		RedisPoolSize:           getEnvInt("REDIS_POOL_SIZE", 10),
		WorkerEnabled:           getEnvBool("WORKER_ENABLED", false),
		WorkerConcurrency:       getEnvInt("WORKER_CONCURRENCY", 2),
		JobMaxAttempts:          getEnvInt("JOB_MAX_ATTEMPTS", 3),
		JobLeaseDurationSeconds: getEnvInt("JOB_LEASE_DURATION_SECONDS", 30),
		JobDrainTimeoutSeconds:  getEnvInt("JOB_DRAIN_TIMEOUT_SECONDS", 10),
		WorkerHealthPort:        getEnv("WORKER_HEALTH_PORT", "8081"),

		// Deployment & Migration
		AutoMigrate:     getEnvBool("AUTO_MIGRATE", defaultAutoMigrate),
		FrontendBaseURL: getEnv("FRONTEND_BASE_URL", "http://localhost:3000"),

		// Email Delivery
		SMTPEnabled:  getEnvBool("SMTP_ENABLED", false),
		SMTPHost:     os.Getenv("SMTP_HOST"),
		SMTPPort:     getEnvInt("SMTP_PORT", 587),
		SMTPUsername: os.Getenv("SMTP_USERNAME"),
		SMTPPassword: os.Getenv("SMTP_PASSWORD"),
		SMTPFrom:     getEnv("SMTP_FROM", "no-reply@echoself.ai"),
		SMTPUseTLS:   getEnvBool("SMTP_USE_TLS", true),
	}

	return cfg
}

func getEnv(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	return value
}

func getEnvInt64(key string, fallback int64) int64 {
	valStr := os.Getenv(key)
	if valStr == "" {
		return fallback
	}

	val, err := strconv.ParseInt(valStr, 10, 64)
	if err != nil || val <= 0 {
		return fallback
	}

	return val
}

func getEnvInt(key string, fallback int) int {
	valStr := os.Getenv(key)
	if valStr == "" {
		return fallback
	}

	val, err := strconv.Atoi(valStr)
	if err != nil || val <= 0 {
		return fallback
	}

	return val
}

func getEnvBool(key string, fallback bool) bool {
	valStr := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	if valStr == "" {
		return fallback
	}

	return valStr == "true" || valStr == "1" || valStr == "yes"
}
