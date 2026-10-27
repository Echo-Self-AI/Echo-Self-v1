package config

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	ErrMissingDatabaseURL     = errors.New("DATABASE_URL is required")
	ErrMissingJWTSecret       = errors.New("JWT_SECRET is required")
	ErrWeakJWTSecret          = errors.New("JWT_SECRET must be at least 32 characters in production")
	ErrMissingR2Bucket        = errors.New("R2_BUCKET_NAME is required when cloud storage is enabled")
	ErrMissingR2Credentials   = errors.New("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required when cloud storage is enabled")
	ErrMissingR2EndpointOrAcc = errors.New("R2_ENDPOINT or R2_ACCOUNT_ID is required when cloud storage is enabled")
	ErrInvalidQuotaConfig     = errors.New("media quota configuration is invalid")
	ErrMissingRedisURL        = errors.New("REDIS_URL is required when worker is enabled")
	ErrInvalidWorkerConfig    = errors.New("worker configuration is invalid")
	ErrInvalidSMTPConfig      = errors.New("SMTP configuration is invalid")
	ErrInvalidFrontendURL     = errors.New("FRONTEND_BASE_URL must be a valid URL")
)

func (c *Config) Validate() error {
	if strings.TrimSpace(c.DatabaseURL) == "" {
		return ErrMissingDatabaseURL
	}

	if strings.TrimSpace(c.JWTSecret) == "" {
		return ErrMissingJWTSecret
	}

	switch c.AppEnv {
	case "development", "test", "production":
	default:
		return fmt.Errorf("APP_ENV must be development, test, or production")
	}

	if c.AppEnv == "production" && len(c.JWTSecret) < 32 {
		return ErrWeakJWTSecret
	}

	port, err := strconv.Atoi(c.AppPort)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("APP_PORT must be a valid port between 1 and 65535")
	}

	if c.WorkerHealthPort != "" {
		wPort, err := strconv.Atoi(c.WorkerHealthPort)
		if err != nil || wPort < 1 || wPort > 65535 {
			return fmt.Errorf("WORKER_HEALTH_PORT must be a valid port between 1 and 65535")
		}
	}

	if strings.TrimSpace(c.FrontendBaseURL) != "" {
		if !strings.HasPrefix(c.FrontendBaseURL, "http://") && !strings.HasPrefix(c.FrontendBaseURL, "https://") {
			return ErrInvalidFrontendURL
		}
	}

	if c.R2Enabled {
		if strings.TrimSpace(c.R2BucketName) == "" {
			return ErrMissingR2Bucket
		}
		if strings.TrimSpace(c.R2AccessKeyID) == "" || strings.TrimSpace(c.R2SecretAccessKey) == "" {
			return ErrMissingR2Credentials
		}
		if strings.TrimSpace(c.R2Endpoint) == "" && strings.TrimSpace(c.R2AccountID) == "" {
			return ErrMissingR2EndpointOrAcc
		}
	}

	if c.MediaMaxUploadSizeBytes <= 0 || c.MediaUserQuotaBytes <= 0 || c.MediaUserMaxObjects <= 0 {
		return ErrInvalidQuotaConfig
	}

	if c.MediaUserQuotaBytes < c.MediaMaxUploadSizeBytes {
		return fmt.Errorf("%w: MEDIA_USER_QUOTA_BYTES must be greater than or equal to MEDIA_MAX_UPLOAD_SIZE_BYTES", ErrInvalidQuotaConfig)
	}

	if len(c.MediaAllowedTypes) == 0 {
		return fmt.Errorf("%w: at least one media type must be allowed", ErrInvalidQuotaConfig)
	}

	if c.WorkerEnabled {
		if strings.TrimSpace(c.RedisURL) == "" {
			return ErrMissingRedisURL
		}
		if c.WorkerConcurrency <= 0 || c.JobMaxAttempts <= 0 || c.JobLeaseDurationSeconds <= 0 || c.JobDrainTimeoutSeconds <= 0 {
			return ErrInvalidWorkerConfig
		}
	}

	if c.SMTPEnabled {
		if strings.TrimSpace(c.SMTPHost) == "" {
			return fmt.Errorf("%w: SMTP_HOST is required when SMTP is enabled", ErrInvalidSMTPConfig)
		}
		if c.SMTPPort <= 0 || c.SMTPPort > 65535 {
			return fmt.Errorf("%w: SMTP_PORT must be between 1 and 65535", ErrInvalidSMTPConfig)
		}
		if strings.TrimSpace(c.SMTPFrom) == "" {
			return fmt.Errorf("%w: SMTP_FROM is required when SMTP is enabled", ErrInvalidSMTPConfig)
		}
	}

	return nil
}
