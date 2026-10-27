package config

import (
	"errors"
	"testing"
)

func validConfig() *Config {
	return &Config{
		AppEnv:                   "test",
		AppPort:                  "8080",
		DatabaseURL:              "postgres://example:example@localhost:5432/echoself",
		JWTSecret:                "test-secret",
		MediaMaxUploadSizeBytes:  10 * 1024 * 1024,
		MediaUserQuotaBytes:      50 * 1024 * 1024,
		MediaUserMaxObjects:      20,
		MediaAllowedTypes:        []string{"audio/mpeg", "image/png"},
		MediaUploadRatePerMinute: 10,
		MediaUploadRateBurst:     5,
	}
}

func TestValidateAcceptsValidConfig(t *testing.T) {
	cfg := validConfig()

	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config, got error: %v", err)
	}
}

func TestValidateRequiresDatabaseURL(t *testing.T) {
	cfg := validConfig()
	cfg.DatabaseURL = ""

	if err := cfg.Validate(); !errors.Is(err, ErrMissingDatabaseURL) {
		t.Fatalf("expected ErrMissingDatabaseURL, got %v", err)
	}
}

func TestValidateRequiresJWTSecret(t *testing.T) {
	cfg := validConfig()
	cfg.JWTSecret = ""

	if err := cfg.Validate(); !errors.Is(err, ErrMissingJWTSecret) {
		t.Fatalf("expected ErrMissingJWTSecret, got %v", err)
	}
}

func TestValidateRejectsInvalidEnvironment(t *testing.T) {
	cfg := validConfig()
	cfg.AppEnv = "staging"

	if err := cfg.Validate(); err == nil {
		t.Fatal("expected invalid APP_ENV to be rejected")
	}
}

func TestValidateRejectsInvalidPort(t *testing.T) {
	tests := []string{"", "abc", "0", "-1", "65536"}

	for _, port := range tests {
		t.Run(port, func(t *testing.T) {
			cfg := validConfig()
			cfg.AppPort = port

			if err := cfg.Validate(); err == nil {
				t.Fatalf("expected invalid port %q to be rejected", port)
			}
		})
	}
}

func TestValidateR2Configuration(t *testing.T) {
	cfg := validConfig()
	cfg.R2Enabled = true

	// Missing bucket name
	if err := cfg.Validate(); !errors.Is(err, ErrMissingR2Bucket) {
		t.Fatalf("expected ErrMissingR2Bucket, got %v", err)
	}

	cfg.R2BucketName = "echoself-media"

	// Missing credentials
	if err := cfg.Validate(); !errors.Is(err, ErrMissingR2Credentials) {
		t.Fatalf("expected ErrMissingR2Credentials, got %v", err)
	}

	cfg.R2AccessKeyID = "test-access-key"
	cfg.R2SecretAccessKey = "test-secret-key"

	// Missing endpoint or account id
	if err := cfg.Validate(); !errors.Is(err, ErrMissingR2EndpointOrAcc) {
		t.Fatalf("expected ErrMissingR2EndpointOrAcc, got %v", err)
	}

	// Valid with account id
	cfg.R2AccountID = "test-account-id"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config with R2AccountID, got %v", err)
	}

	// Valid with endpoint instead of account id
	cfg.R2AccountID = ""
	cfg.R2Endpoint = "https://custom-r2.endpoint.com"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config with R2Endpoint, got %v", err)
	}
}

func TestValidateMediaQuotaConfiguration(t *testing.T) {
	// Upload size <= 0
	cfg := validConfig()
	cfg.MediaMaxUploadSizeBytes = 0
	if err := cfg.Validate(); !errors.Is(err, ErrInvalidQuotaConfig) {
		t.Fatalf("expected ErrInvalidQuotaConfig, got %v", err)
	}

	// User quota < upload size
	cfg = validConfig()
	cfg.MediaUserQuotaBytes = cfg.MediaMaxUploadSizeBytes - 1
	if err := cfg.Validate(); !errors.Is(err, ErrInvalidQuotaConfig) {
		t.Fatalf("expected ErrInvalidQuotaConfig, got %v", err)
	}

	// Allowed types empty
	cfg = validConfig()
	cfg.MediaAllowedTypes = nil
	if err := cfg.Validate(); !errors.Is(err, ErrInvalidQuotaConfig) {
		t.Fatalf("expected ErrInvalidQuotaConfig, got %v", err)
	}
}
