package config

import (
	"errors"
	"testing"
)

func validConfig() *Config {
	return &Config{
		AppEnv:      "test",
		AppPort:     "8080",
		DatabaseURL: "postgres://example:example@localhost:5432/echoself",
		JWTSecret:   "test-secret",
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
