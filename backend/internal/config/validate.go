package config

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	ErrMissingDatabaseURL = errors.New("DATABASE_URL is required")
	ErrMissingJWTSecret   = errors.New("JWT_SECRET is required")
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

	port, err := strconv.Atoi(c.AppPort)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("APP_PORT must be a valid port between 1 and 65535")
	}

	return nil
}
