package services

import (
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

const testJWTSecret = "test-secret-that-is-at-least-32-bytes-long"

func TestGenerateAndParseAccessToken(t *testing.T) {
	service, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	token, err := service.GenerateAccessToken("user-123")
	if err != nil {
		t.Fatalf("generate token: %v", err)
	}

	claims, err := service.ParseAccessToken(token)
	if err != nil {
		t.Fatalf("parse token: %v", err)
	}

	if claims.Subject != "user-123" {
		t.Fatalf("expected subject user-123, got %q", claims.Subject)
	}

	if claims.ExpiresAt == nil {
		t.Fatal("expected token expiration")
	}

	if time.Until(claims.ExpiresAt.Time) <= 0 {
		t.Fatal("expected token to expire in future")
	}
}

func TestParseAccessTokenRejectsInvalidToken(t *testing.T) {
	service, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatalf("create token service: %v", err)
	}

	_, err = service.ParseAccessToken("not-a-valid-token")
	if err == nil {
		t.Fatal("expected invalid token to be rejected")
	}
}

func TestParseAccessTokenRejectsWrongSecret(t *testing.T) {
	issuer, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatal(err)
	}

	other, err := NewTokenService("another-secret-that-is-at-least-32-bytes")
	if err != nil {
		t.Fatal(err)
	}

	token, err := issuer.GenerateAccessToken("user-123")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := other.ParseAccessToken(token); err == nil {
		t.Fatal("expected token signed with another secret to be rejected")
	}
}

func TestParseAccessTokenRejectsExpiredToken(t *testing.T) {
	service, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatal(err)
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, AccessClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "user-123",
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(-time.Minute)),
			IssuedAt:  jwt.NewNumericDate(time.Now().Add(-2 * time.Minute)),
		},
	})

	signed, err := token.SignedString([]byte(testJWTSecret))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := service.ParseAccessToken(signed); err == nil {
		t.Fatal("expected expired token to be rejected")
	}
}

func TestNewTokenServiceRejectsShortSecret(t *testing.T) {
	_, err := NewTokenService("too-short")
	if err == nil {
		t.Fatal("expected short secret to be rejected")
	}
}

func TestParseAccessTokenRejectsMissingExpiration(t *testing.T) {
	service, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatal(err)
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, AccessClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:  "user-123",
			IssuedAt: jwt.NewNumericDate(time.Now().Add(-time.Minute)),
		},
	})

	signed, err := token.SignedString([]byte(testJWTSecret))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := service.ParseAccessToken(signed); err == nil {
		t.Fatal("expected token without expiration to be rejected")
	}
}

func TestParseAccessTokenRejectsFutureIssuedAt(t *testing.T) {
	service, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatal(err)
	}

	now := time.Now()

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, AccessClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "user-123",
			IssuedAt:  jwt.NewNumericDate(now.Add(10 * time.Minute)),
			ExpiresAt: jwt.NewNumericDate(now.Add(20 * time.Minute)),
		},
	})

	signed, err := token.SignedString([]byte(testJWTSecret))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := service.ParseAccessToken(signed); err == nil {
		t.Fatal("expected token issued in the future to be rejected")
	}
}

func TestParseAccessTokenRejectsNoneAlgorithm(t *testing.T) {
	service, err := NewTokenService(testJWTSecret)
	if err != nil {
		t.Fatal(err)
	}

	token := jwt.NewWithClaims(jwt.SigningMethodNone, AccessClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "user-123",
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(15 * time.Minute)),
		},
	})

	signed, err := token.SignedString(jwt.UnsafeAllowNoneSignatureType)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := service.ParseAccessToken(signed); err == nil {
		t.Fatal("expected unsigned token to be rejected")
	}
}
