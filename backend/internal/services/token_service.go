package services

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

var ErrInvalidTokenConfig = errors.New("JWT secret is not configured")

type TokenService struct {
	secret []byte
}

type AccessClaims struct {
	jwt.RegisteredClaims
}

func NewTokenService(secret string) (*TokenService, error) {
	if len(secret) < 32 {
		return nil, ErrInvalidTokenConfig
	}

	return &TokenService{
		secret: []byte(secret),
	}, nil
}

func (s *TokenService) GenerateAccessToken(userID string) (string, error) {
	now := time.Now()

	claims := AccessClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   userID,
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(15 * time.Minute)),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)

	return token.SignedString(s.secret)
}

func (s *TokenService) ParseAccessToken(tokenString string) (*AccessClaims, error) {
	claims := &AccessClaims{}

	token, err := jwt.ParseWithClaims(
		tokenString,
		claims,
		func(token *jwt.Token) (any, error) {
			if token.Method != jwt.SigningMethodHS256 {
				return nil, errors.New("unexpected signing method")
			}

			return s.secret, nil
		},
		jwt.WithValidMethods([]string{"HS256"}),
		jwt.WithExpirationRequired(),
		jwt.WithIssuedAt(),
	)
	if err != nil || !token.Valid {
		return nil, errors.New("invalid access token")
	}

	return claims, nil
}
