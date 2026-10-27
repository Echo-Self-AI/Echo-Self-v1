package services

import (
	"context"
	"errors"
	"fmt"
	"net/mail"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

var (
	ErrInvalidRegistration = errors.New("invalid registration details")
	ErrEmailAlreadyExists  = errors.New("email already registered")
	ErrInvalidCredentials  = errors.New("invalid credentials")
)

type AuthService struct {
	users    *repositories.UserRepository
	tokens   *TokenService
	sessions *repositories.RefreshSessionRepository
}

func NewAuthService(
	users *repositories.UserRepository,
	tokens *TokenService,
	sessions *repositories.RefreshSessionRepository,
) *AuthService {
	return &AuthService{
		users:    users,
		tokens:   tokens,
		sessions: sessions,
	}
}

func (s *AuthService) Register(
	ctx context.Context,
	input RegisterInput,
) (*RegisterResult, error) {
	email := strings.ToLower(strings.TrimSpace(input.Email))
	name := strings.TrimSpace(input.DisplayName)

	if len(email) == 0 || len(email) > 255 {
		return nil, ErrInvalidRegistration
	}

	parsed, err := mail.ParseAddress(email)
	if err != nil || parsed.Address != email {
		return nil, ErrInvalidRegistration
	}

	if utf8.RuneCountInString(input.Password) < 8 || len(input.Password) > 72 {
		return nil, ErrInvalidRegistration
	}

	if utf8.RuneCountInString(name) == 0 || utf8.RuneCountInString(name) > 100 {
		return nil, ErrInvalidRegistration
	}

	existing, err := s.users.FindByEmail(ctx, email)
	if err != nil {
		return nil, fmt.Errorf("find existing user: %w", err)
	}
	if existing != nil {
		return nil, ErrEmailAlreadyExists
	}

	hash, err := bcrypt.GenerateFromPassword(
		[]byte(input.Password),
		bcrypt.DefaultCost,
	)
	if err != nil {
		return nil, fmt.Errorf("hash password: %w", err)
	}

	user := &models.User{
		Email:        email,
		PasswordHash: string(hash),
		DisplayName:  name,
	}

	if err := s.users.Create(ctx, user); err != nil {
		return nil, fmt.Errorf("create user: %w", err)
	}

	return &RegisterResult{
		ID:          user.ID.String(),
		Email:       user.Email,
		DisplayName: user.DisplayName,
	}, nil
}

func (s *AuthService) Login(
	ctx context.Context,
	input LoginInput,
) (*LoginResult, error) {
	email := strings.ToLower(strings.TrimSpace(input.Email))

	if email == "" || input.Password == "" || len(input.Password) > 72 {
		return nil, ErrInvalidCredentials
	}

	user, err := s.users.FindByEmail(ctx, email)
	if err != nil {
		return nil, fmt.Errorf("find user: %w", err)
	}

	if user == nil {
		return nil, ErrInvalidCredentials
	}

	if err := bcrypt.CompareHashAndPassword(
		[]byte(user.PasswordHash),
		[]byte(input.Password),
	); err != nil {
		return nil, ErrInvalidCredentials
	}

	token, err := s.tokens.GenerateAccessToken(user.ID.String())
	if err != nil {
		return nil, fmt.Errorf("generate access token: %w", err)
	}

	refreshToken, err := s.createRefreshSession(ctx, user.ID)
	if err != nil {
		return nil, err
	}

	return &LoginResult{
		AccessToken:  token,
		RefreshToken: refreshToken,
		TokenType:    "Bearer",
		ExpiresIn:    900,
	}, nil
}

func (s *AuthService) createRefreshSession(
	ctx context.Context,
	userID uuid.UUID,
) (string, error) {
	rawToken, err := generateRefreshToken()
	if err != nil {
		return "", fmt.Errorf("generate refresh token: %w", err)
	}

	session := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    userID,
		TokenHash: hashRefreshToken(rawToken),
		ExpiresAt: time.Now().Add(30 * 24 * time.Hour),
	}

	if err := s.sessions.Create(ctx, session); err != nil {
		return "", fmt.Errorf("create refresh session: %w", err)
	}

	return rawToken, nil
}

func (s *AuthService) Refresh(
	ctx context.Context,
	rawToken string,
) (*LoginResult, error) {
	if rawToken == "" || len(rawToken) != 64 {
		return nil, ErrInvalidCredentials
	}

	session, err := s.findRefreshSession(ctx, hashRefreshToken(rawToken))
	if err != nil {
		return nil, err
	}

	if session == nil || session.RevokedAt != nil ||
		!session.ExpiresAt.After(time.Now()) {
		return nil, ErrInvalidCredentials
	}

	// Generate both tokens before consuming the current refresh token.
	accessToken, err := s.tokens.GenerateAccessToken(session.UserID.String())
	if err != nil {
		return nil, fmt.Errorf("generate access token: %w", err)
	}

	newRawToken, err := generateRefreshToken()
	if err != nil {
		return nil, fmt.Errorf("generate refresh token: %w", err)
	}

	now := time.Now()
	newSession := &models.RefreshSession{
		ID:        uuid.New(),
		UserID:    session.UserID,
		TokenHash: hashRefreshToken(newRawToken),
		ExpiresAt: now.Add(30 * 24 * time.Hour),
	}

	if err := s.sessions.Rotate(
		ctx,
		hashRefreshToken(rawToken),
		newSession,
		now,
	); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) || errors.Is(err, repositories.ErrTokenReuseDetected) {
			return nil, ErrInvalidCredentials
		}

		return nil, fmt.Errorf("rotate refresh session: %w", err)
	}

	return &LoginResult{
		AccessToken:  accessToken,
		RefreshToken: newRawToken,
		TokenType:    "Bearer",
		ExpiresIn:    900,
	}, nil
}

func (s *AuthService) Logout(
	ctx context.Context,
	rawToken string,
) error {
	if rawToken == "" || len(rawToken) != 64 {
		return nil
	}

	return s.sessions.Revoke(
		ctx,
		hashRefreshToken(rawToken),
		time.Now(),
	)
}

func (s *AuthService) findRefreshSession(
	ctx context.Context,
	tokenHash string,
) (*models.RefreshSession, error) {
	session, err := s.sessions.FindByTokenHash(ctx, tokenHash)
	if err != nil {
		return nil, fmt.Errorf("find refresh session: %w", err)
	}

	return session, nil
}
