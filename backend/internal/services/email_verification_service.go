package services

import (
	"context"
	"errors"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
)

var (
	ErrInvalidVerificationToken = errors.New("invalid or expired verification token")
	ErrEmailAlreadyVerified     = errors.New("email is already verified")
)

const emailVerificationTokenTTL = 24 * time.Hour

type EmailVerificationService struct {
	users         *repositories.UserRepository
	verifications *repositories.EmailVerificationRepository
}

func NewEmailVerificationService(
	users *repositories.UserRepository,
	verifications *repositories.EmailVerificationRepository,
) *EmailVerificationService {
	return &EmailVerificationService{
		users:         users,
		verifications: verifications,
	}
}

func (s *EmailVerificationService) CreateToken(
	ctx context.Context,
	userID uuid.UUID,
) (string, error) {
	user, err := s.users.FindByID(ctx, userID.String())
	if err != nil {
		return "", err
	}
	if user == nil {
		return "", errors.New("user not found")
	}
	if user.IsEmailVerified {
		return "", ErrEmailAlreadyVerified
	}

	rawToken, err := generateEmailVerificationToken()
	if err != nil {
		return "", err
	}

	verification := &models.EmailVerification{
		ID:        uuid.New(),
		UserID:    userID,
		TokenHash: hashEmailVerificationToken(rawToken),
		ExpiresAt: time.Now().Add(emailVerificationTokenTTL),
	}

	if err := s.verifications.Create(ctx, verification); err != nil {
		return "", err
	}

	return rawToken, nil
}

func (s *EmailVerificationService) Verify(
	ctx context.Context,
	rawToken string,
) error {
	if len(rawToken) != 64 {
		return ErrInvalidVerificationToken
	}

	err := s.verifications.VerifyAndConsume(
		ctx,
		hashEmailVerificationToken(rawToken),
		time.Now(),
	)

	switch {
	case errors.Is(err, repositories.ErrInvalidVerificationRecord):
		return ErrInvalidVerificationToken
	case errors.Is(err, repositories.ErrAlreadyVerified):
		return ErrEmailAlreadyVerified
	default:
		return err
	}
}