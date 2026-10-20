package services

import (
	"context"
	"errors"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

var (
	ErrInvalidPasswordResetToken = errors.New("invalid or expired password reset token")
	ErrInvalidNewPassword        = errors.New("invalid new password")
)

const passwordResetTokenTTL = 30 * time.Minute

type PasswordResetService struct {
	users  *repositories.UserRepository
	resets *repositories.PasswordResetRepository
}

func NewPasswordResetService(
	users *repositories.UserRepository,
	resets *repositories.PasswordResetRepository,
) *PasswordResetService {
	return &PasswordResetService{
		users:  users,
		resets: resets,
	}
}

func (s *PasswordResetService) CreateToken(
	ctx context.Context,
	email string,
) (string, error) {
	user, err := s.users.FindByEmail(ctx, email)
	if err != nil {
		return "", err
	}

	// Do not reveal whether an email address has an account.
	if user == nil {
		return "", nil
	}

	rawToken, err := generatePasswordResetToken()
	if err != nil {
		return "", err
	}

	reset := &models.PasswordReset{
		ID:        uuid.New(),
		UserID:    user.ID,
		TokenHash: hashPasswordResetToken(rawToken),
		ExpiresAt: time.Now().Add(passwordResetTokenTTL),
	}

	if err := s.resets.InvalidateOutstanding(ctx, user.ID, time.Now()); err != nil {
		return "", err
	}

	if err := s.resets.Create(ctx, reset); err != nil {
		return "", err
	}

	return rawToken, nil
}

func (s *PasswordResetService) ResetPassword(
	ctx context.Context,
	rawToken string,
	newPassword string,
) error {
	if len(rawToken) != 64 {
		return ErrInvalidPasswordResetToken
	}

	if len(newPassword) < 8 || len(newPassword) > 72 {
		return ErrInvalidNewPassword
	}

	passwordHash, err := bcrypt.GenerateFromPassword(
		[]byte(newPassword),
		bcrypt.DefaultCost,
	)
	if err != nil {
		return err
	}

	err = s.resets.ResetPassword(
		ctx,
		hashPasswordResetToken(rawToken),
		time.Now(),
		string(passwordHash),
	)
	if errors.Is(err, repositories.ErrInvalidPasswordReset) {
		return ErrInvalidPasswordResetToken
	}

	return err
}
