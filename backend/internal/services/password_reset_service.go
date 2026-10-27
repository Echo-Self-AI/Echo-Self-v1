package services

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/email"
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
	users           *repositories.UserRepository
	resets          *repositories.PasswordResetRepository
	sender          email.Sender
	frontendBaseURL string
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

// SetEmailSender configures the transactional email sender and frontend URL for reset links.
func (s *PasswordResetService) SetEmailSender(sender email.Sender, frontendBaseURL string) {
	s.sender = sender
	s.frontendBaseURL = strings.TrimRight(frontendBaseURL, "/")
}

func (s *PasswordResetService) CreateToken(
	ctx context.Context,
	emailAddress string,
) (string, error) {
	user, err := s.users.FindByEmail(ctx, emailAddress)
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

	if s.sender != nil && user.Email != "" {
		resetURL := fmt.Sprintf("%s/reset-password?token=%s", s.frontendBaseURL, rawToken)
		msg := email.Message{
			To:       user.Email,
			Subject:  "Reset your EchoSelf password",
			TextBody: fmt.Sprintf("A password reset was requested for your EchoSelf account.\n\nReset your password by opening the following link:\n%s\n\nThis link will expire in 30 minutes. If you did not request this, please ignore this email.", resetURL),
			HTMLBody: fmt.Sprintf("<p>A password reset was requested for your EchoSelf account.</p><p>Reset your password by clicking the link below:</p><p><a href=\"%s\">Reset Password</a></p><p>This link will expire in 30 minutes. If you did not request this, please ignore this email.</p>", resetURL),
		}
		if err := s.sender.Send(ctx, msg); err != nil {
			slog.ErrorContext(ctx, "failed to send password reset email", "component", "password_reset", "error", err)
			return "", fmt.Errorf("send password reset email: %w", err)
		}
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
