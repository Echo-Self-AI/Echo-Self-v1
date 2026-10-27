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
)

var (
	ErrInvalidVerificationToken = errors.New("invalid or expired verification token")
	ErrEmailAlreadyVerified     = errors.New("email is already verified")
)

const emailVerificationTokenTTL = 24 * time.Hour

type EmailVerificationService struct {
	users           *repositories.UserRepository
	verifications   *repositories.EmailVerificationRepository
	sender          email.Sender
	frontendBaseURL string
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

// SetEmailSender configures the transactional email sender and frontend URL for verification links.
func (s *EmailVerificationService) SetEmailSender(sender email.Sender, frontendBaseURL string) {
	s.sender = sender
	s.frontendBaseURL = strings.TrimRight(frontendBaseURL, "/")
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

	if s.sender != nil && user.Email != "" {
		verifyURL := fmt.Sprintf("%s/verify-email?token=%s", s.frontendBaseURL, rawToken)
		msg := email.Message{
			To:       user.Email,
			Subject:  "Verify your EchoSelf account",
			TextBody: fmt.Sprintf("Welcome to EchoSelf!\n\nPlease verify your email by opening the following link:\n%s\n\nThis link will expire in 24 hours.", verifyURL),
			HTMLBody: fmt.Sprintf("<p>Welcome to EchoSelf!</p><p>Please verify your email by clicking the link below:</p><p><a href=\"%s\">Verify Email</a></p><p>This link will expire in 24 hours.</p>", verifyURL),
		}
		if err := s.sender.Send(ctx, msg); err != nil {
			slog.ErrorContext(ctx, "failed to send verification email", "component", "email_verification", "error", err)
			return "", fmt.Errorf("send verification email: %w", err)
		}
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
