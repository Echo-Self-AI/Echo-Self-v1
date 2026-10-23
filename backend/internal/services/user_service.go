package services

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/repositories"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

var (
	ErrUserNotFound   = errors.New("user not found")
	ErrInvalidProfile = errors.New("invalid profile details")
)

type UserProfileResult struct {
	ID              string    `json:"id"`
	Email           string    `json:"email"`
	DisplayName     string    `json:"display_name"`
	IsEmailVerified bool      `json:"is_email_verified"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

type UpdateProfileInput struct {
	DisplayName string `json:"display_name"`
}

type UserService struct {
	users *repositories.UserRepository
}

func NewUserService(users *repositories.UserRepository) *UserService {
	return &UserService{users: users}
}

func (s *UserService) GetProfile(
	ctx context.Context,
	userIDStr string,
) (*UserProfileResult, error) {
	if _, err := uuid.Parse(userIDStr); err != nil {
		return nil, ErrUserNotFound
	}

	user, err := s.users.FindByID(ctx, userIDStr)
	if err != nil {
		return nil, fmt.Errorf("find user by id: %w", err)
	}
	if user == nil {
		return nil, ErrUserNotFound
	}

	return &UserProfileResult{
		ID:              user.ID.String(),
		Email:           user.Email,
		DisplayName:     user.DisplayName,
		IsEmailVerified: user.IsEmailVerified,
		CreatedAt:       user.CreatedAt,
		UpdatedAt:       user.UpdatedAt,
	}, nil
}

func (s *UserService) UpdateProfile(
	ctx context.Context,
	userIDStr string,
	input UpdateProfileInput,
) (*UserProfileResult, error) {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrUserNotFound
	}

	name := strings.TrimSpace(input.DisplayName)
	runeCount := utf8.RuneCountInString(name)
	if runeCount == 0 || runeCount > 100 {
		return nil, ErrInvalidProfile
	}

	updated, err := s.users.UpdateProfile(ctx, userID, name)
	if err != nil {
		return nil, fmt.Errorf("update user profile: %w", err)
	}
	if updated == nil {
		return nil, ErrUserNotFound
	}

	return &UserProfileResult{
		ID:              updated.ID.String(),
		Email:           updated.Email,
		DisplayName:     updated.DisplayName,
		IsEmailVerified: updated.IsEmailVerified,
		CreatedAt:       updated.CreatedAt,
		UpdatedAt:       updated.UpdatedAt,
	}, nil
}

func (s *UserService) DeleteAccount(
	ctx context.Context,
	userIDStr string,
) error {
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return ErrUserNotFound
	}

	if err := s.users.DeleteAccount(ctx, userID, time.Now()); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrUserNotFound
		}
		return fmt.Errorf("delete account: %w", err)
	}

	return nil
}
