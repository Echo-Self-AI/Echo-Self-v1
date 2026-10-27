package services

import (
	"testing"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/email"
)

func TestEmailVerification_WithSender(t *testing.T) {
	mockSender := email.NewMockSender()
	service := &EmailVerificationService{}
	service.SetEmailSender(mockSender, "https://app.echoself.com")

	if service.frontendBaseURL != "https://app.echoself.com" {
		t.Errorf("expected frontendBaseURL 'https://app.echoself.com', got %s", service.frontendBaseURL)
	}
	if service.sender != mockSender {
		t.Error("expected sender to be assigned")
	}
}

func TestPasswordReset_WithSender(t *testing.T) {
	mockSender := email.NewMockSender()
	service := &PasswordResetService{}
	service.SetEmailSender(mockSender, "https://app.echoself.com/")

	if service.frontendBaseURL != "https://app.echoself.com" {
		t.Errorf("expected trailing slash removed from frontendBaseURL, got %s", service.frontendBaseURL)
	}
	if service.sender != mockSender {
		t.Error("expected sender to be assigned")
	}
}
