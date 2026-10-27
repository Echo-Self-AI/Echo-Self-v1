package email

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestMockSender(t *testing.T) {
	mock := NewMockSender()
	ctx := context.Background()

	msg := Message{
		To:       "user@example.com",
		Subject:  "Test Email",
		TextBody: "Hello world",
		HTMLBody: "<p>Hello world</p>",
	}

	if err := mock.Send(ctx, msg); err != nil {
		t.Fatalf("unexpected send error: %v", err)
	}

	last := mock.LastMessage()
	if last == nil {
		t.Fatal("expected message recorded, got nil")
	}
	if last.To != "user@example.com" {
		t.Errorf("expected recipient user@example.com, got %s", last.To)
	}
	if last.Subject != "Test Email" {
		t.Errorf("expected subject 'Test Email', got %s", last.Subject)
	}
}

func TestLogSender(t *testing.T) {
	logSender := NewLogSender(nil, "noreply@echoself.com")
	ctx := context.Background()

	msg := Message{
		To:       "user@domain.com",
		Subject:  "Verify Your Email",
		TextBody: "Click link to verify",
	}

	if err := logSender.Send(ctx, msg); err != nil {
		t.Fatalf("unexpected send error: %v", err)
	}

	// Empty recipient
	if err := logSender.Send(ctx, Message{}); err != ErrInvalidRecipient {
		t.Errorf("expected ErrInvalidRecipient, got %v", err)
	}
}

func TestBuildMIME(t *testing.T) {
	mime := buildMIME("from@example.com", "to@example.com", "Subject Line", "Plain text body", "<h1>HTML body</h1>")

	if !strings.Contains(mime, "From: from@example.com") {
		t.Error("missing From header")
	}
	if !strings.Contains(mime, "To: to@example.com") {
		t.Error("missing To header")
	}
	if !strings.Contains(mime, "Subject: Subject Line") {
		t.Error("missing Subject header")
	}
	if !strings.Contains(mime, "multipart/alternative") {
		t.Error("missing multipart/alternative header")
	}
	if !strings.Contains(mime, "Plain text body") {
		t.Error("missing plain text body")
	}
	if !strings.Contains(mime, "<h1>HTML body</h1>") {
		t.Error("missing html body")
	}
}

func TestSMTPSender_Validation(t *testing.T) {
	sender := NewSMTPSender(SMTPConfig{
		Host:    "127.0.0.1",
		Port:    1025,
		Timeout: 50 * time.Millisecond,
	})

	ctx := context.Background()

	// Empty To
	err := sender.Send(ctx, Message{
		TextBody: "Hello",
	})
	if err != ErrInvalidRecipient {
		t.Errorf("expected ErrInvalidRecipient, got %v", err)
	}

	// Empty Body
	err = sender.Send(ctx, Message{
		To: "test@example.com",
	})
	if err != ErrEmptyBody {
		t.Errorf("expected ErrEmptyBody, got %v", err)
	}
}
