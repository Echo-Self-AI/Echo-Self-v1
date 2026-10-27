package email

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/smtp"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	ErrInvalidRecipient = errors.New("invalid email recipient")
	ErrEmptyBody        = errors.New("email body cannot be empty")
	ErrDeliveryFailed   = errors.New("email delivery failed")
)

// Message encapsulates the email to be sent.
type Message struct {
	To       string
	Subject  string
	TextBody string
	HTMLBody string
}

// Sender defines the interface for transmitting transactional emails.
type Sender interface {
	Send(ctx context.Context, msg Message) error
}

// MockSender is an in-memory sender for testing.
type MockSender struct {
	mu       sync.Mutex
	Messages []Message
	Err      error
}

func NewMockSender() *MockSender {
	return &MockSender{
		Messages: make([]Message, 0),
	}
}

func (m *MockSender) Send(ctx context.Context, msg Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.Err != nil {
		return m.Err
	}
	m.Messages = append(m.Messages, msg)
	return nil
}

func (m *MockSender) LastMessage() *Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.Messages) == 0 {
		return nil
	}
	msg := m.Messages[len(m.Messages)-1]
	return &msg
}

// LogSender logs email dispatch metadata via slog without sending real network packets.
type LogSender struct {
	logger *slog.Logger
	from   string
}

func NewLogSender(logger *slog.Logger, from string) *LogSender {
	if logger == nil {
		logger = slog.Default()
	}
	if from == "" {
		from = "noreply@echoself.internal"
	}
	return &LogSender{
		logger: logger,
		from:   from,
	}
}

func (l *LogSender) Send(ctx context.Context, msg Message) error {
	if msg.To == "" {
		return ErrInvalidRecipient
	}
	// Log recipient domain and subject only, never raw tokens or link content
	domain := ""
	if parts := strings.Split(msg.To, "@"); len(parts) == 2 {
		domain = parts[1]
	}
	l.logger.InfoContext(ctx, "transactional email simulated",
		"component", "email_sender",
		"recipient_domain", domain,
		"subject", msg.Subject,
		"has_html", msg.HTMLBody != "",
		"has_text", msg.TextBody != "",
	)
	return nil
}

// SMTPConfig holds parameters for SMTP delivery.
type SMTPConfig struct {
	Host     string
	Port     int
	Username string
	Password string
	From     string
	UseTLS   bool
	Timeout  time.Duration
}

// SMTPSender delivers emails over SMTP with timeouts and TLS support.
type SMTPSender struct {
	cfg SMTPConfig
}

func NewSMTPSender(cfg SMTPConfig) *SMTPSender {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Second
	}
	if cfg.From == "" {
		cfg.From = "noreply@echoself.com"
	}
	return &SMTPSender{cfg: cfg}
}

func (s *SMTPSender) Send(ctx context.Context, msg Message) error {
	if strings.TrimSpace(msg.To) == "" {
		return ErrInvalidRecipient
	}
	if msg.TextBody == "" && msg.HTMLBody == "" {
		return ErrEmptyBody
	}

	addr := net.JoinHostPort(s.cfg.Host, strconv.Itoa(s.cfg.Port))

	deadline, ok := ctx.Deadline()
	timeout := s.cfg.Timeout
	if ok {
		remaining := time.Until(deadline)
		if remaining < timeout {
			timeout = remaining
		}
	}

	dialer := &net.Dialer{Timeout: timeout}
	conn, err := dialer.DialContext(ctx, "tcp", addr)
	if err != nil {
		return fmt.Errorf("%w: dial: %v", ErrDeliveryFailed, err)
	}
	defer conn.Close()

	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	} else {
		_ = conn.SetDeadline(time.Now().Add(timeout))
	}

	client, err := smtp.NewClient(conn, s.cfg.Host)
	if err != nil {
		return fmt.Errorf("%w: smtp client init: %v", ErrDeliveryFailed, err)
	}
	defer client.Quit()

	if s.cfg.UseTLS {
		tlsConfig := &tls.Config{
			ServerName: s.cfg.Host,
			MinVersion: tls.VersionTLS12,
		}
		if ok, _ := client.Extension("STARTTLS"); ok {
			if err := client.StartTLS(tlsConfig); err != nil {
				return fmt.Errorf("%w: starttls: %v", ErrDeliveryFailed, err)
			}
		}
	}

	if s.cfg.Username != "" && s.cfg.Password != "" {
		auth := smtp.PlainAuth("", s.cfg.Username, s.cfg.Password, s.cfg.Host)
		if err := client.Auth(auth); err != nil {
			return fmt.Errorf("%w: auth: %v", ErrDeliveryFailed, err)
		}
	}

	if err := client.Mail(s.cfg.From); err != nil {
		return fmt.Errorf("%w: mail from: %v", ErrDeliveryFailed, err)
	}
	if err := client.Rcpt(msg.To); err != nil {
		return fmt.Errorf("%w: rcpt to: %v", ErrDeliveryFailed, err)
	}

	w, err := client.Data()
	if err != nil {
		return fmt.Errorf("%w: data: %v", ErrDeliveryFailed, err)
	}
	defer w.Close()

	payload := buildMIME(s.cfg.From, msg.To, msg.Subject, msg.TextBody, msg.HTMLBody)
	if _, err := w.Write([]byte(payload)); err != nil {
		return fmt.Errorf("%w: write: %v", ErrDeliveryFailed, err)
	}

	return nil
}

func buildMIME(from, to, subject, textBody, htmlBody string) string {
	boundary := fmt.Sprintf("boundary_%d", time.Now().UnixNano())

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("From: %s\r\n", from))
	sb.WriteString(fmt.Sprintf("To: %s\r\n", to))
	sb.WriteString(fmt.Sprintf("Subject: %s\r\n", subject))
	sb.WriteString("MIME-Version: 1.0\r\n")

	if htmlBody != "" && textBody != "" {
		sb.WriteString(fmt.Sprintf("Content-Type: multipart/alternative; boundary=\"%s\"\r\n\r\n", boundary))

		// Plain text part
		sb.WriteString(fmt.Sprintf("--%s\r\n", boundary))
		sb.WriteString("Content-Type: text/plain; charset=UTF-8\r\n")
		sb.WriteString("Content-Transfer-Encoding: 7bit\r\n\r\n")
		sb.WriteString(textBody)
		sb.WriteString("\r\n\r\n")

		// HTML part
		sb.WriteString(fmt.Sprintf("--%s\r\n", boundary))
		sb.WriteString("Content-Type: text/html; charset=UTF-8\r\n")
		sb.WriteString("Content-Transfer-Encoding: 7bit\r\n\r\n")
		sb.WriteString(htmlBody)
		sb.WriteString("\r\n\r\n")

		sb.WriteString(fmt.Sprintf("--%s--\r\n", boundary))
	} else if htmlBody != "" {
		sb.WriteString("Content-Type: text/html; charset=UTF-8\r\n\r\n")
		sb.WriteString(htmlBody)
		sb.WriteString("\r\n")
	} else {
		sb.WriteString("Content-Type: text/plain; charset=UTF-8\r\n\r\n")
		sb.WriteString(textBody)
		sb.WriteString("\r\n")
	}

	return sb.String()
}
