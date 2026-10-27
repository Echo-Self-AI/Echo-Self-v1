package storage

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestMemoryStorage(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStorage()

	key := "test/sample.txt"
	content := "Hello Cloudflare R2"
	contentType := "text/plain"

	// Initial check: object does not exist
	exists, err := store.Exists(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error checking existence: %v", err)
	}
	if exists {
		t.Fatal("expected object to not exist initially")
	}

	// Upload object
	err = store.Upload(ctx, key, strings.NewReader(content), int64(len(content)), contentType)
	if err != nil {
		t.Fatalf("unexpected error uploading: %v", err)
	}

	// Check exists
	exists, err = store.Exists(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error checking existence: %v", err)
	}
	if !exists {
		t.Fatal("expected object to exist after upload")
	}

	// Download object
	rc, size, ct, err := store.Download(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error downloading: %v", err)
	}
	defer rc.Close()

	if size != int64(len(content)) {
		t.Fatalf("expected size %d, got %d", len(content), size)
	}
	if ct != contentType {
		t.Fatalf("expected content type %q, got %q", contentType, ct)
	}

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, rc); err != nil {
		t.Fatalf("unexpected error reading stream: %v", err)
	}
	if buf.String() != content {
		t.Fatalf("expected body %q, got %q", content, buf.String())
	}

	// Delete object
	err = store.Delete(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error deleting: %v", err)
	}

	// Verify not found after delete
	_, _, _, err = store.Download(ctx, key)
	if !errors.Is(err, ErrObjectNotFound) {
		t.Fatalf("expected ErrObjectNotFound, got %v", err)
	}

	exists, err = store.Exists(ctx, key)
	if err != nil {
		t.Fatalf("unexpected error checking existence: %v", err)
	}
	if exists {
		t.Fatal("expected object to not exist after delete")
	}
}

func TestMemoryStorage_Errors(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStorage()

	// Empty key errors
	if err := store.Upload(ctx, "", strings.NewReader("data"), 4, "text/plain"); !errors.Is(err, ErrInvalidKey) {
		t.Fatalf("expected ErrInvalidKey, got %v", err)
	}
	if _, _, _, err := store.Download(ctx, ""); !errors.Is(err, ErrInvalidKey) {
		t.Fatalf("expected ErrInvalidKey, got %v", err)
	}
	if err := store.Delete(ctx, ""); !errors.Is(err, ErrInvalidKey) {
		t.Fatalf("expected ErrInvalidKey, got %v", err)
	}
	if _, err := store.Exists(ctx, ""); !errors.Is(err, ErrInvalidKey) {
		t.Fatalf("expected ErrInvalidKey, got %v", err)
	}

	// Injected errors
	simulatedErr := errors.New("simulated storage failure")
	store.FailUpload = simulatedErr
	if err := store.Upload(ctx, "k", strings.NewReader("data"), 4, "text/plain"); !errors.Is(err, simulatedErr) {
		t.Fatalf("expected simulated error, got %v", err)
	}
	store.FailUpload = nil

	store.FailDownload = simulatedErr
	if _, _, _, err := store.Download(ctx, "k"); !errors.Is(err, simulatedErr) {
		t.Fatalf("expected simulated error, got %v", err)
	}
	store.FailDownload = nil

	store.FailDelete = simulatedErr
	if err := store.Delete(ctx, "k"); !errors.Is(err, simulatedErr) {
		t.Fatalf("expected simulated error, got %v", err)
	}
	store.FailDelete = nil

	store.FailExists = simulatedErr
	if _, err := store.Exists(ctx, "k"); !errors.Is(err, simulatedErr) {
		t.Fatalf("expected simulated error, got %v", err)
	}
}

func TestR2Storage_Validation(t *testing.T) {
	ctx := context.Background()

	// Missing bucket name
	_, err := NewR2Storage(ctx, R2Config{
		AccountID:       "acc123",
		AccessKeyID:     "key123",
		SecretAccessKey: "sec123",
	})
	if err == nil {
		t.Fatal("expected error for missing bucket name")
	}

	// Missing credentials
	_, err = NewR2Storage(ctx, R2Config{
		BucketName: "my-bucket",
	})
	if err == nil {
		t.Fatal("expected error for missing credentials")
	}

	// Missing account ID and endpoint
	_, err = NewR2Storage(ctx, R2Config{
		BucketName:      "my-bucket",
		AccessKeyID:     "key123",
		SecretAccessKey: "sec123",
	})
	if err == nil {
		t.Fatal("expected error for missing account ID and endpoint")
	}

	// Valid config with Account ID
	client, err := NewR2Storage(ctx, R2Config{
		BucketName:      "my-bucket",
		AccountID:       "acc123",
		AccessKeyID:     "key123",
		SecretAccessKey: "sec123",
	})
	if err != nil {
		t.Fatalf("unexpected error initializing R2 storage: %v", err)
	}
	if client == nil {
		t.Fatal("expected client to be initialized")
	}

	// Valid config with custom Endpoint
	client, err = NewR2Storage(ctx, R2Config{
		BucketName:      "my-bucket",
		Endpoint:        "https://custom-r2.example.com",
		AccessKeyID:     "key123",
		SecretAccessKey: "sec123",
		Region:          "auto",
	})
	if err != nil {
		t.Fatalf("unexpected error initializing R2 storage with endpoint: %v", err)
	}
	if client == nil {
		t.Fatal("expected client to be initialized")
	}
}
