package storage

import (
	"context"
	"errors"
	"io"
)

var (
	ErrObjectNotFound = errors.New("storage: object not found")
	ErrInvalidKey     = errors.New("storage: invalid object key")
)

// Storage defines the minimal abstraction required for object storage operations.
type Storage interface {
	Upload(ctx context.Context, key string, body io.Reader, size int64, contentType string) error
	Download(ctx context.Context, key string) (io.ReadCloser, int64, string, error)
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
}
