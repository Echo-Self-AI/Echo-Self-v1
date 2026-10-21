package server

import (
	"context"
	"errors"
	"net/http"
)

// Shutdown gracefully stops the HTTP server.
func Shutdown(ctx context.Context, srv *http.Server) error {
	err := srv.Shutdown(ctx)
	if err != nil {
		_ = srv.Close()
		return err
	}

	return nil
}

// IsExpectedServeError reports whether the server stopped normally.
func IsExpectedServeError(err error) bool {
	return err == nil || errors.Is(err, http.ErrServerClosed)
}
