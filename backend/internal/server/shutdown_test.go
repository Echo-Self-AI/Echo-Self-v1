package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestShutdown(t *testing.T) {
	requestStarted := make(chan struct{})
	releaseRequest := make(chan struct{})

	mux := http.NewServeMux()
	mux.HandleFunc("/slow", func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)

		select {
		case <-releaseRequest:
			w.WriteHeader(http.StatusOK)
		case <-r.Context().Done():
			return
		}
	})

	srv := httptest.NewUnstartedServer(mux)
	srv.Config.ReadHeaderTimeout = 5 * time.Second
	srv.Start()

	clientDone := make(chan error, 1)
	go func() {
		resp, err := srv.Client().Get(srv.URL + "/slow")
		if err != nil {
			clientDone <- err
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			clientDone <- errors.New("unexpected response status")
			return
		}
		clientDone <- nil
	}()

	select {
	case <-requestStarted:
	case <-time.After(2 * time.Second):
		srv.Close()
		t.Fatal("request did not start")
	}

	shutdownDone := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		shutdownDone <- Shutdown(ctx, srv.Config)
	}()

	// The active request should finish before graceful shutdown completes.
	select {
	case err := <-shutdownDone:
		close(releaseRequest)
		srv.Close()
		t.Fatalf("shutdown completed before active request was released: %v", err)
	case <-time.After(100 * time.Millisecond):
	}

	close(releaseRequest)

	select {
	case err := <-shutdownDone:
		if err != nil {
			t.Fatalf("shutdown failed: %v", err)
		}
	case <-time.After(4 * time.Second):
		srv.Close()
		t.Fatal("shutdown did not complete")
	}

	select {
	case err := <-clientDone:
		if err != nil {
			t.Fatalf("client request failed: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("client request did not finish")
	}
}

func TestIsExpectedServeError(t *testing.T) {
	if !IsExpectedServeError(nil) {
		t.Error("nil should be treated as an expected serve result")
	}

	if !IsExpectedServeError(http.ErrServerClosed) {
		t.Error("http.ErrServerClosed should be treated as expected")
	}

	if IsExpectedServeError(errors.New("unexpected server failure")) {
		t.Error("unexpected server errors should not be treated as expected")
	}
}
