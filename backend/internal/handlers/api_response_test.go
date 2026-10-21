package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRespondError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		respondError(
			c,
			http.StatusBadRequest,
			"INVALID_REQUEST",
			"The request body is invalid",
		)
	})

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/test", nil)
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d",
			http.StatusBadRequest, recorder.Code)
	}

	var response APIErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}

	if response.Error.Code != "INVALID_REQUEST" {
		t.Errorf("unexpected error code: %q", response.Error.Code)
	}

	if response.Error.Message != "The request body is invalid" {
		t.Errorf("unexpected error message: %q", response.Error.Message)
	}
}

func TestRespondInternalError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		respondInternalError(c, "The operation failed")
	})

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/test", nil)
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("expected status %d, got %d",
			http.StatusInternalServerError, recorder.Code)
	}

	var response APIErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}

	if response.Error.Code != "INTERNAL_SERVER_ERROR" {
		t.Errorf("unexpected error code: %q", response.Error.Code)
	}

	if response.Error.Message != "The operation failed" {
		t.Errorf("unexpected error message: %q", response.Error.Message)
	}
}
