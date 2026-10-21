package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type APIErrorResponse struct {
	Error APIError `json:"error"`
}

func respondError(c *gin.Context, status int, code, message string) {
	c.AbortWithStatusJSON(status, APIErrorResponse{
		Error: APIError{
			Code:    code,
			Message: message,
		},
	})
}

func respondInternalError(c *gin.Context, message string) {
	respondError(
		c,
		http.StatusInternalServerError,
		"INTERNAL_SERVER_ERROR",
		message,
	)
}

func respondRequestTimeout(c *gin.Context) {
	respondError(
		c,
		http.StatusRequestTimeout,
		"REQUEST_TIMEOUT",
		"The request timed out",
	)
}
