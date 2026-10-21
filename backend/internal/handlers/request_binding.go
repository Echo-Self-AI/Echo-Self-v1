package handlers

import (
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"
)

func bindJSON(c *gin.Context, destination any) bool {
	if err := c.ShouldBindJSON(destination); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			respondError(
				c,
				http.StatusRequestEntityTooLarge,
				"REQUEST_BODY_TOO_LARGE",
				"Request body exceeds the allowed size",
			)
			return false
		}

		respondError(
			c,
			http.StatusBadRequest,
			"INVALID_REQUEST_BODY",
			"The request body is invalid",
		)
		return false
	}

	return true
}
