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
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{
				"error": "request body too large",
			})
			return false
		}

		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return false
	}

	return true
}
