package middleware

import (
	"github.com/gin-gonic/gin"
)

type middlewareAPIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type middlewareAPIErrorResponse struct {
	Error middlewareAPIError `json:"error"`
}

func respondMiddlewareError(
	c *gin.Context,
	status int,
	code string,
	message string,
) {
	c.AbortWithStatusJSON(status, middlewareAPIErrorResponse{
		Error: middlewareAPIError{
			Code:    code,
			Message: message,
		},
	})
}
