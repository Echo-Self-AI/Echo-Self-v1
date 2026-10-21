package middleware

import (
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const RequestIDKey = "requestID"

func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := uuid.NewString()

		c.Set(RequestIDKey, requestID)
		c.Header("X-Request-ID", requestID)

		c.Next()
	}
}
