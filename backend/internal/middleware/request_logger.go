package middleware

import (
	"log/slog"
	"time"

	"github.com/gin-gonic/gin"
)

// RequestLogger logs HTTP request metadata in structured JSON format.
func RequestLogger(logger *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		c.Next()

		route := c.FullPath()
		if route == "" {
			route = "unmatched"
		}

		requestID, _ := c.Get(RequestIDKey)

		logger.Info("http request",
			"request_id", requestID,
			"method", c.Request.Method,
			"route", route,
			"status", c.Writer.Status(),
			"duration_ms", time.Since(start).Milliseconds(),
		)
	}
}
