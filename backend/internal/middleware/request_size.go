package middleware

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

const MaxRequestBodySize int64 = 1 << 20

func RequestSizeLimit(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Body != nil {
			c.Request.Body = http.MaxBytesReader(
				c.Writer,
				c.Request.Body,
				maxBytes,
			)
		}

		c.Next()
	}
}
