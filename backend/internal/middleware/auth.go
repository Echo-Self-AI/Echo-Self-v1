package middleware

import (
	"net/http"
	"strings"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
)

const UserIDKey = "userID"

func AuthRequired(tokens *services.TokenService) gin.HandlerFunc {
	return func(c *gin.Context) {
		header := c.GetHeader("Authorization")
		parts := strings.Fields(header)

		if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "missing or invalid authorization header",
			})
			return
		}

		claims, err := tokens.ParseAccessToken(parts[1])
		if err != nil || claims.Subject == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "invalid or expired access token",
			})
			return
		}

		c.Set(UserIDKey, claims.Subject)
		c.Next()
	}
}
