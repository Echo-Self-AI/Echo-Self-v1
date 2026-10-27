package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

type HealthHandler struct {
	db           *gorm.DB
	pingDatabase func(context.Context) error
	pingRedis    func(context.Context) error
}

func NewHealthHandler(db *gorm.DB) *HealthHandler {
	return &HealthHandler{
		db: db,
		pingDatabase: func(ctx context.Context) error {
			sqlDB, err := db.DB()
			if err != nil {
				return err
			}
			return sqlDB.PingContext(ctx)
		},
	}
}

// SetRedisPinger configures the Redis health check probe for readiness evaluation.
func (h *HealthHandler) SetRedisPinger(pinger func(context.Context) error) {
	h.pingRedis = pinger
}

func (h *HealthHandler) Liveness(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "echoself-api",
	})
}

func (h *HealthHandler) Readiness(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	if h.pingDatabase == nil || h.pingDatabase(ctx) != nil {
		respondError(
			c,
			http.StatusServiceUnavailable,
			"DATABASE_UNAVAILABLE",
			"Database is unavailable",
		)
		return
	}

	if h.pingRedis != nil {
		if err := h.pingRedis(ctx); err != nil {
			respondError(
				c,
				http.StatusServiceUnavailable,
				"REDIS_UNAVAILABLE",
				"Redis is unavailable",
			)
			return
		}
	}

	resp := gin.H{
		"status":   "ready",
		"database": "ok",
	}
	if h.pingRedis != nil {
		resp["redis"] = "ok"
	}

	c.JSON(http.StatusOK, resp)
}
