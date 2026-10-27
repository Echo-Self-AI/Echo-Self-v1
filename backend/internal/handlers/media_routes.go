package handlers

import (
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

type MediaRouteConfig struct {
	MaxUploadSizeBytes  int64
	UploadRatePerMinute int
	UploadRateBurst     int
}

func RegisterMediaRoutes(
	router *gin.Engine,
	handler *MediaHandler,
	tokenService *services.TokenService,
	cfg MediaRouteConfig,
) {
	mediaGroup := router.Group("/api/v1/media", middleware.AuthRequired(tokenService))

	// Convert per-minute rate to rate.Limit (events per second)
	ratePerSec := rate.Limit(float64(cfg.UploadRatePerMinute) / 60.0)
	if ratePerSec <= 0 {
		ratePerSec = rate.Limit(1)
	}
	burst := cfg.UploadRateBurst
	if burst <= 0 {
		burst = 5
	}

	mediaGroup.POST(
		"",
		middleware.RateLimitWithConfig(ratePerSec, burst),
		middleware.RequestSizeLimit(cfg.MaxUploadSizeBytes),
		handler.Upload,
	)

	mediaGroup.GET("", handler.List)
	mediaGroup.GET("/:id", handler.Get)
	mediaGroup.GET("/:id/content", handler.Content)
	mediaGroup.DELETE("/:id", handler.Delete)
}
