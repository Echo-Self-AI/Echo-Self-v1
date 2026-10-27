package handlers

import (
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/middleware"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

type JobRouteConfig struct {
	SubmitRatePerSecond rate.Limit
	SubmitBurst         int
}

func RegisterJobRoutes(
	router *gin.Engine,
	handler *JobHandler,
	tokenService *services.TokenService,
	cfg JobRouteConfig,
) {
	if cfg.SubmitRatePerSecond <= 0 {
		cfg.SubmitRatePerSecond = rate.Limit(5)
	}
	if cfg.SubmitBurst <= 0 {
		cfg.SubmitBurst = 10
	}

	jobsGroup := router.Group("/api/v1/jobs", middleware.AuthRequired(tokenService))

	jobsGroup.POST(
		"",
		middleware.RateLimitWithConfig(cfg.SubmitRatePerSecond, cfg.SubmitBurst),
		handler.Submit,
	)

	jobsGroup.GET("", handler.List)
	jobsGroup.GET("/:id", handler.Get)
	jobsGroup.POST("/:id/cancel", handler.Cancel)
}
