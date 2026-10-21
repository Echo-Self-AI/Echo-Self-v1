package handlers

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/services"
	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

func TestAuthRouteRateLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name     string
		path     string
		burst    int
		setLimit func(*authRateLimits)
	}{
		{
			name:  "login",
			path:  "/api/v1/auth/login",
			burst: 5,
			setLimit: func(l *authRateLimits) {
				l.login.requestsPerSecond = rate.Limit(0)
			},
		},
		{
			name:  "registration",
			path:  "/api/v1/auth/register",
			burst: 3,
			setLimit: func(l *authRateLimits) {
				l.register.requestsPerSecond = rate.Limit(0)
			},
		},
		{
			name:  "password reset request",
			path:  "/api/v1/auth/password-reset/request",
			burst: 3,
			setLimit: func(l *authRateLimits) {
				l.passwordResetRequest.requestsPerSecond = rate.Limit(0)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limits := defaultAuthRateLimits()
			tt.setLimit(&limits)

			router := gin.New()
			authHandler := &AuthHandler{}

			// The real route-registration function is used.
			// A nil token service is safe here because these tested
			// routes do not use the authentication middleware.
			registerAuthRoutesWithLimits(
				router,
				authHandler,
				(*services.TokenService)(nil),
				limits,
			)

			for i := 0; i < tt.burst; i++ {
				req := httptest.NewRequest(http.MethodPost, tt.path, nil)
				req.RemoteAddr = "192.0.2.80:12345"

				recorder := httptest.NewRecorder()
				router.ServeHTTP(recorder, req)

				if recorder.Code != http.StatusBadRequest {
					t.Fatalf(
						"request %d: expected handler's %d response, got %d",
						i+1,
						http.StatusBadRequest,
						recorder.Code,
					)
				}
			}

			req := httptest.NewRequest(http.MethodPost, tt.path, nil)
			req.RemoteAddr = "192.0.2.80:12345"

			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			if recorder.Code != http.StatusTooManyRequests {
				t.Fatalf(
					"request after burst: expected %d, got %d",
					http.StatusTooManyRequests,
					recorder.Code,
				)
			}
		})
	}
}
