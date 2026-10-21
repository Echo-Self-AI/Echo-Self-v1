package middleware

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

const (
	rateLimitClientTTL       = 10 * time.Minute
	rateLimitCleanupInterval = time.Minute
)

type clientLimiter struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

type rateLimiterStore struct {
	mu          sync.Mutex
	clients     map[string]*clientLimiter
	lastCleanup time.Time
}

func newRateLimiterStore() *rateLimiterStore {
	return &rateLimiterStore{
		clients: make(map[string]*clientLimiter),
	}
}

func (s *rateLimiterStore) getLimiter(
	key string,
	requestsPerSecond rate.Limit,
	burst int,
) *rate.Limiter {
	now := time.Now()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.clients == nil {
		s.clients = make(map[string]*clientLimiter)
	}

	// Perform cleanup during requests instead of starting a
	// background goroutine for every middleware instance.
	if s.lastCleanup.IsZero() ||
		now.Sub(s.lastCleanup) >= rateLimitCleanupInterval {
		cutoff := now.Add(-rateLimitClientTTL)

		for clientIP, client := range s.clients {
			if client.lastSeen.Before(cutoff) {
				delete(s.clients, clientIP)
			}
		}

		s.lastCleanup = now
	}

	client, exists := s.clients[key]
	if !exists {
		client = &clientLimiter{
			limiter: rate.NewLimiter(requestsPerSecond, burst),
		}
		s.clients[key] = client
	}

	client.lastSeen = now
	return client.limiter
}

// RateLimit allows 10 requests per second, with a burst of 20.
func RateLimit() gin.HandlerFunc {
	return RateLimitWithConfig(rate.Limit(10), 20)
}

// RateLimitWithConfig creates a per-IP rate limiter.
func RateLimitWithConfig(
	requestsPerSecond rate.Limit,
	burst int,
) gin.HandlerFunc {
	store := newRateLimiterStore()

	return func(c *gin.Context) {
		limiter := store.getLimiter(
			c.ClientIP(),
			requestsPerSecond,
			burst,
		)

		if !limiter.Allow() {
			respondMiddlewareError(
				c,
				http.StatusTooManyRequests,
				"RATE_LIMIT_EXCEEDED",
				"Too many requests; please try again later",
			)
			return
		}

		c.Next()
	}
}
