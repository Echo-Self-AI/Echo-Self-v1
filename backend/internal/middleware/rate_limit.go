package middleware

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

type clientLimiter struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

type rateLimiterStore struct {
	mu      sync.Mutex
	clients map[string]*clientLimiter
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
	s.mu.Lock()
	defer s.mu.Unlock()

	client, exists := s.clients[key]
	if !exists {
		client = &clientLimiter{
			limiter: rate.NewLimiter(requestsPerSecond, burst),
		}
		s.clients[key] = client
	}

	client.lastSeen = time.Now()
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

	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()

		for range ticker.C {
			cutoff := time.Now().Add(-10 * time.Minute)

			store.mu.Lock()
			for ip, client := range store.clients {
				if client.lastSeen.Before(cutoff) {
					delete(store.clients, ip)
				}
			}
			store.mu.Unlock()
		}
	}()

	return func(c *gin.Context) {
		limiter := store.getLimiter(
			c.ClientIP(),
			requestsPerSecond,
			burst,
		)

		if !limiter.Allow() {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": "too many requests; please try again later",
			})
			return
		}

		c.Next()
	}
}
