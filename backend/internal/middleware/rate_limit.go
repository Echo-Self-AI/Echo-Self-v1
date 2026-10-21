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

func (s *rateLimiterStore) getLimiter(key string) *rate.Limiter {
	s.mu.Lock()
	defer s.mu.Unlock()

	client, exists := s.clients[key]
	if !exists {
		client = &clientLimiter{
			limiter: rate.NewLimiter(10, 20),
		}
		s.clients[key] = client
	}

	client.lastSeen = time.Now()
	return client.limiter
}

func RateLimit() gin.HandlerFunc {
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
		ip := c.ClientIP()
		limiter := store.getLimiter(ip)

		if !limiter.Allow() {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": "too many requests; please try again later",
			})
			return
		}

		c.Next()
	}
}
