package handlers

import "golang.org/x/time/rate"

type authRateLimit struct {
	requestsPerSecond rate.Limit
	burst             int
}

type authRateLimits struct {
	register                 authRateLimit
	login                    authRateLimit
	refresh                  authRateLimit
	emailVerification        authRateLimit
	emailVerificationConfirm authRateLimit
	passwordResetRequest     authRateLimit
	passwordResetConfirm     authRateLimit
}

func defaultAuthRateLimits() authRateLimits {
	return authRateLimits{
		register:                 authRateLimit{rate.Limit(1.0 / 60.0), 3},
		login:                    authRateLimit{rate.Limit(1.0 / 60.0), 5},
		refresh:                  authRateLimit{rate.Limit(1), 10},
		emailVerification:        authRateLimit{rate.Limit(1.0 / 60.0), 3},
		emailVerificationConfirm: authRateLimit{rate.Limit(1.0 / 60.0), 5},
		passwordResetRequest:     authRateLimit{rate.Limit(1.0 / 60.0), 3},
		passwordResetConfirm:     authRateLimit{rate.Limit(1.0 / 60.0), 5},
	}
}
