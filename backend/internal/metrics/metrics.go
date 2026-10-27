package metrics

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

// Registry holds the application metrics in a thread-safe manner.
type Registry struct {
	mu sync.RWMutex

	// HTTP metrics
	activeRequests  int64
	httpRequests    map[string]*uint64 // key: method|status|path
	httpDurationSum map[string]*uint64 // key: method|status|path -> nanoseconds
	httpDurationCnt map[string]*uint64 // key: method|status|path

	// Background job metrics
	jobStates       map[string]*uint64 // key: type|status
	jobDurationSum  map[string]*uint64 // key: type -> nanoseconds
	jobDurationCnt  map[string]*uint64 // key: type
	jobRetries      map[string]*uint64 // key: type
	leaseFailures   uint64
	staleRecoveries uint64

	// Outbox metrics
	outboxDispatches  map[string]*uint64 // key: status
	outboxDurationSum uint64             // nanoseconds
	outboxDurationCnt uint64
}

// NewRegistry initializes an empty Registry.
func NewRegistry() *Registry {
	return &Registry{
		httpRequests:     make(map[string]*uint64),
		httpDurationSum:  make(map[string]*uint64),
		httpDurationCnt:  make(map[string]*uint64),
		jobStates:        make(map[string]*uint64),
		jobDurationSum:   make(map[string]*uint64),
		jobDurationCnt:   make(map[string]*uint64),
		jobRetries:       make(map[string]*uint64),
		outboxDispatches: make(map[string]*uint64),
	}
}

// DefaultRegistry is the global metrics registry used by the application.
var DefaultRegistry = NewRegistry()

func (r *Registry) getOrCreateCounter(m map[string]*uint64, key string) *uint64 {
	r.mu.RLock()
	if ptr, ok := m[key]; ok {
		r.mu.RUnlock()
		return ptr
	}
	r.mu.RUnlock()

	r.mu.Lock()
	defer r.mu.Unlock()
	if ptr, ok := m[key]; ok {
		return ptr
	}
	var val uint64
	m[key] = &val
	return &val
}

// IncActiveRequests increments the gauge of currently active HTTP requests.
func (r *Registry) IncActiveRequests() {
	atomic.AddInt64(&r.activeRequests, 1)
}

// DecActiveRequests decrements the gauge of currently active HTTP requests.
func (r *Registry) DecActiveRequests() {
	atomic.AddInt64(&r.activeRequests, -1)
}

// RecordHTTPRequest records an HTTP request outcome and latency.
func (r *Registry) RecordHTTPRequest(method, path string, status int, duration time.Duration) {
	cleanPath := normalizePath(path)
	cleanMethod := strings.ToUpper(method)
	statusStr := strconv.Itoa(status)
	key := cleanMethod + "|" + statusStr + "|" + cleanPath

	cntPtr := r.getOrCreateCounter(r.httpRequests, key)
	atomic.AddUint64(cntPtr, 1)

	durSumPtr := r.getOrCreateCounter(r.httpDurationSum, key)
	atomic.AddUint64(durSumPtr, uint64(duration.Nanoseconds()))

	durCntPtr := r.getOrCreateCounter(r.httpDurationCnt, key)
	atomic.AddUint64(durCntPtr, 1)
}

// RecordJobState records a transition or current state for a background job.
func (r *Registry) RecordJobState(jobType, status string) {
	cleanType := cleanLabel(jobType)
	cleanStatus := cleanLabel(status)
	key := cleanType + "|" + cleanStatus

	cntPtr := r.getOrCreateCounter(r.jobStates, key)
	atomic.AddUint64(cntPtr, 1)
}

// RecordJobDuration records the processing duration of a finished job.
func (r *Registry) RecordJobDuration(jobType string, duration time.Duration) {
	cleanType := cleanLabel(jobType)

	sumPtr := r.getOrCreateCounter(r.jobDurationSum, cleanType)
	atomic.AddUint64(sumPtr, uint64(duration.Nanoseconds()))

	cntPtr := r.getOrCreateCounter(r.jobDurationCnt, cleanType)
	atomic.AddUint64(cntPtr, 1)
}

// RecordJobRetry records a scheduled job retry.
func (r *Registry) RecordJobRetry(jobType string) {
	cleanType := cleanLabel(jobType)
	cntPtr := r.getOrCreateCounter(r.jobRetries, cleanType)
	atomic.AddUint64(cntPtr, 1)
}

// RecordLeaseRenewalFailure increments the lease failure counter.
func (r *Registry) RecordLeaseRenewalFailure() {
	atomic.AddUint64(&r.leaseFailures, 1)
}

// RecordStaleJobRecovery increments the stale job recovery counter.
func (r *Registry) RecordStaleJobRecovery() {
	atomic.AddUint64(&r.staleRecoveries, 1)
}

// RecordOutboxDispatch records outbox batch execution.
func (r *Registry) RecordOutboxDispatch(status string, duration time.Duration) {
	cleanStatus := cleanLabel(status)
	cntPtr := r.getOrCreateCounter(r.outboxDispatches, cleanStatus)
	atomic.AddUint64(cntPtr, 1)

	atomic.AddUint64(&r.outboxDurationSum, uint64(duration.Nanoseconds()))
	atomic.AddUint64(&r.outboxDurationCnt, 1)
}

// ExposePrometheus formats all metrics into Prometheus text format.
func (r *Registry) ExposePrometheus() string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var sb strings.Builder

	// 1. Active requests
	sb.WriteString("# HELP echoself_http_active_requests Currently executing HTTP requests.\n")
	sb.WriteString("# TYPE echoself_http_active_requests gauge\n")
	active := atomic.LoadInt64(&r.activeRequests)
	if active < 0 {
		active = 0
	}
	sb.WriteString(fmt.Sprintf("echoself_http_active_requests %d\n\n", active))

	// 2. HTTP requests total
	sb.WriteString("# HELP echoself_http_requests_total Total number of HTTP requests processed.\n")
	sb.WriteString("# TYPE echoself_http_requests_total counter\n")
	for key, ptr := range r.httpRequests {
		parts := strings.Split(key, "|")
		if len(parts) == 3 {
			val := atomic.LoadUint64(ptr)
			sb.WriteString(fmt.Sprintf("echoself_http_requests_total{method=%q,status=%q,path=%q} %d\n", parts[0], parts[1], parts[2], val))
		}
	}
	sb.WriteString("\n")

	// 3. HTTP request duration
	sb.WriteString("# HELP echoself_http_request_duration_seconds HTTP request latencies in seconds.\n")
	sb.WriteString("# TYPE echoself_http_request_duration_seconds summary\n")
	for key, sumPtr := range r.httpDurationSum {
		cntPtr := r.httpDurationCnt[key]
		if cntPtr != nil {
			parts := strings.Split(key, "|")
			if len(parts) == 3 {
				sumSec := float64(atomic.LoadUint64(sumPtr)) / 1e9
				cntVal := atomic.LoadUint64(cntPtr)
				sb.WriteString(fmt.Sprintf("echoself_http_request_duration_seconds_sum{method=%q,status=%q,path=%q} %.6f\n", parts[0], parts[1], parts[2], sumSec))
				sb.WriteString(fmt.Sprintf("echoself_http_request_duration_seconds_count{method=%q,status=%q,path=%q} %d\n", parts[0], parts[1], parts[2], cntVal))
			}
		}
	}
	sb.WriteString("\n")

	// 4. Job states
	sb.WriteString("# HELP echoself_jobs_total Total background jobs processed by type and outcome.\n")
	sb.WriteString("# TYPE echoself_jobs_total counter\n")
	for key, ptr := range r.jobStates {
		parts := strings.Split(key, "|")
		if len(parts) == 2 {
			val := atomic.LoadUint64(ptr)
			sb.WriteString(fmt.Sprintf("echoself_jobs_total{type=%q,status=%q} %d\n", parts[0], parts[1], val))
		}
	}
	sb.WriteString("\n")

	// 5. Job duration
	sb.WriteString("# HELP echoself_job_duration_seconds Processing duration of jobs in seconds.\n")
	sb.WriteString("# TYPE echoself_job_duration_seconds summary\n")
	for jobType, sumPtr := range r.jobDurationSum {
		cntPtr := r.jobDurationCnt[jobType]
		if cntPtr != nil {
			sumSec := float64(atomic.LoadUint64(sumPtr)) / 1e9
			cntVal := atomic.LoadUint64(cntPtr)
			sb.WriteString(fmt.Sprintf("echoself_job_duration_seconds_sum{type=%q} %.6f\n", jobType, sumSec))
			sb.WriteString(fmt.Sprintf("echoself_job_duration_seconds_count{type=%q} %d\n", jobType, cntVal))
		}
	}
	sb.WriteString("\n")

	// 6. Job retries
	sb.WriteString("# HELP echoself_job_retries_total Total background job retries scheduled.\n")
	sb.WriteString("# TYPE echoself_job_retries_total counter\n")
	for jobType, ptr := range r.jobRetries {
		val := atomic.LoadUint64(ptr)
		sb.WriteString(fmt.Sprintf("echoself_job_retries_total{type=%q} %d\n", jobType, val))
	}
	sb.WriteString("\n")

	// 7. Lease and recovery operations
	sb.WriteString("# HELP echoself_job_lease_renewal_failures_total Total worker lease renewal failures.\n")
	sb.WriteString("# TYPE echoself_job_lease_renewal_failures_total counter\n")
	sb.WriteString(fmt.Sprintf("echoself_job_lease_renewal_failures_total %d\n\n", atomic.LoadUint64(&r.leaseFailures)))

	sb.WriteString("# HELP echoself_job_stale_recoveries_total Total orphaned or stale jobs recovered.\n")
	sb.WriteString("# TYPE echoself_job_stale_recoveries_total counter\n")
	sb.WriteString(fmt.Sprintf("echoself_job_stale_recoveries_total %d\n\n", atomic.LoadUint64(&r.staleRecoveries)))

	// 8. Outbox metrics
	sb.WriteString("# HELP echoself_outbox_dispatches_total Total outbox dispatch attempts.\n")
	sb.WriteString("# TYPE echoself_outbox_dispatches_total counter\n")
	for status, ptr := range r.outboxDispatches {
		val := atomic.LoadUint64(ptr)
		sb.WriteString(fmt.Sprintf("echoself_outbox_dispatches_total{status=%q} %d\n", status, val))
	}
	sb.WriteString("\n")

	return sb.String()
}

// Handler returns a Gin HandlerFunc exposing metrics in Prometheus text format.
func Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		output := DefaultRegistry.ExposePrometheus()
		c.Data(http.StatusOK, "text/plain; version=0.04; charset=utf-8", []byte(output))
	}
}

// Middleware creates a Gin middleware to record active requests, totals, and latencies.
func Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip metrics endpoint itself to avoid recursive skew
		if c.Request.URL.Path == "/metrics" {
			c.Next()
			return
		}

		DefaultRegistry.IncActiveRequests()
		start := time.Now()

		c.Next()

		duration := time.Since(start)
		DefaultRegistry.DecActiveRequests()

		// Prefer gin matched route path to avoid unbounded label cardinality
		matchedPath := c.FullPath()
		if matchedPath == "" {
			matchedPath = normalizePath(c.Request.URL.Path)
		}

		DefaultRegistry.RecordHTTPRequest(c.Request.Method, matchedPath, c.Writer.Status(), duration)
	}
}

func normalizePath(path string) string {
	if path == "" {
		return "/"
	}
	// Avoid unbounded cardinality: if path has > 5 segments, truncate
	segments := strings.Split(strings.Trim(path, "/"), "/")
	if len(segments) > 5 {
		return "/" + strings.Join(segments[:5], "/") + "/..."
	}
	return path
}

func cleanLabel(val string) string {
	if val == "" {
		return "unknown"
	}
	// Strip characters that break prometheus labels
	val = strings.ReplaceAll(val, "\"", "")
	val = strings.ReplaceAll(val, "\n", "")
	if len(val) > 64 {
		val = val[:64]
	}
	return val
}
