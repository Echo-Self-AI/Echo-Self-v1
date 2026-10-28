package processor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// HTTPProcessorConfig configures the HTTP communication with the Python AI service.
type HTTPProcessorConfig struct {
	BaseURL string
	Timeout time.Duration
}

// HTTPProcessor implements Processor over HTTP to communicate with the EchoSelf AI microservice.
type HTTPProcessor struct {
	cfg        HTTPProcessorConfig
	httpClient *http.Client
	jobsMu     sync.RWMutex
	cachedJobs []string
}

// NewHTTPProcessor creates a new HTTPProcessor instance.
func NewHTTPProcessor(cfg HTTPProcessorConfig, client *http.Client) *HTTPProcessor {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")

	if client == nil {
		client = &http.Client{
			Timeout: cfg.Timeout,
		}
	}

	return &HTTPProcessor{
		cfg:        cfg,
		httpClient: client,
		cachedJobs: []string{
			"text_generation",
			"llm_generate",
			"voice_synthesis",
			"voice_embedding",
			"avatar_generate",
			"avatar_embedding",
			"video_generate",
			"consent_verify",
			"consent_register",
			"consent_revoke",
			"consent_purge",
		},
	}
}

// SupportedJobTypes returns the registered AI job types supported by the service.
func (p *HTTPProcessor) SupportedJobTypes() []string {
	p.jobsMu.RLock()
	defer p.jobsMu.RUnlock()

	out := make([]string, len(p.cachedJobs))
	copy(out, p.cachedJobs)
	return out
}

// RefreshSupportedJobTypes fetches the current supported jobs dynamically from /supported_jobs.
func (p *HTTPProcessor) RefreshSupportedJobTypes(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.cfg.BaseURL+"/supported_jobs", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create supported_jobs request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: failed to fetch supported jobs: %v", ErrTransientFailure, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: unexpected status %d fetching supported jobs", ErrTransientFailure, resp.StatusCode)
	}

	var data struct {
		SupportedJobTypes []string `json:"supported_job_types"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode supported jobs: %w", err)
	}

	p.jobsMu.Lock()
	p.cachedJobs = data.SupportedJobTypes
	p.jobsMu.Unlock()

	return data.SupportedJobTypes, nil
}

// Health checks the availability of the AI microservice via /health.
func (p *HTTPProcessor) Health(ctx context.Context) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.cfg.BaseURL+"/health", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create health request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: health check connection error: %v", ErrTransientFailure, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: AI service reported unhealthy status: %d", ErrTransientFailure, resp.StatusCode)
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode health response: %w", err)
	}
	return result, nil
}

// Process executes an AI job by posting a ProcessRequest to /process.
func (p *HTTPProcessor) Process(ctx context.Context, req ProcessRequest) (*ProcessResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("%w: failed to marshal process request: %v", ErrInvalidRequest, err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		p.cfg.BaseURL+"/process",
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to construct HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("%w: connection to AI service failed: %v", ErrTransientFailure, err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%w: failed reading AI service response: %v", ErrTransientFailure, err)
	}

	switch resp.StatusCode {
	case http.StatusOK:
		var result ProcessResult
		if err := json.Unmarshal(respBytes, &result); err != nil {
			return nil, fmt.Errorf("failed to parse AI service result: %w", err)
		}
		return &result, nil

	case http.StatusBadRequest:
		return nil, fmt.Errorf("%w: AI service rejected bad request: %s", ErrInvalidRequest, string(respBytes))

	case http.StatusUnprocessableEntity:
		var failResult struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(respBytes, &failResult)
		errMsg := failResult.Error
		if errMsg == "" {
			errMsg = string(respBytes)
		}
		return nil, fmt.Errorf("%w: AI processing failed: %s", ErrPermanentFailure, errMsg)

	case http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout, http.StatusInternalServerError:
		return nil, fmt.Errorf("%w: AI service temporary error (%d): %s", ErrTransientFailure, resp.StatusCode, string(respBytes))

	default:
		return nil, fmt.Errorf("%w: unexpected HTTP status %d: %s", ErrPermanentFailure, resp.StatusCode, string(respBytes))
	}
}
