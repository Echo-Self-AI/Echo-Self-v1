package processor

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/google/uuid"
)


// getFreePort asks the kernel for a free open TCP port.
func getFreePort(t *testing.T) int {
	addr, err := net.ResolveTCPAddr("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to resolve TCP addr: %v", err)
	}
	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		t.Fatalf("failed to listen on TCP addr: %v", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func TestE2E_GoToPythonAIService(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow end-to-end integration test in short mode")
	}

	// Verify Python / uv is available
	aiDir, err := filepath.Abs("../../../ai")
	if err != nil {
		t.Fatalf("failed to determine ai directory: %v", err)
	}
	if _, err := os.Stat(aiDir); os.IsNotExist(err) {
		t.Skip("ai directory not found, skipping e2e bridge test")
	}

	port := getFreePort(t)
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)

	// Launch Python AI microservice server
	cmd := exec.Command("uv", "run", "python", "-m", "echoself_ai.service.server",
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--device", "cpu",
	)
	cmd.Dir = aiDir
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		t.Skipf("unable to launch Python AI service via uv: %v", err)
	}

	defer func() {
		if cmd.Process != nil {
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()


	// Wait for server to become healthy (up to 10 seconds)
	httpClient := &http.Client{Timeout: 2 * time.Second}
	ready := false
	for i := 0; i < 20; i++ {
		time.Sleep(300 * time.Millisecond)
		resp, err := httpClient.Get(baseURL + "/health")
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			ready = true
			break
		}
		if resp != nil {
			resp.Body.Close()
		}
	}

	if !ready {
		t.Fatalf("Python AI service failed to become ready on %s within timeout", baseURL)
	}

	proc := NewHTTPProcessor(HTTPProcessorConfig{
		BaseURL: baseURL,
		Timeout: 30 * time.Second,
	}, httpClient)

	ctx := context.Background()

	// 1. Verify Health
	health, err := proc.Health(ctx)
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}
	if health["status"] != "healthy" {
		t.Errorf("expected health status 'healthy', got %v", health["status"])
	}

	// 2. Test LLM Generation Job
	jobID := uuid.New()
	userID := uuid.New()
	reqLLM := ProcessRequest{
		JobID:   jobID,
		JobType: "llm_generate",
		UserID:  userID,
		Payload: map[string]any{
			"prompt":         "EchoSelf",
			"max_new_tokens": float64(4),
		},
	}
	resLLM, err := proc.Process(ctx, reqLLM)
	if err != nil {
		t.Fatalf("llm_generate job failed: %v", err)
	}
	if resLLM == nil || resLLM.Output == nil {
		t.Fatalf("expected non-nil output from llm_generate, got %v", resLLM)
	}
	if genText, ok := resLLM.Output["generated_text"].(string); !ok || genText == "" {
		t.Errorf("expected generated_text in output, got %v", resLLM.Output)
	}

	// 3. Register Consent via Go Processor
	reqConsent := ProcessRequest{
		JobID:   uuid.New(),
		JobType: "consent_register",
		UserID:  userID,
		Payload: map[string]any{
			"modality":  "all",
			"avatar_id": "avatar_e2e",
			"scope":     "all",
		},
	}
	resConsent, err := proc.Process(ctx, reqConsent)
	if err != nil {
		t.Fatalf("consent_register job failed: %v", err)
	}
	if resConsent == nil || resConsent.Output == nil {
		t.Fatalf("expected non-nil output from consent_register, got %v", resConsent)
	}

	// 4. Generate Avatar via Go Processor
	reqAvatar := ProcessRequest{
		JobID:   uuid.New(),
		JobType: "avatar_generate",
		UserID:  userID,
		Payload: map[string]any{
			"avatar_id": "avatar_e2e",
		},
	}
	resAvatar, err := proc.Process(ctx, reqAvatar)
	if err != nil {
		t.Fatalf("avatar_generate job failed: %v", err)
	}
	if resAvatar == nil || resAvatar.Output == nil {
		t.Fatalf("expected non-nil output from avatar_generate, got %v", resAvatar)
	}
	if imgB64, ok := resAvatar.Output["image_base64"].(string); !ok || imgB64 == "" {
		t.Errorf("expected image_base64 in avatar output, got %v", resAvatar.Output)
	}

	// 5. Generate Animated Video via Go Processor
	reqVideo := ProcessRequest{
		JobID:   uuid.New(),
		JobType: "video_generate",
		UserID:  userID,
		Payload: map[string]any{
			"prompt":     "waving softly",
			"avatar_id":  "avatar_e2e",
			"num_frames": float64(4),
			"fps":        float64(8),
		},
	}
	resVideo, err := proc.Process(ctx, reqVideo)
	if err != nil {
		t.Fatalf("video_generate job failed: %v", err)
	}
	if resVideo == nil || resVideo.Output == nil {
		t.Fatalf("expected non-nil output from video_generate, got %v", resVideo)
	}
	if gifB64, ok := resVideo.Output["video_gif_base64"].(string); !ok || gifB64 == "" {
		t.Errorf("expected video_gif_base64 in video output, got %v", resVideo.Output)
	}
}
