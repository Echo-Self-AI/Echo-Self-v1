# EchoSelf Go Backend — Production MVP Manual

EchoSelf is a secure, privacy-first AI digital avatar backend built in Go. This repository contains the complete Clean Architecture Go backend implementing user lifecycle management, granular purpose-specific consent auditing, authenticated private media streaming, transactional outbox dispatch, and distributed Redis background job workers.

---

## 1. Architecture Overview

The backend is composed of two primary independent binaries and a migration utility:
- **`cmd/api` (REST API Service):** Gin-based HTTP server managing authentication, consent verification, media uploads/streaming, job submission, and Prometheus metrics.
- **`cmd/worker` (Background Worker Daemon):** Headless background engine running the transactional outbox dispatcher, Redis queue consumer, lease heartbeat renewal, stale job sweeper, and orphaned storage object purger. It also exposes a lightweight HTTP probe server on port `8081` for container readiness/liveness.
- **`cmd/migrate` (Database Migration Runner):** Standalone zero-dependency CLI for applying, stepping, rolling back, forcing, and baselining versioned SQL schema migrations.

```
       [ Client (Web/Mobile) ]
                  |
     HTTPS (Reverse Proxy / CDN)
                  |
      +-----------v-----------+
      |  cmd/api (Port 8080)  | <---- (Liveness/Readiness, Prometheus /metrics)
      +-----------+-----------+
                  |
         +--------+--------+
         |                 |
         v                 v
   [ PostgreSQL 16 ]    [ Redis 7 ]
         ^                 ^
         |                 |
      +--+-----------------+--+
      | cmd/worker (Port 8081)| <---- (Probe Server /health, /ready, /metrics)
      +-----------+-----------+
                  |
                  v
       [ Cloudflare R2 / S3 ]
```

---

## 2. Environment Variables & Configuration Reference

Configure these settings via `.env` or system environment variables:

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_ENV` | string | `development` | Environment mode (`development`, `test`, `production`). In production, Gin runs in ReleaseMode and strict validations are enforced. |
| `APP_PORT` | string | `8080` | Port for the HTTP API server. |
| `DATABASE_URL` | string | (required) | PostgreSQL connection URI (`postgres://user:pass@host:5432/db?sslmode=disable`). |
| `TEST_DATABASE_URL` | string | optional | Isolated database URI for integration tests. |
| `JWT_SECRET` | string | (required) | HMAC-SHA256 secret key. Must be $\ge 32$ characters in production. |
| `AUTO_MIGRATE` | bool | `false` | When `true`, automatically applies pending migrations on startup. Keep `false` in multi-replica production environments. |
| `FRONTEND_BASE_URL` | string | `http://localhost:3000` | Base URL used to construct transactional email verification and password reset links. Must be a valid HTTP/HTTPS URL. |
| `WORKER_ENABLED` | bool | `false` | Flag indicating worker mode. |
| `WORKER_CONCURRENCY` | int | `2` | Number of parallel job-processing goroutines per worker instance. |
| `WORKER_HEALTH_PORT`| string | `8081` | Port for the background worker's HTTP probe server (`/health`, `/ready`, `/metrics`). |
| `JOB_MAX_ATTEMPTS` | int | `3` | Maximum retry attempts for transient worker failures before marking terminal failure. |
| `JOB_LEASE_DURATION_SECONDS` | int | `30` | Distributed lease duration in seconds for claimed jobs. |
| `JOB_DRAIN_TIMEOUT_SECONDS` | int | `10` | Maximum time to wait for active jobs to finish during graceful worker shutdown. |
| `REDIS_URL` | string | `redis://localhost:6379/0` | Redis connection URI. |
| `REDIS_POOL_SIZE` | int | `10` | Maximum Redis client connection pool size. |
| `R2_ENABLED` | bool | `false` | When `true`, enables Cloudflare R2 storage; when `false`, uses an in-memory storage driver for local development and tests. |
| `R2_ACCOUNT_ID` | string | optional | Cloudflare Account ID. Required when `R2_ENABLED=true`. |
| `R2_ACCESS_KEY_ID` | string | optional | Cloudflare R2 Access Key ID. |
| `R2_SECRET_ACCESS_KEY`| string | optional | Cloudflare R2 Secret Access Key. |
| `R2_BUCKET_NAME` | string | `echoself-private-media` | R2 bucket name. |
| `R2_ENDPOINT` | string | optional | Custom S3/R2 API endpoint URL. |
| `R2_REGION` | string | `auto` | Storage region. |
| `MEDIA_MAX_UPLOAD_SIZE_BYTES` | int | `10485760` (10MB) | Maximum file upload size. |
| `MEDIA_USER_QUOTA_BYTES` | int | `52428800` (50MB) | Aggregate storage quota allowed per user. |
| `MEDIA_USER_MAX_OBJECTS` | int | `20` | Maximum number of uploaded media items allowed per user. |
| `MEDIA_ALLOWED_TYPES` | []string | audio, image formats | Comma-separated list of allowed MIME types. |
| `MEDIA_UPLOAD_RATE_PER_MINUTE` | int | `10` | Per-user upload rate limit. |
| `MEDIA_UPLOAD_RATE_BURST` | int | `5` | Maximum upload burst capacity. |
| `SMTP_ENABLED` | bool | `false` | When `true`, dispatches real emails over SMTP. When `false`, logs emails safely via slog without exposing secrets. |
| `SMTP_HOST` | string | `smtp.example.com` | SMTP server hostname. |
| `SMTP_PORT` | int | `587` | SMTP server port (e.g. 587 for STARTTLS, 465 for TLS). |
| `SMTP_USERNAME` | string | optional | SMTP authentication username. |
| `SMTP_PASSWORD` | string | optional | SMTP authentication password. |
| `SMTP_FROM` | string | `noreply@echoself.com` | From address header in outgoing emails. |
| `SMTP_USE_TLS` | bool | `true` | Enables STARTTLS/TLS encryption for SMTP connections. |

---

## 3. Local Development Setup

### 3.1 Prerequisites
- Go 1.24+
- Docker and Docker Compose (or local PostgreSQL 16 and Redis 7 installations)

### 3.2 Starting Dependencies
Using Docker Compose:
```bash
# Start PostgreSQL and Redis in the background
docker compose up -d postgres redis
```

### 3.3 Running Migrations
Run the standalone migration runner:
```bash
# Apply all pending migrations
go run ./cmd/migrate up

# Inspect current schema version
go run ./cmd/migrate version
```

### 3.4 Starting the API and Worker
In terminal 1 (API Server):
```bash
go run ./cmd/api
```
In terminal 2 (Worker Daemon):
```bash
go run ./cmd/worker
```

---

## 4. Docker & Multi-Container Deployment

Full multi-container local stack:
```bash
# Build and start all services (PostgreSQL, Redis, Migrate, API, Worker)
docker compose up -d --build

# Inspect running containers and health checks
docker compose ps

# Tail logs
docker compose logs -f api worker
```

To shut down without destroying database volumes:
```bash
docker compose down
```

---

## 5. Health Checks, Observability & Metrics

### 5.1 API Health Probes (`cmd/api`)
- **Liveness:** `GET http://localhost:8080/health`
  - Validates process responsiveness.
  - Returns `200 OK {"status":"ok","service":"echoself-api"}`.
- **Readiness:** `GET http://localhost:8080/ready`
  - Executes bounded (2-second) ping checks on PostgreSQL and Redis.
  - Returns `200 OK {"status":"ready","database":"ok","redis":"ok"}`.
  - Returns `503 Service Unavailable` if PostgreSQL or Redis is unreachable.

### 5.2 Worker Health Probes (`cmd/worker`)
- **Liveness:** `GET http://localhost:8081/health`
  - Returns `200 OK {"status":"ok","service":"echoself-worker"}`.
- **Readiness:** `GET http://localhost:8081/ready`
  - Validates active database and Redis connections.
  - Returns `200 OK {"status":"ready","database":"ok","redis":"ok"}`.

### 5.3 Prometheus Metrics
Both the API (`:8080/metrics`) and Worker (`:8081/metrics`) export Prometheus-formatted metrics:
```
# HELP echoself_http_active_requests Currently executing HTTP requests.
# TYPE echoself_http_active_requests gauge
echoself_http_active_requests 0

# HELP echoself_http_requests_total Total number of HTTP requests processed.
# TYPE echoself_http_requests_total counter
echoself_http_requests_total{method="GET",status="200",path="/api/v1/users/me"} 42

# HELP echoself_jobs_total Total background jobs processed by type and outcome.
# TYPE echoself_jobs_total counter
echoself_jobs_total{type="avatar_generation",status="succeeded"} 15
echoself_jobs_total{type="avatar_generation",status="failed"} 1

# HELP echoself_job_lease_renewal_failures_total Total worker lease renewal failures.
# TYPE echoself_job_lease_renewal_failures_total counter
echoself_job_lease_renewal_failures_total 0

# HELP echoself_job_stale_recoveries_total Total orphaned or stale jobs recovered.
# TYPE echoself_job_stale_recoveries_total counter
echoself_job_stale_recoveries_total 0
```

---

## 6. Database Operations & Backups

### 6.1 Versioned Migrations
Migrations are stored in `migrations/` and embedded directly into the Go binaries using `embed.FS`:
- `000001_initial_schema.up.sql` / `.down.sql`: Users, refresh sessions, email verifications, password resets.
- `000002_consent_and_identity.up.sql` / `.down.sql`: Granular consent records, purpose types, append-only audit logs.
- `000003_media_metadata.up.sql` / `.down.sql`: Media metadata, soft-deletion tracking, storage quotas.
- `000004_background_jobs.up.sql` / `.down.sql`: Distributed jobs, attempt tracking, monotonic fencing, transactional outbox.

### 6.2 Pre-Deployment Migration Command
In production CI/CD pipelines, execute migrations once prior to rolling out new application replicas:
```bash
./migrate up
```

### 6.3 Backup and Restore Procedure
**PostgreSQL Backup:**
```bash
pg_dump -h localhost -U postgres -d echoself -F c -b -v -f echoself_$(date +%Y%m%d_%H%M%S).dump
```
**PostgreSQL Restore:**
```bash
pg_restore -h localhost -U postgres -d echoself -v -c echoself_20261027_120000.dump
```
*Note on Object Storage Consistency:* Restoring a database snapshot alongside Cloudflare R2 object storage may result in orphaned objects or dangling media references. The built-in worker sweeper (`ProcessPendingDeletions`) periodically reconciles pending deletions and cleans up orphaned objects.

---

## 7. Security Hardening & Defenses

1. **Monotonic Fencing on Distributed Jobs:** Every worker mutation (`MarkSucceeded`, `MarkFailed`, `ScheduleRetry`, `RenewLease`) checks `attempt_count` with row-level locks to prevent zombie workers from corrupting job state after lease expiration.
2. **Refresh Token Reuse Detection (RFC 6819):** Refresh tokens are hashed (SHA-256) and rotated upon each refresh. Replaying an already-consumed token triggers immediate revocation of the entire active session family for that user.
3. **Serialized Row-Level Quota Checks:** Storage quotas are enforced atomically using `SELECT ... FOR UPDATE` on the user record within a database transaction, eliminating TOCTOU race conditions under concurrent uploads.
4. **Consent-Gated Processing:** Workers re-verify purpose-specific consent and user active status at execution time immediately prior to calling the processing interface.
5. **No Secret Leakage:** Configuration validation, health endpoints, structured logs, and Prometheus metrics strictly redact credentials, tokens, hashes, and raw passwords.

---

## 8. Verification Commands

Run the full automated test suite with race condition detection:
```bash
# 1. Format code
gofmt -w .

# 2. Run static analysis
go vet ./...

# 3. Execute all unit and component tests with race detector
go test -race -count=1 ./...

# 4. Compile API, Worker, and Migrate binaries
go build -o /dev/null ./cmd/api
go build -o /dev/null ./cmd/worker
go build -o /dev/null ./cmd/migrate
```

---

## 9. Known Limitations & Remaining Risks

1. **Mock AI Processor:** The backend includes a production-ready asynchronous job infrastructure and `MockProcessor`. The real Python/GPU inference engine is designed to connect to the queue or processor abstraction in Phase 8+.
2. **Third-Party Email Delivery:** SMTP delivery is fully implemented with TLS and timeouts; connecting to external providers (e.g. AWS SES, SendGrid, Postmark) requires configuring live `SMTP_*` credentials and DNS records (SPF, DKIM, DMARC) in `.env`.
3. **Cloudflare R2 Credentials:** Real cloud storage requires valid Cloudflare R2 API tokens; the in-memory driver is enabled by default when `R2_ENABLED=false`.
