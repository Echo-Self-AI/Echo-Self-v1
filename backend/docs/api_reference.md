# EchoSelf Backend API Reference

This document provides the definitive specification of all registered HTTP endpoints in the EchoSelf Go backend API.

---

## 1. Operational & Observability Endpoints

### 1.1 Liveness Probe
- **Method / Path:** `GET /health`
- **Auth:** None (Public)
- **Response (200 OK):**
  ```json
  {
    "status": "ok",
    "service": "echoself-api"
  }
  ```

### 1.2 Readiness Probe
- **Method / Path:** `GET /ready`
- **Auth:** None (Public)
- **Response (200 OK):**
  ```json
  {
    "status": "ready",
    "database": "ok",
    "redis": "ok"
  }
  ```
- **Error (503 Service Unavailable):**
  ```json
  {
    "error": {
      "code": "DATABASE_UNAVAILABLE" | "REDIS_UNAVAILABLE",
      "message": "Database is unavailable" | "Redis is unavailable"
    }
  }
  ```

### 1.3 Prometheus Metrics
- **Method / Path:** `GET /metrics`
- **Auth:** None (Protected at edge / VPC boundary in production)
- **Format:** `text/plain; version=0.0.4; charset=utf-8`
- **Metrics Tracked:**
  - `echoself_http_active_requests` (gauge)
  - `echoself_http_requests_total{method, status, path}` (counter)
  - `echoself_http_request_duration_seconds_{sum,count}{method, status, path}` (summary)
  - `echoself_jobs_total{type, status}` (counter)
  - `echoself_job_duration_seconds_{sum,count}{type}` (summary)
  - `echoself_job_retries_total{type}` (counter)
  - `echoself_job_lease_renewal_failures_total` (counter)
  - `echoself_job_stale_recoveries_total` (counter)
  - `echoself_outbox_dispatches_total{status}` (counter)

---

## 2. Authentication (`/api/v1/auth`)

All unauthenticated endpoints are rate-limited. Passwords must be between 8 and 72 characters.

### 2.1 Register User
- **Method / Path:** `POST /api/v1/auth/register`
- **Request Body:**
  ```json
  {
    "email": "user@example.com",
    "password": "Password123!",
    "display_name": "Jane Doe"
  }
  ```
- **Response (201 Created):**
  ```json
  {
    "user": {
      "id": "c1f7b029-...",
      "email": "user@example.com",
      "display_name": "Jane Doe",
      "is_email_verified": false
    },
    "tokens": {
      "access_token": "eyJhbGciOiJIUzI1NiIsIn...",
      "refresh_token": "64_hex_chars..."
    }
  }
  ```

### 2.2 Login
- **Method / Path:** `POST /api/v1/auth/login`
- **Request Body:**
  ```json
  {
    "email": "user@example.com",
    "password": "Password123!"
  }
  ```
- **Response (200 OK):** Same structure as Register.

### 2.3 Refresh Session
- **Method / Path:** `POST /api/v1/auth/refresh`
- **Request Body:**
  ```json
  {
    "refresh_token": "64_hex_chars..."
  }
  ```
- **Response (200 OK):** New `access_token` and rotated single-use `refresh_token`. Replay of consumed tokens revokes all sessions in the user family.

### 2.4 Logout
- **Method / Path:** `POST /api/v1/auth/logout`
- **Request Body:**
  ```json
  {
    "refresh_token": "64_hex_chars..."
  }
  ```
- **Response (204 No Content)**

### 2.5 Request Email Verification
- **Method / Path:** `POST /api/v1/auth/verify-email/request`
- **Auth:** `Bearer <access_token>`
- **Response (201 Created):**
  ```json
  {
    "message": "verification token created"
  }
  ```

### 2.6 Confirm Email Verification
- **Method / Path:** `POST /api/v1/auth/verify-email/confirm`
- **Request Body:**
  ```json
  {
    "token": "64_character_hex_token"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "message": "email verified successfully"
  }
  ```

### 2.7 Request Password Reset
- **Method / Path:** `POST /api/v1/auth/password-reset/request`
- **Request Body:**
  ```json
  {
    "email": "user@example.com"
  }
  ```
- **Response (200 OK):** Constant response to prevent user enumeration.

### 2.8 Confirm Password Reset
- **Method / Path:** `POST /api/v1/auth/password-reset/confirm`
- **Request Body:**
  ```json
  {
    "token": "64_character_hex_token",
    "new_password": "NewSecurePassword123!"
  }
  ```
- **Response (200 OK):** Password updated, outstanding tokens invalidated.

---

## 3. Users (`/api/v1/users`)

All endpoints require `Authorization: Bearer <access_token>`.

### 3.1 Get Profile
- **Method / Path:** `GET /api/v1/users/me`
- **Response (200 OK):** User profile details.

### 3.2 Update Profile
- **Method / Path:** `PATCH /api/v1/users/me`
- **Request Body:**
  ```json
  {
    "display_name": "Updated Name"
  }
  ```
- **Response (200 OK):** Updated user profile.

### 3.3 Delete Account
- **Method / Path:** `DELETE /api/v1/users/me`
- **Response (204 No Content):** Initiates soft-deletion with 30-day grace period, cancels queued/running background jobs, and marks media for storage cleanup.

---

## 4. Consents (`/api/v1/consents`)

All endpoints require `Authorization: Bearer <access_token>`.

### 4.1 Grant Purpose-Specific Consent
- **Method / Path:** `POST /api/v1/consents`
- **Request Body:**
  ```json
  {
    "media_id": "uuid",
    "purpose": "avatar_synthesis",
    "expires_at": "2026-12-31T23:59:59Z"
  }
  ```
- **Response (201 Created)**

### 4.2 List Consents
- **Method / Path:** `GET /api/v1/consents`
- **Response (200 OK):** Array of active consent records.

### 4.3 Revoke Consent
- **Method / Path:** `POST /api/v1/consents/:id/revoke`
- **Response (200 OK):** Immediately revokes consent; future AI operations are blocked.

### 4.4 Get Consent Audit Logs
- **Method / Path:** `GET /api/v1/consents/:id/audit`
- **Response (200 OK):** Immutable history of consent transitions (grant, revoke, expire).

---

## 5. Private Media (`/api/v1/media`)

All endpoints require `Authorization: Bearer <access_token>`.

### 5.1 Upload Media
- **Method / Path:** `POST /api/v1/media/upload`
- **Content-Type:** `multipart/form-data` (`file` form field)
- **Validation:** Enforces MIME sniff, user quota (`50MB`), max upload size (`10MB`), max object count (`20`).
- **Response (201 Created):** Media object metadata.

### 5.2 List Media
- **Method / Path:** `GET /api/v1/media`
- **Response (200 OK):** Array of user-owned media objects.

### 5.3 Stream Media
- **Method / Path:** `GET /api/v1/media/:id/stream`
- **Headers:** Supports standard HTTP `Range` requests for partial byte streaming (`200 OK` or `206 Partial Content`).

### 5.4 Delete Media
- **Method / Path:** `DELETE /api/v1/media/:id`
- **Response (204 No Content):** Metadata deleted; storage object queued for asynchronous cleanup.

---

## 6. Background Jobs (`/api/v1/jobs`)

All endpoints require `Authorization: Bearer <access_token>`.

### 6.1 Submit Job
- **Method / Path:** `POST /api/v1/jobs`
- **Request Body:**
  ```json
  {
    "job_type": "avatar_generation",
    "media_id": "optional-uuid",
    "payload": {
      "resolution": "1080p"
    }
  }
  ```
- **Response (202 Accepted):** Job created and queued in outbox.

### 6.2 List Jobs
- **Method / Path:** `GET /api/v1/jobs?status=processing&limit=20&offset=0`
- **Response (200 OK):** Paginated list of jobs.

### 6.3 Get Job Status
- **Method / Path:** `GET /api/v1/jobs/:id`
- **Response (200 OK):** Current state, progress, attempt count, output, or error details.

### 6.4 Cancel Job
- **Method / Path:** `POST /api/v1/jobs/:id/cancel`
- **Response (200 OK):** Sets status to `cancel_requested` or `cancelled`. Fenced workers abort active work immediately.
