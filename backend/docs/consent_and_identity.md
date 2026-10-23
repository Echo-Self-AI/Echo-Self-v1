# Consent & Identity Management Guide

EchoSelf is a privacy-first platform that establishes an explicit, purpose-specific consent architecture before sensitive AI generation workflows (voice cloning, face synthesis, video avatars) are permitted.

---

## 1. Core Principles

1. **Authentication is Not Consent**: Merely logging in or creating an account does not authorize voice or face cloning.
2. **Email Verification is Not Identity Proof**: Having verified access to an email address proves mailbox control, not real-world identity or likeness ownership.
3. **Consent is Purpose-Specific**: Consents are granted individually for well-defined purposes, never bundled into a generic blanket opt-in.
4. **Immediate Revocation**: Revoking consent takes effect immediately and causes any future authorization checks to fail closed.
5. **Fail-Closed Security**: If a database error or network issue occurs while evaluating consent, the protected operation is denied by default.
6. **Immutable Audit Trail**: All consent transitions (`granted`, `revoked`, `expired`) create append-only audit records containing IP addresses, user agents, and timestamps without logging secrets or raw personal media.

---

## 2. Data Model & States

### Tables

1. **`consent_records`**:
   - `id`: UUID (Primary Key)
   - `user_id`: UUID (Foreign Key to `users.id`, cascaded on deletion)
   - `consent_type`: String (Purpose allowlist)
   - `status`: String (`granted`, `revoked`, `expired`)
   - `policy_version`: String (e.g. `v1.0`)
   - `granted_at`: TIMESTAMPTZ
   - `revoked_at`: TIMESTAMPTZ (null while active)
   - `expires_at`: TIMESTAMPTZ (optional TTL)
   - `created_at`, `updated_at`: TIMESTAMPTZ
   - **Database Constraint**: Partial unique index on `(user_id, consent_type) WHERE status = 'granted'` guarantees at the database level that a user cannot have more than one active consent for a given purpose.

2. **`consent_audit_logs`**:
   - `id`: UUID (Primary Key)
   - `consent_id`: UUID (Foreign Key to `consent_records.id`)
   - `user_id`: UUID (Foreign Key to `users.id`)
   - `action`: String (`granted`, `revoked`, `expired`)
   - `consent_type`: String
   - `policy_version`: String
   - `ip_address`: String (Client IP)
   - `user_agent`: String (Client browser/app identifier)
   - `created_at`: TIMESTAMPTZ

### State Transitions

```
[None / Revoked] ---> (Grant Consent) ---> [Granted]
                                              |
                     (Revoke Consent) --------+---> [Revoked]
                                              |
                     (TTL Expires) -----------+---> [Expired]
```

---

## 3. Supported Consent Purposes

The backend enforces an explicit allowlist:

| Purpose Identifier | Description |
|---|---|
| `voice_data_processing` | Permission to analyze uploaded audio samples for acoustic properties |
| `voice_cloning` | Permission to synthesize new speech using the user's voice model |
| `face_likeness_processing` | Permission to extract facial landmarks and training embeddings from uploaded photos |
| `face_synthesis` | Permission to generate video avatars and synthetic face frames |
| `ai_personalization` | Permission to analyze conversation transcripts for personalization |

---

## 4. REST API Endpoints

All endpoints require `Authorization: Bearer <access_token>`.

### 4.1 Grant Consent
- **Method**: `POST`
- **Path**: `/api/v1/consents`
- **Request Body**:
  ```json
  {
    "consent_type": "voice_cloning",
    "policy_version": "v1.0",
    "expires_at": "2027-01-01T00:00:00Z"
  }
  ```
- **Response**: `201 Created`
  ```json
  {
    "id": "e4b7b252-0941-4b1f-9988-75cbe27adce5",
    "user_id": "713c9c90-95e2-4118-8f52-db2d129759ad",
    "consent_type": "voice_cloning",
    "status": "granted",
    "policy_version": "v1.0",
    "granted_at": "2026-10-23T12:00:00Z",
    "created_at": "2026-10-23T12:00:00Z",
    "updated_at": "2026-10-23T12:00:00Z"
  }
  ```

### 4.2 List Consents
- **Method**: `GET`
- **Path**: `/api/v1/consents`
- **Response**: `200 OK` (Array of consent records belonging strictly to the authenticated user).

### 4.3 Get Consent by ID
- **Method**: `GET`
- **Path**: `/api/v1/consents/:id`
- **Response**: `200 OK` (or `404 Not Found` if the record does not exist or belongs to another user).

### 4.4 Revoke Consent
- **Method**: `POST`
- **Path**: `/api/v1/consents/:id/revoke`
- **Response**: `200 OK`
  ```json
  {
    "message": "consent revoked successfully"
  }
  ```

### 4.5 Retrieve Audit Trail
- **Method**: `GET`
- **Path**: `/api/v1/consents/:id/audit`
- **Response**: `200 OK` (Array of audit log entries tracking the grant and revocation timeline).

---

## 5. How Future AI Services Must Check Consent

Any future internal service or background worker (e.g. voice cloning queue, Python FastAPI pipeline) must invoke the reusable authorization check before commencing processing:

```go
err := consentService.CheckConsent(ctx, services.ConsentCheckRequest{
    UserID:          userID,
    ConsentType:     services.PurposeVoiceCloning,
    RequiredVersion: "v1.0",
})
if err != nil {
    // ABORT: Operation is NOT authorized.
    // Handles missing, revoked, expired, or outdated consent.
}
```

---

## 6. Real-World Identity Verification Boundary

In EchoSelf:
- **Authentication** verifies the user holds a valid session token.
- **Email verification** confirms control over an email mailbox.
- **Consent** confirms the user explicitly authorized a specific computational purpose.
- **Identity / Likeness Verification** represents the future boundary for real-world legal and biometric identity verification (e.g. government ID checks or liveness face-matching).
- An extension interface `IdentityVerifier` is provided in `internal/services/identity_verification.go`.

---

## 7. Database Migrations

- Migrations are versioned under `migrations/`:
  - `000001_initial_schema`: Initial users and auth baseline.
  - `000002_consent_and_identity`: Consent records and audit logs.
- To apply migrations:
  ```bash
  go run ./cmd/migrate up
  ```
- To check version:
  ```bash
  go run ./cmd/migrate version
  ```
