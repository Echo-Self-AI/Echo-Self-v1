# EchoSelf Private Cloud Storage with Cloudflare R2

This document details the configuration, architecture, security controls, and operational guidelines for EchoSelf's private media storage utilizing **Cloudflare R2**.

---

## 1. Cloudflare R2 Free Tier Allowance & Limits

Cloudflare R2 provides an S3-compatible object storage solution with zero internet egress bandwidth charges. Under the standard free tier:

* **Storage**: 10 GB-month of standard storage.
* **Class A Operations** (mutations: `PutObject`, `ListBuckets`, `ListObjects`, etc.): 1,000,000 requests/month.
* **Class B Operations** (reads: `GetObject`, `HeadObject`, etc.): 10,000,000 requests/month.
* **Internet Egress**: **$0.00** (Zero egress fees across all tiers).
* **Operation Classes Reference**: Verified against [Cloudflare R2 Pricing](https://developers.cloudflare.com/r2/pricing/).

> **Budget Constraint**: EchoSelf operates within a ₹0 ($0) budget during development. To guarantee no unexpected charges, our backend enforces hard application-level quotas and never requires a paid plan or automatic upgrades.

---

## 2. Cloudflare R2 Setup & Configuration

### A. Create an R2 Bucket
1. Sign in to the [Cloudflare Dashboard](https://dash.cloudflare.com/).
2. Navigate to **R2** in the side navigation.
3. Click **Create bucket** and choose a name (e.g., `echoself-private-media`).
4. Keep the bucket **Private** (default). **Do NOT enable Public Access or R2.dev public URLs**.

### B. Generate API Tokens
1. Under **R2** > **Manage R2 API Tokens**, click **Create API Token**.
2. Select **Object Read & Write** permissions scoped specifically to your bucket.
3. Obtain:
   - **Access Key ID**
   - **Secret Access Key**
   - **Account ID** (found on the R2 overview page)

### C. Environment Variables

Configure these settings in your `.env` (or environment manager):

| Variable | Description | Default / Example |
|---|---|---|
| `R2_ENABLED` | Enables Cloudflare R2 driver. When `false`, uses an in-memory storage driver | `false` |
| `R2_ACCOUNT_ID` | Cloudflare account identifier | `your-cloudflare-account-id` |
| `R2_ACCESS_KEY_ID` | R2 Token Access Key ID | `your-r2-access-key-id` |
| `R2_SECRET_ACCESS_KEY` | R2 Token Secret Access Key | `your-r2-secret-access-key` |
| `R2_BUCKET_NAME` | Name of the private bucket | `echoself-private-media` |
| `R2_ENDPOINT` | Custom endpoint (optional; auto-derived from Account ID if blank) | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |
| `R2_REGION` | S3 signing region | `auto` |

### D. Free-Tier Quota & Rate Limit Variables

| Variable | Description | Default |
|---|---|---|
| `MEDIA_MAX_UPLOAD_SIZE_BYTES` | Maximum allowed size per file upload | `10485760` (10 MiB) |
| `MEDIA_USER_QUOTA_BYTES` | Cumulative stored bytes quota per user | `52428800` (50 MiB) |
| `MEDIA_USER_MAX_OBJECTS` | Maximum active stored objects per user | `20` |
| `MEDIA_ALLOWED_TYPES` | Comma-separated list of permitted MIME types | `audio/mpeg,audio/wav,audio/ogg,audio/webm,image/jpeg,image/png,image/webp` |
| `MEDIA_UPLOAD_RATE_PER_MINUTE` | Per-user/IP upload rate limit | `10` req/min |
| `MEDIA_UPLOAD_RATE_BURST` | Token bucket burst capacity | `5` |

---

## 3. Storage Architecture & Security

### A. Private Bucket Isolation
* Direct public bucket access is disabled.
* Stored object keys are **cryptographically random UUIDs** (`media/<user_id>/<media_uuid>`).
* Client-provided filenames, relative paths, and directory traversal sequences (`../`) are never used in storage keys or storage paths.

### B. Authenticated Streaming
* Media retrieval is authenticated via Bearer JWT at `GET /api/v1/media/:id/content`.
* The Go backend streams the file directly from Cloudflare R2 and sends standard defensive response headers:
  * `X-Content-Type-Options: nosniff`
  * `Content-Disposition: inline; filename="sanitized_name"`
  * `Cache-Control: private, no-cache, no-store, must-revalidate`

### C. Content-Type Sniffing (Magic Bytes)
* Client-declared `Content-Type` headers and file extensions are **untrusted**.
* The initial 512 bytes are inspected using MIME detection (`mimetype.Detect`) to verify file signatures.
* Category enforcement ensures:
  * `voice_sample` must be an audio format (`audio/*`).
  * `face_sample` must be an image format (`image/*`).

---

## 4. Database Schema (Migration `000003_media_metadata`)

```sql
CREATE TABLE IF NOT EXISTS media (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(50) NOT NULL,
    storage_key VARCHAR(255) NOT NULL UNIQUE,
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'ready',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);
```

Indexes:
* `idx_media_user_id` on `(user_id)`
* `idx_media_user_id_deleted_at` on `(user_id, deleted_at)`
* `idx_media_storage_key` on `(storage_key)`
* `idx_media_user_category` on `(user_id, category)`
* `idx_media_status` on `(status)`

---

## 5. API Reference

All requests require `Authorization: Bearer <JWT_ACCESS_TOKEN>`.

### 1. Upload Media
`POST /api/v1/media`
* **Content-Type**: `multipart/form-data`
* **Form Fields**:
  * `file`: Binary file contents.
  * `category`: `voice_sample`, `face_sample`, or `general` (default).
* **Response `201 Created`**:
```json
{
  "id": "e30e61ec-d7a8-4444-9351-4c637a7fef1e",
  "user_id": "893c5d7a-1234-4567-89ab-cdef01234567",
  "category": "voice_sample",
  "content_type": "audio/wav",
  "file_size": 245760,
  "original_filename": "sample_voice.wav",
  "status": "ready",
  "created_at": "2026-10-27T10:00:00Z",
  "updated_at": "2026-10-27T10:00:00Z"
}
```

### 2. List User Media
`GET /api/v1/media?category=voice_sample`
* **Response `200 OK`**: JSON array of `MediaResult` objects belonging to the authenticated user.

### 3. Get Media Metadata
`GET /api/v1/media/:id`
* **Response `200 OK`**: JSON metadata of the media file.
* **Response `404 Not Found`**: Returned if the media does not exist or belongs to another user.

### 4. Stream Media Content
`GET /api/v1/media/:id/content`
* **Response `200 OK`**: Binary stream with `Content-Type`, `Content-Length`, and `X-Content-Type-Options: nosniff`.
* **Response `404 Not Found`**: Returned if the media does not exist, is soft-deleted, or belongs to another user.

### 5. Delete Media
`DELETE /api/v1/media/:id`
* **Response `204 No Content`**: Marks record soft-deleted/`deletion_pending` and purges the object from R2.

---

## 6. Consent & AI Processing Boundaries

EchoSelf enforces a strict separation between **storage** and **processing**:
* Storing a voice file (`voice_sample`) does **NOT** grant consent to train or clone voices.
* Storing a face image (`face_sample`) does **NOT** grant consent to synthesize likenesses.
* Before any AI pipeline or background worker initiates processing, it calls `MediaService.VerifyProcessingConsent(ctx, userID, mediaID, purpose)`.
* This check is **fail-closed**: if consent is revoked, expired, or version-mismatched, AI processing is immediately refused.

---

## 7. Account Deletion & Failure Handling

* When an account is deleted via `DELETE /api/v1/users/me`, `UserRepository.DeleteAccount` executes an atomic database transaction that:
  1. Soft-deletes the user record.
  2. Revokes active refresh sessions and invalidates tokens.
  3. Soft-deletes user media records and transitions their status to `deletion_pending`.
* Storage deletion is retry-safe: if R2 temporarily rejects a deletion call, the record remains in `deletion_pending` state so a background cleanup job can retry safely.
* Compensating transactions: If object upload to R2 succeeds but metadata insertion into PostgreSQL fails, the uploaded object is immediately purged from R2 so no unindexed orphan files consume storage quota.

---

## 8. Provider-Side Usage & Cost Safeguards

1. **Dashboard Monitoring**:
   * Regularly verify total storage and Class A/B operation counts in the Cloudflare Dashboard under **R2** > **Overview**.
2. **Notifications & Alerts**:
   * Cloudflare provides billing notifications. Configure an alert threshold before reaching 10 GB or 1,000,000 Class A operations.
3. **Budget Safety**:
   * Do not attach a payment method or enable auto-scaling paid add-ons during development.
