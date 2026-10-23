# EchoSelf Data Deletion & Account Lifecycle Policy

This document defines the data retention, lifecycle, and account deletion policies for the EchoSelf platform.

---

## 1. Overview

EchoSelf guarantees user privacy and data ownership. When a user requests the deletion of their account via `DELETE /api/v1/users/me`, the system executes an atomic lifecycle transition to immediately revoke access, invalidate outstanding tokens, and mark user records for permanent purge.

---

## 2. Immediate Deletion Actions (Synchronous)

When an authenticated user requests account deletion:

1. **User Record Soft-Deletion**:
   - The user's record in `users` has its `deleted_at` timestamp set to the current time (`gorm.DeletedAt`).
   - Normal queries filter out soft-deleted users.
   - Any subsequent requests with existing JWT access tokens will fail with `401 Unauthorized` (`USER_NOT_FOUND`).
2. **Session Revocation**:
   - All active refresh sessions (`refresh_sessions`) belonging to the user are immediately revoked (`revoked_at = NOW()`).
   - Any attempt to use outstanding refresh tokens fails immediately with `401 Unauthorized` (`INVALID_REFRESH_TOKEN`).
3. **Token Invalidation**:
   - All pending email verification records (`email_verification`) are invalidated (`used_at = NOW()`).
   - All pending password reset records (`password_resets`) are invalidated (`used_at = NOW()`).
4. **Transaction Atomicity**:
   - All of the above steps occur within a single database transaction (`UserRepository.DeleteAccount`). If any step fails, the entire transaction is aborted and no partial deletion occurs.

---

## 3. Associated Data Lifecycle (Cascade Policy)

As EchoSelf features expand, the following rules govern user-associated resources:

| Resource Type | Storage Location | Deletion Policy | Retention Period |
|---|---|---|---|
| User Profile | PostgreSQL (`users`) | Soft-deleted immediately; hard-purged after retention window | 30 days |
| Refresh Sessions | PostgreSQL (`refresh_sessions`) | Revoked immediately | Purged after 30 days |
| Verification / Reset Tokens | PostgreSQL (`email_verification`, `password_resets`) | Marked used/invalidated immediately | Purged after 7 days |
| Voice & Face Identity Profiles | PostgreSQL & Object Storage (MinIO/S3) | Flagged for deletion; media objects deleted via background worker | Immediate revocation; purge within 24 hours |
| Private Media Storage | Object Storage (MinIO/S3) | Media files purged upon confirmation | Immediate / 24 hours |
| EchoBot Conversations | PostgreSQL | Soft-deleted; purged after retention window | 30 days |
| Audit Logs | PostgreSQL (`audit_logs`) | Anonymized (user ID masked) for compliance; secrets/raw media never logged | 90 days for security auditing |

---

## 4. Cancellation & Account Recovery

- Within the 30-day soft-deletion window, users may contact support to request account reinstatement (subject to identity re-verification).
- After 30 days, a scheduled cleanup worker permanently hard-deletes the soft-deleted records from the database (`db.Unscoped().Delete(...)`).

---

## 5. Security & Privacy Guarantees

- **No Secret Logging**: Raw passwords, access tokens, refresh tokens, verification tokens, or reset tokens are never logged or exported during the deletion process.
- **Resource Ownership**: Users can only trigger deletion for their own account identified by the verified JWT `sub` claim. Cross-account deletion requests are strictly prohibited and return `401 Unauthorized` or `403 Forbidden`.
