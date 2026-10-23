-- Migration 000002: Consent records and consent audit trail

CREATE TABLE IF NOT EXISTS consent_records (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    consent_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('granted', 'revoked', 'expired')),
    policy_version VARCHAR(20) NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consent_records_user_id ON consent_records (user_id);
CREATE INDEX IF NOT EXISTS idx_consent_records_user_type ON consent_records (user_id, consent_type);

-- Only one active (granted) record per user per consent_type at any given time
CREATE UNIQUE INDEX IF NOT EXISTS idx_consent_records_user_type_active ON consent_records (user_id, consent_type) WHERE status = 'granted';

CREATE TABLE IF NOT EXISTS consent_audit_logs (
    id UUID PRIMARY KEY,
    consent_id UUID NOT NULL REFERENCES consent_records(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    consent_type VARCHAR(50) NOT NULL,
    policy_version VARCHAR(20) NOT NULL,
    ip_address VARCHAR(45),
    user_agent VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consent_audit_logs_consent_id ON consent_audit_logs (consent_id);
CREATE INDEX IF NOT EXISTS idx_consent_audit_logs_user_id ON consent_audit_logs (user_id);
