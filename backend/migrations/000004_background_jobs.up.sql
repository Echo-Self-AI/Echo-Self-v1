-- Migration 000004: Background jobs and transactional outbox

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'queued',
    input_media_id UUID REFERENCES media(id) ON DELETE SET NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    result JSONB,
    error_code VARCHAR(100),
    error_message TEXT,
    attempt_count INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    locked_by VARCHAR(100),
    locked_until TIMESTAMPTZ,
    retry_at TIMESTAMPTZ,
    cancel_requested_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_user_status ON jobs (user_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_stale_recovery ON jobs (status, locked_until) WHERE status = 'processing';
CREATE INDEX IF NOT EXISTS idx_jobs_retry_scheduled ON jobs (status, retry_at) WHERE status = 'retry_wait';

CREATE TABLE IF NOT EXISTS job_outbox (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_outbox_pending ON job_outbox (status, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_job_outbox_job_id ON job_outbox (job_id);
