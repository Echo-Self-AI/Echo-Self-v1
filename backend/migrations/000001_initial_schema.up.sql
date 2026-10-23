-- EchoSelf initial database schema baseline
-- Matches existing tables and indexes created by GORM AutoMigrate

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    password_hash TEXT NOT NULL,
    display_name VARCHAR(100) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    is_email_verified BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_deleted_at ON users (deleted_at);

CREATE TABLE IF NOT EXISTS refresh_sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    token_hash CHAR(64) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_refresh_sessions_token_hash ON refresh_sessions (token_hash);
CREATE INDEX IF NOT EXISTS idx_refresh_sessions_user_id ON refresh_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_sessions_expires_at ON refresh_sessions (expires_at);

CREATE TABLE IF NOT EXISTS email_verification (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    token_hash CHAR(64) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_email_verification_token_hash ON email_verification (token_hash);
CREATE INDEX IF NOT EXISTS idx_email_verification_user_id ON email_verification (user_id);
CREATE INDEX IF NOT EXISTS idx_email_verification_expires_at ON email_verification (expires_at);

CREATE TABLE IF NOT EXISTS password_resets (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    token_hash CHAR(64) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_password_resets_token_hash ON password_resets (token_hash);
CREATE INDEX IF NOT EXISTS idx_password_resets_user_id ON password_resets (user_id);
CREATE INDEX IF NOT EXISTS idx_password_resets_expires_at ON password_resets (expires_at);
