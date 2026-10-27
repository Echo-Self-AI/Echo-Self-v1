-- Migration 000003: Media metadata table for Cloudflare R2 private storage

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

CREATE INDEX IF NOT EXISTS idx_media_user_id ON media (user_id);
CREATE INDEX IF NOT EXISTS idx_media_user_id_deleted_at ON media (user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_media_storage_key ON media (storage_key);
CREATE INDEX IF NOT EXISTS idx_media_user_category ON media (user_id, category);
CREATE INDEX IF NOT EXISTS idx_media_status ON media (status);
