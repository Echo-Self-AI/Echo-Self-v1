-- Rollback for baseline initial schema

DROP TABLE IF EXISTS password_resets;
DROP TABLE IF EXISTS email_verification;
DROP TABLE IF EXISTS refresh_sessions;
DROP TABLE IF EXISTS users;
