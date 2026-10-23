-- Rollback migration 000002: Consent records and consent audit trail

DROP TABLE IF EXISTS consent_audit_logs;
DROP TABLE IF EXISTS consent_records;
