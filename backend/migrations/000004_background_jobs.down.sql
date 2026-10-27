-- Migration 000004 down: Drop jobs and outbox tables

DROP TABLE IF EXISTS job_outbox;
DROP TABLE IF EXISTS jobs;
