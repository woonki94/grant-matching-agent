-- Add profile_last_refreshed_at to faculty table
-- Run on Postgres
ALTER TABLE faculty
  ADD COLUMN IF NOT EXISTS profile_last_refreshed_at TIMESTAMPTZ;
