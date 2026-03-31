-- Drop legacy free-form reason column from one-to-one match results.
-- Run on Postgres.
ALTER TABLE match_results
  DROP COLUMN IF EXISTS reason;

