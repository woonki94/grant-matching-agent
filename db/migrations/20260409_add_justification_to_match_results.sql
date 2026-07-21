-- Add justification column to match_results for cached per-faculty×grant justification text.
-- Once populated on first request, eliminates per-request Sonnet LLM calls in the 1:1 pipeline.
ALTER TABLE match_results
  ADD COLUMN IF NOT EXISTS justification TEXT;
