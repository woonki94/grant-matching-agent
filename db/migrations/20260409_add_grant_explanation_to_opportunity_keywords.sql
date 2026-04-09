-- Add grant_explanation column to opportunity_keywords for pre-generated LLM explanations.
-- Once populated by the generate_grant_explanations script, this eliminates per-request
-- Haiku LLM calls in the 1:1 matching justification pipeline.
ALTER TABLE opportunity_keywords
  ADD COLUMN IF NOT EXISTS grant_explanation TEXT;
