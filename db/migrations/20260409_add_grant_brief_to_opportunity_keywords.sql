-- Add grant_brief column to opportunity_keywords for cached 3-5 sentence grant summary.
-- Used by GroupJustificationEngine to skip the grant_brief Haiku LLM call.
ALTER TABLE opportunity_keywords
  ADD COLUMN IF NOT EXISTS grant_brief TEXT;
