-- Add embedding vector columns for extracted-content tables.
-- These tables are used for content we want to embed:
--   - faculty_additional_info
--   - opportunity_additional_info
--   - opportunity_attachment

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE faculty_additional_info
  ADD COLUMN IF NOT EXISTS content_embedding VECTOR(1024);

ALTER TABLE opportunity_additional_info
  ADD COLUMN IF NOT EXISTS content_embedding VECTOR(1024);

ALTER TABLE opportunity_attachment
  ADD COLUMN IF NOT EXISTS content_embedding VECTOR(1024);
