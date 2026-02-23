-- Add broad/specific grant category columns on opportunity keywords.
-- Keep model changes separate; this migration is DB-only.
ALTER TABLE opportunity_keywords
  ADD COLUMN IF NOT EXISTS broad_category TEXT;

ALTER TABLE opportunity_keywords
  ADD COLUMN IF NOT EXISTS specific_categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
