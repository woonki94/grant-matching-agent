-- Add abstract embedding column to faculty_publication for publication-level retrieval.
-- Run on Postgres.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE faculty_publication
  ADD COLUMN IF NOT EXISTS abstract_embedding VECTOR(1024);
