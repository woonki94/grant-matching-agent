-- Add chunk_index to existing extracted-content tables so one source can have many chunks.

ALTER TABLE faculty_additional_info
  ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;

ALTER TABLE opportunity_additional_info
  ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;

ALTER TABLE opportunity_attachment
  ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;

-- Rebuild uniqueness constraints to include chunk_index.
ALTER TABLE faculty_additional_info
  DROP CONSTRAINT IF EXISTS ux_faculty_additional_info_opp_url;
ALTER TABLE faculty_additional_info
  ADD CONSTRAINT ux_faculty_additional_info_opp_url
  UNIQUE (faculty_id, additional_info_url, chunk_index);

ALTER TABLE opportunity_additional_info
  DROP CONSTRAINT IF EXISTS ux_opportunity_additional_info_opp_url;
ALTER TABLE opportunity_additional_info
  ADD CONSTRAINT ux_opportunity_additional_info_opp_url
  UNIQUE (opportunity_id, additional_info_url, chunk_index);

ALTER TABLE opportunity_attachment
  DROP CONSTRAINT IF EXISTS ux_attachment_opportunity_file;
ALTER TABLE opportunity_attachment
  ADD CONSTRAINT ux_attachment_opportunity_file
  UNIQUE (opportunity_id, file_name, chunk_index);

ALTER TABLE opportunity_attachment
  DROP CONSTRAINT IF EXISTS ux_attachment_opportunity_download;
ALTER TABLE opportunity_attachment
  ADD CONSTRAINT ux_attachment_opportunity_download
  UNIQUE (opportunity_id, file_download_path, chunk_index);

CREATE INDEX IF NOT EXISTS ix_faculty_additional_info_chunk_index
  ON faculty_additional_info (chunk_index);

CREATE INDEX IF NOT EXISTS ix_additional_info_chunk_index
  ON opportunity_additional_info (chunk_index);

CREATE INDEX IF NOT EXISTS ix_attachment_chunk_index
  ON opportunity_attachment (chunk_index);
