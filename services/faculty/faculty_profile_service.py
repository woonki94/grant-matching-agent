from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3

from config import settings
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty, FacultyAdditionalInfo, FacultyPublication


class FacultyProfileService:
    """Read-only service that shapes faculty profile data for frontend consumption."""

    def __init__(self, *, session_factory=SessionLocal, presigned_ttl_seconds: int = 3600):
        self.session_factory = session_factory
        self.presigned_ttl_seconds = max(60, int(presigned_ttl_seconds or 3600))
        self._s3_client = None

    def get_faculty_profile(
        self,
        *,
        faculty_id: Optional[int] = None,
        email: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_email = (str(email or "").strip().lower() or None)
        use_faculty_id = int(faculty_id) if faculty_id else None

        if not use_faculty_id and not normalized_email:
            return None

        with self.session_factory() as sess:
            dao = FacultyDAO(sess)
            fac = (
                dao.get_with_relations_by_id(use_faculty_id)
                if use_faculty_id
                else dao.get_with_relations_by_email(normalized_email)
            )
            if not fac:
                return None
            return self._serialize_faculty(fac)

    def list_faculty_profiles(self, *, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.session_factory() as sess:
            dao = FacultyDAO(sess)
            rows = dao.list_with_relations(limit=limit, offset=offset)
            return [self._serialize_faculty(fac) for fac in rows]

    def _serialize_faculty(self, fac: Faculty) -> Dict[str, Any]:
        publications = sorted(
            list(fac.publications or []),
            key=lambda p: ((p.year or 0), (p.title or "")),
            reverse=True,
        )
        publication_titles = [self._serialize_publication(p) for p in publications]
        publication_years = [int(p.year) for p in publications if p.year is not None]
        publication_fetched_upto_year = max(publication_years) if publication_years else None

        attached_files = [self._serialize_attached_file(row) for row in (fac.additional_info or [])]
        organizations = self._normalize_organizations(fac)
        all_keywords = ((fac.keyword and fac.keyword.keywords) or {}) or {}
        profile_refreshed_at = self._isoformat_or_none(getattr(fac, "profile_last_refreshed_at", None))

        return {
            "faculty_id": int(fac.faculty_id),
            "name": fac.name,
            "email": fac.email,
            "position": fac.position,
            "organizations": organizations,
            "basic_info": {
                "faculty_name": fac.name,
                "email": fac.email,
                "position": fac.position,
                "organizations": organizations,
            },
            "data_from": {
                "publication_titles": publication_titles,
                "publication_fetched_upto_year": publication_fetched_upto_year,
                "publication_fetched_upto_at": profile_refreshed_at,
                "info_source_url": fac.source_url,
                "attached_files": attached_files,
            },
            "all_keywords": all_keywords,
            "keyword_source": (fac.keyword.source if fac.keyword else None),
        }

    @staticmethod
    def _normalize_organizations(fac: Faculty) -> List[str]:
        orgs = list(getattr(fac, "organizations", None) or [])
        if orgs:
            return [str(x).strip() for x in orgs if str(x).strip()]

        one_line = str(getattr(fac, "organization", "") or "").strip()
        if not one_line:
            return []

        parts = [x.strip() for x in one_line.split("|")]
        cleaned = [x for x in parts if x]
        return cleaned or [one_line]

    def _serialize_attached_file(self, row: FacultyAdditionalInfo) -> Dict[str, Any]:
        s3_uri, presigned_url = self._build_download_paths(getattr(row, "content_path", None))
        downloadable_path = presigned_url or s3_uri

        return {
            "additional_info_id": int(row.id),
            "source_url": row.additional_info_url,
            "content_path": row.content_path,
            "downloadable_path": downloadable_path,
            "s3_uri": s3_uri,
            "presigned_url": presigned_url,
            "extract_status": row.extract_status,
            "detected_type": row.detected_type,
            "content_char_count": row.content_char_count,
            "extracted_at": self._isoformat_or_none(row.extracted_at),
        }

    @staticmethod
    def _serialize_publication(p: FacultyPublication) -> Dict[str, Any]:
        return {
            "title": p.title,
            "year": p.year,
        }

    @staticmethod
    def _isoformat_or_none(dt_value: Optional[datetime]) -> Optional[str]:
        if dt_value is None:
            return None
        if isinstance(dt_value, datetime):
            return dt_value.isoformat()
        return str(dt_value)

    def _build_download_paths(self, content_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        parsed = self._parse_bucket_key(content_path)
        if not parsed:
            return None, None

        bucket, key = parsed
        s3_uri = f"s3://{bucket}/{key}"

        try:
            s3 = self._get_s3_client()
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=self.presigned_ttl_seconds,
            )
            return s3_uri, presigned_url
        except Exception:
            return s3_uri, None

    def _parse_bucket_key(self, content_path: Optional[str]) -> Optional[Tuple[str, str]]:
        cp = str(content_path or "").strip()
        if not cp:
            return None

        if cp.startswith("s3://"):
            rest = cp[5:]
            parts = rest.split("/", 1)
            if len(parts) != 2:
                return None
            bucket = parts[0].strip()
            key = parts[1].lstrip("/")
            if not bucket or not key:
                return None
            return bucket, key

        bucket = (settings.extracted_content_bucket or "").strip()
        if not bucket:
            return None
        return bucket, cp.lstrip("/")

    def _get_s3_client(self):
        if self._s3_client is not None:
            return self._s3_client

        session = (
            boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
            if settings.aws_profile
            else boto3.Session(region_name=settings.aws_region)
        )
        self._s3_client = session.client("s3")
        return self._s3_client

