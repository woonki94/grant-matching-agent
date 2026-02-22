from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import boto3

from config import settings
from db.db_conn import SessionLocal
from db.models.faculty import Faculty, FacultyAdditionalInfo
from services.faculty.profile_parser import parse_profile
from services.extract_content import short_hash
from utils.content_extractor import fetch_and_extract_one
from dao.faculty_dao import FacultyDAO
from mappers.page_to_faculty import map_faculty_profile_to_dto


logger = logging.getLogger(__name__)

_PERSONAL_SUBDIR = "faculties_additional_links"


def _s3_client():
    session = (
        boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
        if settings.aws_profile
        else boto3.Session(region_name=settings.aws_region)
    )
    return session.client("s3")


def _upload_text_to_s3(text: str, *, key: str) -> None:
    bucket = (settings.extracted_content_bucket or "").strip()
    if not bucket:
        raise RuntimeError("extracted_content_bucket is required for personal website extraction")
    s3 = _s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8", errors="ignore"),
        ContentType="text/plain; charset=utf-8",
    )


def _build_personal_key(item_id: int, url: str) -> str:
    prefix = (settings.extracted_content_prefix_faculty or "").strip().strip("/")
    fname = f"{item_id}__{short_hash(url)}.txt"
    if prefix:
        return f"{prefix}/{_PERSONAL_SUBDIR}/{fname}"
    return f"{_PERSONAL_SUBDIR}/{fname}"


def _enrich_from_osu_profile(faculty_id: int, osu_webpage: str) -> None:
    if not osu_webpage:
        return
    try:
        profile = parse_profile(osu_webpage)
    except Exception:
        logger.exception("Failed to parse OSU profile", extra={"faculty_id": faculty_id})
        return

    with SessionLocal() as sess:
        fac = sess.get(Faculty, faculty_id)
        if not fac:
            return

        if not fac.name and profile.get("name"):
            fac.name = profile.get("name")
        if profile.get("position"):
            fac.position = profile.get("position")
        if profile.get("organization"):
            fac.organization = profile.get("organization")
        if profile.get("address"):
            fac.address = profile.get("address")
        if profile.get("biography"):
            fac.biography = profile.get("biography")
        if profile.get("expertise"):
            fac.expertise = profile.get("expertise")
        if profile.get("degrees"):
            fac.degrees = profile.get("degrees")

        fac.profile_last_refreshed_at = datetime.now(timezone.utc)
        # Reuse existing mapper to extract additional links and upsert them
        try:
            dto = map_faculty_profile_to_dto(profile)
            if dto.additional_info:
                dao = FacultyDAO(sess)
                dao.upsert_additional_info(faculty_id, dto.additional_info)
        except Exception:
            logger.exception("Failed to upsert OSU additional links", extra={"faculty_id": faculty_id})

        sess.commit()


def _enrich_from_personal_website(faculty_id: int, personal_website: str) -> None:
    if not personal_website:
        return
    with SessionLocal() as sess:
        existing = (
            sess.query(FacultyAdditionalInfo)
            .filter(
                FacultyAdditionalInfo.faculty_id == faculty_id,
                FacultyAdditionalInfo.additional_info_url == personal_website,
            )
            .one_or_none()
        )
        if existing:
            return

        item = FacultyAdditionalInfo(
            faculty_id=faculty_id,
            additional_info_url=personal_website,
            extract_status="pending",
        )
        sess.add(item)
        sess.flush()

        extracted_at = datetime.now(timezone.utc)
        try:
            result = fetch_and_extract_one(
                personal_website,
                user_agent=settings.scraper_user_agent,
            )
            text = result.get("text") or ""
            if not text.strip():
                err = result.get("error") or "no_text"
                item.extract_status = "failed"
                item.detected_type = "personal_webpage"
                item.extract_error = err
                item.extracted_at = extracted_at
                sess.commit()
                return

            key = _build_personal_key(item.id, personal_website)
            _upload_text_to_s3(text, key=key)

            item.content_path = key
            item.detected_type = "personal_webpage"
            item.content_char_count = len(text)
            item.extracted_at = extracted_at
            item.extract_status = "success"
            item.extract_error = None
            sess.commit()
        except Exception as exc:
            item.extract_status = "failed"
            item.detected_type = "personal_webpage"
            item.extract_error = str(exc)[:5000]
            item.extracted_at = extracted_at
            sess.commit()
            logger.exception("Failed to extract personal website", extra={"faculty_id": faculty_id})


def enrich_new_faculty(
    *,
    email: str,
    faculty_id: int,
    osu_webpage: Optional[str],
    personal_website: Optional[str],
) -> None:
    """
    Enrich a newly inserted faculty record from available profile sources.

    Publication ingestion (from uploaded CV PDF) is handled separately via
    utils/publication_extractor.py — not here.
    """
    if not faculty_id:
        return

    if osu_webpage:
        _enrich_from_osu_profile(faculty_id, osu_webpage)
    if personal_website:
        _enrich_from_personal_website(faculty_id, personal_website)

    logger.info(
        "Profile enrichment completed",
        extra={
            "email": email,
            "osu_used": bool(osu_webpage),
            "personal_used": bool(personal_website),
        },
    )
