from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import settings
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from mappers.page_to_faculty import map_faculty_profile_to_dto
from services.faculty.enrich_profile import enrich_faculty_publications_from_cv, enrich_new_faculty
from services.faculty.profile_parser import parse_profile

logger = logging.getLogger(__name__)

_STALE_DAYS = 30
_MAX_SCRAPE_WORKERS = 5


class FacultyContextAgent:
    def __init__(self, *, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _normalize_emails(emails: List[str]) -> List[str]:
        out: List[str] = []
        for e in emails or []:
            x = str(e or "").strip().lower()
            if x and x not in out:
                out.append(x)
        return out

    # ──────────────────────────────────────────────────────────────────
    # DEPRECATED — use resolve_and_ingest_faculties() instead.
    # This method only checks the DB and gives up on missing faculty.
    # It is retained for reference but is no longer called by the
    # orchestrator. Do not add new callers.
    # ──────────────────────────────────────────────────────────────────

    def resolve_faculties(self, *, emails: List[str]) -> Dict[str, Any]:
        self._call("FacultyContextAgent.resolve_faculties")
        normalized = self._normalize_emails(emails)
        faculty_ids: List[int] = []
        missing: List[str] = []
        try:
            with self.session_factory() as sess:
                dao = FacultyDAO(sess)
                for email in normalized:
                    fid = dao.get_faculty_id_by_email(email)
                    if fid is None:
                        missing.append(email)
                    else:
                        faculty_ids.append(int(fid))
        except Exception as e:
            return {
                "emails": normalized,
                "faculty_ids": [],
                "missing_emails": normalized,
                "all_in_db": False,
                "error": f"{type(e).__name__}: {e}",
            }

        return {
            "emails": normalized,
            "faculty_ids": faculty_ids,
            "missing_emails": missing,
            "all_in_db": len(missing) == 0 and len(faculty_ids) == len(normalized),
        }

    # ──────────────────────────────────────────────────────────────────
    # Ingestion helpers
    # ──────────────────────────────────────────────────────────────────

    def _classify_emails(
        self,
        normalized: List[str],
        stale_threshold_days: int,
    ) -> Dict[str, List[str]]:
        """
        Partition emails into three buckets:
          - known:   in DB and profile refreshed within the threshold
          - stale:   in DB but never refreshed or refreshed too long ago
          - unknown: not in DB at all
        """
        known: List[str] = []
        stale: List[str] = []
        unknown: List[str] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=stale_threshold_days)

        try:
            with self.session_factory() as sess:
                dao = FacultyDAO(sess)
                for email in normalized:
                    fac = dao.get_by_email(email)
                    if fac is None:
                        unknown.append(email)
                        continue

                    last_refreshed = getattr(fac, "profile_last_refreshed_at", None)
                    if last_refreshed is None:
                        stale.append(email)
                    else:
                        if last_refreshed.tzinfo is None:
                            last_refreshed = last_refreshed.replace(tzinfo=timezone.utc)
                        if last_refreshed < cutoff:
                            stale.append(email)
                        else:
                            known.append(email)
        except Exception:
            logger.exception("FacultyContextAgent._classify_emails failed; treating all as unknown")
            return {"known": [], "stale": [], "unknown": list(normalized)}

        return {"known": known, "stale": stale, "unknown": unknown}

    def _scrape_and_upsert_one(self, email: str, osu_url: str) -> int:
        """
        Scrape the OSU engineering profile for `email` using the supplied URL,
        upsert the record into the DB, and enrich from personal/lab website links.
        Publication ingestion from a CV PDF is handled separately.

        osu_url: the exact OSU engineering profile URL provided by the caller.
                 No URL is derived automatically — callers must always pass it.

        Opens its own DB session — safe to call from a worker thread.
        Returns the faculty_id on success; raises on failure.
        """
        url = osu_url
        profile = parse_profile(url)

        # Guarantee the email field is populated even when the page omits it.
        if not profile.get("email"):
            profile["email"] = email

        dto = map_faculty_profile_to_dto(profile)

        with self.session_factory() as sess:
            dao = FacultyDAO(sess)
            fac = dao.upsert_faculty(dto)
            sess.flush()
            faculty_id: int = int(fac.faculty_id)

            if dto.additional_info:
                dao.upsert_additional_info(faculty_id, dto.additional_info)

            fac.profile_last_refreshed_at = datetime.now(timezone.utc)
            sess.commit()

        # Resolve personal/lab website from additional_info DTOs.
        # Scholar URLs are no longer used — publication ingestion is handled
        # separately via CV upload (utils/publication_extractor.py).
        personal_website_url: Optional[str] = next(
            (
                info.additional_info_url
                for info in (dto.additional_info or [])
                if info.additional_info_url
                and "oregonstate.edu" not in info.additional_info_url
            ),
            None,
        )

        # enrich_new_faculty opens its own sessions internally.
        enrich_new_faculty(
            email=email,
            faculty_id=faculty_id,
            osu_webpage=url,
            personal_website=personal_website_url,
        )

        return faculty_id

    # ──────────────────────────────────────────────────────────────────
    # Public ingestion method
    # ──────────────────────────────────────────────────────────────────

    def resolve_and_ingest_faculties(
        self,
        *,
        emails: List[str],
        keyword_generator: Optional[Any] = None,
        stale_threshold_days: int = _STALE_DAYS,
        cv_pdf_map: Optional[Dict[str, bytes]] = None,
        osu_url_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Full resolution with automatic ingestion of missing or stale faculty.

        Steps:
          1. Classify each email as known / stale / unknown.
          2. Scrape + upsert stale and unknown emails in parallel
             (up to _MAX_SCRAPE_WORKERS threads, each with its own DB session).
          3. Run keyword generation sequentially for every newly ingested
             faculty (serialised to avoid Bedrock rate limits).
          4. If cv_pdf_map is provided, run publication enrichment from the
             per-faculty CV PDFs.  Only emails present as keys in cv_pdf_map
             are enriched — the rest are skipped.  Title-based dedup makes
             re-runs safe.

        cv_pdf_map: Dict[str, bytes]
            email (lowercase) → raw PDF bytes for that faculty member's CV.
            Supports 0, 1, or N CVs in a single call.

        osu_url_map: Dict[str, str]
            email (lowercase) → explicit OSU engineering profile URL.
            Required for any email that is not already in the DB (unknown) or
            needs a profile refresh (stale).  Without an entry here, stale
            emails are kept as-is and unknown emails are marked as failed.

        Returns:
            {
                "resolved":    List[int],   # faculty_ids ready for matching
                "newly_added": List[str],   # emails successfully scraped + inserted
                "failed":      List[str],   # emails whose ingestion failed
            }
        """
        self._call("FacultyContextAgent.resolve_and_ingest_faculties")
        normalized = self._normalize_emails(emails)

        # ── 1. Classify ───────────────────────────────────────────────
        classes = self._classify_emails(normalized, stale_threshold_days)
        url_map = osu_url_map or {}

        # Stale emails with no OSU URL provided → keep existing DB data (treat as known).
        # Unknown emails with no OSU URL provided → cannot create record → fail.
        stale_with_url    = [e for e in classes["stale"]   if e in url_map]
        stale_without_url = [e for e in classes["stale"]   if e not in url_map]
        unknown_with_url  = [e for e in classes["unknown"] if e in url_map]
        unknown_without_url = [e for e in classes["unknown"] if e not in url_map]

        known_emails: List[str] = classes["known"] + stale_without_url
        to_ingest: List[str]    = stale_with_url + unknown_with_url

        logger.info(
            "FacultyContextAgent.resolve_and_ingest_faculties: "
            "known=%d stale_refresh=%d stale_skip=%d unknown_ingest=%d unknown_fail=%d",
            len(classes["known"]),
            len(stale_with_url),
            len(stale_without_url),
            len(unknown_with_url),
            len(unknown_without_url),
        )

        if unknown_without_url:
            logger.warning(
                "FacultyContextAgent: %d email(s) not in DB and no osu_url provided — "
                "they will be marked as failed: %s",
                len(unknown_without_url),
                unknown_without_url,
            )

        # Fetch faculty_ids for already-known emails and build email→fid map.
        email_to_fid: Dict[str, int] = {}
        try:
            with self.session_factory() as sess:
                dao = FacultyDAO(sess)
                for email in known_emails:
                    fid = dao.get_faculty_id_by_email(email)
                    if fid is not None:
                        email_to_fid[email] = int(fid)
        except Exception:
            logger.exception("FacultyContextAgent: could not fetch known faculty_ids")

        # ── 2. Parallel scrape + upsert ───────────────────────────────
        newly_added: List[str] = []
        failed: List[str] = list(unknown_without_url)

        if to_ingest:
            with ThreadPoolExecutor(max_workers=_MAX_SCRAPE_WORKERS) as pool:
                future_to_email = {
                    pool.submit(
                        self._scrape_and_upsert_one,
                        email,
                        url_map[email],   # always present — filtered above
                    ): email
                    for email in to_ingest
                }
                for future in as_completed(future_to_email):
                    email = future_to_email[future]
                    try:
                        faculty_id = future.result()
                        newly_added.append(email)
                        email_to_fid[email] = faculty_id
                        logger.info(
                            "FacultyContextAgent: ingested %s → faculty_id=%s",
                            email,
                            faculty_id,
                        )
                    except Exception:
                        logger.exception("FacultyContextAgent: ingestion failed for %s", email)
                        failed.append(email)

        # ── 3. Sequential keyword generation ─────────────────────────
        if keyword_generator is not None:
            for email in newly_added:
                fid = email_to_fid.get(email)
                if not fid:
                    continue
                try:
                    keyword_generator.generate_faculty_keywords_for_id(
                        fid, force_regenerate=True
                    )
                    logger.info(
                        "FacultyContextAgent: keywords generated for faculty_id=%s", fid
                    )
                except Exception:
                    logger.exception(
                        "FacultyContextAgent: keyword generation failed for faculty_id=%s", fid
                    )

        # ── 4. CV publication enrichment (per-email, explicit map) ────
        # For each email that has an entry in cv_pdf_map, run enrichment
        # against that specific faculty's CV.  Works for 1 or N CVs.
        if cv_pdf_map:
            for email in normalized:
                cv_bytes = cv_pdf_map.get(email)
                fid = email_to_fid.get(email)
                if not cv_bytes or not fid:
                    continue
                try:
                    inserted = enrich_faculty_publications_from_cv(fid, cv_bytes)
                    logger.info(
                        "FacultyContextAgent: CV enrichment inserted %d publications "
                        "for email=%s faculty_id=%s",
                        inserted,
                        email,
                        fid,
                    )
                except Exception:
                    logger.exception(
                        "FacultyContextAgent: CV enrichment failed for email=%s faculty_id=%s",
                        email,
                        fid,
                    )

        resolved = list(email_to_fid.values())
        return {
            "resolved": resolved,
            "newly_added": newly_added,
            "failed": failed,
        }

    # ──────────────────────────────────────────────────────────────────
    # Conversation helpers — unchanged
    # ──────────────────────────────────────────────────────────────────

    def ask_for_email(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_email")
        return {"next_action": "ask_email"}

    def ask_for_group_emails(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_group_emails")
        return {"next_action": "ask_group_emails"}

    def ask_for_user_reference_data(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_user_reference_data")
        return {"next_action": "ask_user_reference_data"}

