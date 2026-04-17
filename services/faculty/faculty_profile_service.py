from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty, FacultyAdditionalInfo, FacultyPublication


class FacultyProfileService:
    """Read/write service that shapes faculty profile data for frontend consumption."""

    def __init__(self, *, session_factory=SessionLocal, presigned_ttl_seconds: int = 3600):
        self.session_factory = session_factory
        self.presigned_ttl_seconds = max(60, int(presigned_ttl_seconds or 3600))
        self._s3_client = None

    # ---------------------------------------------------------------------
    # Read
    # ---------------------------------------------------------------------

    def get_faculty_profile(
        self,
        *,
        faculty_id: Optional[int] = None,
        email: Optional[str] = None,
        publication_year_from: Optional[int] = None,
        publication_year_to: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_email = (str(email or "").strip().lower() or None)
        use_faculty_id = int(faculty_id) if faculty_id else None
        year_from = self._safe_int_or_none(publication_year_from)
        year_to = self._safe_int_or_none(publication_year_to)
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("publication_year_from cannot be greater than publication_year_to.")

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
            return self._serialize_faculty(
                fac,
                publication_year_from=year_from,
                publication_year_to=year_to,
            )

    def list_faculty_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        publication_year_from: Optional[int] = None,
        publication_year_to: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        year_from = self._safe_int_or_none(publication_year_from)
        year_to = self._safe_int_or_none(publication_year_to)
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("publication_year_from cannot be greater than publication_year_to.")
        with self.session_factory() as sess:
            dao = FacultyDAO(sess)
            rows = dao.list_with_relations(limit=limit, offset=offset)
            return [
                self._serialize_faculty(
                    fac,
                    publication_year_from=year_from,
                    publication_year_to=year_to,
                )
                for fac in rows
            ]

    # ---------------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------------

    def edit_faculty_profile(
        self,
        *,
        email: str,
        basic_info: Optional[Dict[str, Any]] = None,
        data_from: Optional[Dict[str, Any]] = None,
        all_keywords: Optional[Dict[str, Any]] = None,
        keyword_source: Optional[str] = None,
        force_regenerate_keywords: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Edit faculty profile by immutable email.

        Mode 1: source edits (basic_info/data_from) -> regenerate keywords.
        Mode 2: direct all_keywords edit -> write directly to faculty keyword tables only.
        Mode 3: force_regenerate_keywords=true -> regenerate from current sources.
        """
        normalized_email = (str(email or "").strip().lower() or None)
        if not normalized_email:
            raise ValueError("email is required.")

        basic_info = basic_info or {}
        data_from = data_from or {}
        has_keyword_payload = all_keywords is not None
        has_source_payload = bool(basic_info) or bool(data_from)
        force_regeneration = bool(force_regenerate_keywords)

        if has_keyword_payload and force_regeneration:
            raise ValueError(
                "all_keywords direct update cannot be combined with force_regenerate_keywords=true."
            )

        # Keyword payload takes priority as a DB-only override path.
        # Ignore source payload fields if they are sent together.
        if has_keyword_payload:
            basic_info = {}
            data_from = {}
            has_source_payload = False

        with self.session_factory() as sess:
            dao = FacultyDAO(sess)
            fac = dao.get_with_relations_by_email(normalized_email)
            if not fac:
                raise LookupError("Faculty not found.")

            source_changed = False
            source_detail = {
                "basic_info_updated": False,
                "data_from_updated": False,
                "publications_added": 0,
                "publications_updated": 0,
                "publications_deleted": 0,
                "attached_files_added": 0,
                "attached_files_updated": 0,
                "attached_files_deleted": 0,
            }

            if basic_info:
                changed = self._apply_basic_info_update(fac, basic_info)
                source_changed = source_changed or changed
                source_detail["basic_info_updated"] = bool(changed)

            if data_from:
                changed, delta = self._apply_data_from_update(sess=sess, fac=fac, payload=data_from)
                source_changed = source_changed or changed
                source_detail["data_from_updated"] = bool(changed)
                for k in (
                    "publications_added",
                    "publications_updated",
                    "publications_deleted",
                    "attached_files_added",
                    "attached_files_updated",
                    "attached_files_deleted",
                ):
                    source_detail[k] += int(delta.get(k, 0))

            direct_keyword_applied = False
            if has_keyword_payload:
                self._apply_direct_keyword_update(
                    dao=dao,
                    faculty_id=int(fac.faculty_id),
                    keywords=(all_keywords or {}),
                    source=keyword_source,
                )
                direct_keyword_applied = True

            sess.commit()
            faculty_id = int(fac.faculty_id)

        keyword_regenerated = False
        keyword_regeneration_error = None
        keyword_update_mode = "none"
        if direct_keyword_applied:
            keyword_update_mode = "frontend_override"
        elif source_changed or force_regeneration:
            try:
                self._regenerate_faculty_keywords(faculty_id=faculty_id)
                keyword_regenerated = True
                keyword_update_mode = (
                    "forced_regeneration"
                    if force_regeneration
                    else "regenerated_from_sources"
                )
            except Exception as e:
                keyword_regeneration_error = f"{type(e).__name__}: {e}"
                keyword_update_mode = (
                    "forced_regeneration_failed"
                    if force_regeneration
                    else "regeneration_failed"
                )
                if force_regeneration:
                    raise RuntimeError(
                        f"Forced keyword regeneration failed: {keyword_regeneration_error}"
                    ) from e

        matches_rebuilt = False
        match_rows_upserted = 0
        match_rebuild_error = None
        # Direct keyword edits are DB-only and should not trigger downstream jobs.
        if keyword_regenerated:
            try:
                match_rows_upserted = int(
                    self._rebuild_faculty_matches(faculty_id=faculty_id)
                )
                matches_rebuilt = True
            except Exception as e:
                match_rebuild_error = f"{type(e).__name__}: {e}"

        updated = self.get_faculty_profile(faculty_id=faculty_id)
        return {
            "faculty": updated,
            "source_changed": bool(source_changed),
            "source_change_detail": source_detail,
            "direct_keyword_applied": bool(direct_keyword_applied),
            "keyword_update_mode": keyword_update_mode,
            "keyword_regenerated": bool(keyword_regenerated),
            "keyword_regeneration_forced": bool(force_regeneration),
            "keyword_regeneration_error": keyword_regeneration_error,
            "updated_keywords": ((updated or {}).get("all_keywords") or {}),
            "matches_rebuilt": bool(matches_rebuilt),
            "match_rows_upserted": int(match_rows_upserted),
            "match_rebuild_error": match_rebuild_error,
        }

    @staticmethod
    def _apply_basic_info_update(fac: Faculty, payload: Dict[str, Any]) -> bool:
        if "email" in payload:
            raise ValueError("email is immutable and cannot be updated.")

        changed = False

        if "faculty_name" in payload or "name" in payload:
            name_value = payload.get("faculty_name") if "faculty_name" in payload else payload.get("name")
            new_name = str(name_value).strip() or None if name_value is not None else None
            if fac.name != new_name:
                fac.name = new_name
                changed = True

        for field in ("position", "phone", "address", "biography"):
            if field not in payload:
                continue
            value = payload.get(field)
            new_value = str(value).strip() or None if value is not None else None
            if getattr(fac, field) != new_value:
                setattr(fac, field, new_value)
                changed = True

        if "degrees" in payload:
            raw = payload.get("degrees")
            if raw is None:
                normalized = None
            else:
                if not isinstance(raw, list):
                    raise ValueError("degrees must be a list or null.")
                normalized = [str(x).strip() for x in raw if str(x).strip()]
            if fac.degrees != normalized:
                fac.degrees = normalized
                changed = True

        if "expertise" in payload:
            raw = payload.get("expertise")
            if raw is None:
                normalized = None
            else:
                if not isinstance(raw, list):
                    raise ValueError("expertise must be a list or null.")
                normalized = [str(x).strip() for x in raw if str(x).strip()]
            if fac.expertise != normalized:
                fac.expertise = normalized
                changed = True

        if "organizations" in payload:
            raw = payload.get("organizations")
            if raw is None:
                normalized_orgs = None
            else:
                if not isinstance(raw, list):
                    raise ValueError("organizations must be a list or null.")
                normalized_orgs = [str(x).strip() for x in raw if str(x).strip()]
            one_line = " | ".join(normalized_orgs or []) if normalized_orgs else None
            if fac.organizations != normalized_orgs:
                fac.organizations = normalized_orgs
                changed = True
            if fac.organization != one_line:
                fac.organization = one_line
                changed = True
        elif "organization" in payload:
            one_line = str(payload.get("organization") or "").strip() or None
            if fac.organization != one_line:
                fac.organization = one_line
                changed = True

        return changed

    def _apply_data_from_update(
        self,
        *,
        sess,
        fac: Faculty,
        payload: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, int]]:
        changed = False
        delta = {
            "publications_added": 0,
            "publications_updated": 0,
            "publications_deleted": 0,
            "attached_files_added": 0,
            "attached_files_updated": 0,
            "attached_files_deleted": 0,
        }

        if "info_source_url" in payload:
            new_url = str(payload.get("info_source_url") or "").strip()
            if not new_url:
                raise ValueError("info_source_url cannot be empty.")
            if fac.source_url != new_url:
                fac.source_url = new_url
                changed = True

        if "publications" in payload:
            pub_delta = self._apply_publication_ops(
                sess=sess,
                faculty_id=int(fac.faculty_id),
                ops=payload.get("publications"),
            )
            for k in ("publications_added", "publications_updated", "publications_deleted"):
                delta[k] += int(pub_delta.get(k, 0))
            if any(delta[k] > 0 for k in ("publications_added", "publications_updated", "publications_deleted")):
                changed = True

        if "attached_files" in payload:
            file_delta = self._apply_attached_file_ops(
                sess=sess,
                faculty_id=int(fac.faculty_id),
                ops=payload.get("attached_files"),
            )
            for k in ("attached_files_added", "attached_files_updated", "attached_files_deleted"):
                delta[k] += int(file_delta.get(k, 0))
            if any(delta[k] > 0 for k in ("attached_files_added", "attached_files_updated", "attached_files_deleted")):
                changed = True

        return changed, delta

    def _apply_publication_ops(self, *, sess, faculty_id: int, ops: Any) -> Dict[str, int]:
        if not isinstance(ops, dict):
            raise ValueError(
                "data_from.publications must be an object with set_fetch_year_range/from/to and/or delete."
            )

        out = {
            "publications_added": 0,
            "publications_updated": 0,
            "publications_deleted": 0,
        }
        allowed_keys = {
            "set_fetch_year_range",
            "set_fetch_from_year",
            "publication_fetched_from_year",
            "set_fetch_upto_year",
            "publication_fetched_upto_year",
            "delete",
        }
        unknown = [k for k in ops.keys() if k not in allowed_keys]
        if unknown:
            raise ValueError(
                "data_from.publications supports only set_fetch_year_range/from/to and delete."
            )

        year_range = ops.get("set_fetch_year_range")
        fetch_from_year = None
        fetch_upto_year = None
        if isinstance(year_range, dict):
            fetch_from_year = FacultyProfileService._safe_int_or_none(
                year_range.get("from")
                or year_range.get("start")
                or year_range.get("min")
                or year_range.get("year_from")
            )
            fetch_upto_year = FacultyProfileService._safe_int_or_none(
                year_range.get("to")
                or year_range.get("end")
                or year_range.get("max")
                or year_range.get("year_to")
            )

        if fetch_from_year is None:
            fetch_from_year = FacultyProfileService._safe_int_or_none(ops.get("set_fetch_from_year"))
        if fetch_from_year is None:
            fetch_from_year = FacultyProfileService._safe_int_or_none(
                ops.get("publication_fetched_from_year")
            )

        if fetch_upto_year is None:
            fetch_upto_year = FacultyProfileService._safe_int_or_none(ops.get("set_fetch_upto_year"))
        if fetch_upto_year is None:
            fetch_upto_year = FacultyProfileService._safe_int_or_none(
                ops.get("publication_fetched_upto_year")
            )

        if (
            fetch_from_year is not None
            and fetch_upto_year is not None
            and int(fetch_from_year) > int(fetch_upto_year)
        ):
            raise ValueError("publication year range is invalid: from_year cannot be greater than to_year.")

        range_requested = (fetch_from_year is not None) or (fetch_upto_year is not None)
        if range_requested:
            fac = (
                sess.query(Faculty)
                .filter(Faculty.faculty_id == int(faculty_id))
                .one_or_none()
            )
            if fac:
                sync_from, sync_to = self._resolve_sync_range_bounds(
                    sess=sess,
                    faculty_id=int(faculty_id),
                    year_from=fetch_from_year,
                    year_to=fetch_upto_year,
                )
                fetched = self._fetch_publications_from_source_for_range(
                    faculty_name=str(getattr(fac, "name", "") or ""),
                    org_hint=str(getattr(fac, "organization", "") or ""),
                    year_from=sync_from,
                    year_to=sync_to,
                )
                added, updated = self._upsert_publications_from_source(
                    sess=sess,
                    faculty_id=int(faculty_id),
                    rows=fetched,
                )
                out["publications_added"] += int(added)
                out["publications_updated"] += int(updated)

        if fetch_from_year is not None:
            # Keep only publications on/after from_year.
            deleted_older = self._delete_publication_rows(
                sess=sess,
                faculty_id=int(faculty_id),
                filters=(
                    FacultyPublication.year.isnot(None),
                    FacultyPublication.year < int(fetch_from_year),
                ),
            )
            out["publications_deleted"] += int(deleted_older or 0)

        if fetch_upto_year is not None:
            # Keep only publications on/before upto_year.
            deleted_newer = self._delete_publication_rows(
                sess=sess,
                faculty_id=int(faculty_id),
                filters=(
                    FacultyPublication.year.isnot(None),
                    FacultyPublication.year > int(fetch_upto_year),
                ),
            )
            out["publications_deleted"] += int(deleted_newer or 0)

        delete_raw = ops.get("delete")
        delete_id: Optional[int] = None
        if delete_raw is not None:
            if isinstance(delete_raw, list):
                cleaned = [
                    FacultyProfileService._safe_int_or_none(x)
                    for x in delete_raw
                ]
                cleaned = [int(x) for x in cleaned if x]
                if len(cleaned) > 1:
                    raise ValueError("publications.delete supports one publication id at a time.")
                delete_id = cleaned[0] if cleaned else None
            else:
                delete_id = FacultyProfileService._safe_int_or_none(delete_raw)

        if delete_id:
            deleted_one = self._delete_publication_rows(
                sess=sess,
                faculty_id=int(faculty_id),
                filters=(FacultyPublication.id == int(delete_id),),
            )
            out["publications_deleted"] += int(deleted_one or 0)

        return out

    @staticmethod
    def _delete_publication_rows(
        *,
        sess,
        faculty_id: int,
        filters: Tuple[Any, ...],
    ) -> int:
        """
        Session-aware publication deletion.

        Avoids ORM stale state when rows are loaded/updated in the same session
        and then removed via bulk DELETE.
        """
        rows = (
            sess.query(FacultyPublication)
            .filter(
                FacultyPublication.faculty_id == int(faculty_id),
                *filters,
            )
            .all()
        )
        for row in rows:
            sess.delete(row)
        return len(rows)

    @staticmethod
    def _resolve_sync_range_bounds(
        *,
        sess,
        faculty_id: int,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> Tuple[int, int]:
        from_year = FacultyProfileService._safe_int_or_none(year_from)
        to_year = FacultyProfileService._safe_int_or_none(year_to)
        current_year = datetime.now().year

        if from_year is None:
            existing_min = (
                sess.query(FacultyPublication.year)
                .filter(
                    FacultyPublication.faculty_id == int(faculty_id),
                    FacultyPublication.year.isnot(None),
                )
                .order_by(FacultyPublication.year.asc())
                .limit(1)
                .scalar()
            )
            from_year = int(existing_min) if existing_min is not None else max(current_year - 25, 1900)

        if to_year is None:
            to_year = int(current_year)

        if int(from_year) > int(to_year):
            from_year = int(to_year)
        return int(from_year), int(to_year)

    @staticmethod
    def _fetch_publications_from_source_for_range(
        *,
        faculty_name: str,
        org_hint: str,
        year_from: int,
        year_to: int,
    ) -> List["FacultyPublicationDTO"]:
        try:
            from services.faculty.author_publication_fetcher import AuthorPublicationFetcher
        except Exception:
            return []

        try:
            fetcher = AuthorPublicationFetcher()
            author_id = fetcher.resolve_author_id(
                faculty_name=str(faculty_name or ""),
                org_hint=str(org_hint or ""),
            )
            if not author_id:
                return []
            return fetcher.fetch_publications_for_author_year_range(
                author_id=author_id,
                year_from=int(year_from),
                year_to=int(year_to),
            )
        except Exception:
            return []

    @staticmethod
    def _upsert_publications_from_source(
        *,
        sess,
        faculty_id: int,
        rows: List["FacultyPublicationDTO"],
    ) -> Tuple[int, int]:
        added = 0
        updated = 0
        for dto in rows or []:
            title = str(getattr(dto, "title", "") or "").strip()
            if not title:
                continue

            work_id = str(getattr(dto, "openalex_work_id", "") or "").strip() or None
            pub = None
            if work_id:
                pub = (
                    sess.query(FacultyPublication)
                    .filter(
                        FacultyPublication.faculty_id == int(faculty_id),
                        FacultyPublication.openalex_work_id == work_id,
                    )
                    .order_by(FacultyPublication.id.desc())
                    .first()
                )
            if pub is None:
                # Title can have historical duplicates (especially legacy CV imports),
                # so pick the most recent row instead of requiring uniqueness.
                pub = (
                    sess.query(FacultyPublication)
                    .filter(
                        FacultyPublication.faculty_id == int(faculty_id),
                        FacultyPublication.title == title,
                    )
                    .order_by(FacultyPublication.id.desc())
                    .first()
                )

            if pub is None:
                pub = FacultyPublication(
                    faculty_id=int(faculty_id),
                    openalex_work_id=work_id,
                )
                sess.add(pub)
                added += 1
            else:
                updated += 1

            pub.title = title
            pub.year = FacultyProfileService._safe_int_or_none(getattr(dto, "year", None))
            pub.abstract = str(getattr(dto, "abstract", "") or "").strip() or None
            pub.scholar_author_id = str(getattr(dto, "scholar_author_id", "") or "").strip() or None
            if work_id is not None:
                pub.openalex_work_id = work_id

        return added, updated

    @staticmethod
    def _apply_attached_file_ops(*, sess, faculty_id: int, ops: Any) -> Dict[str, int]:
        if not isinstance(ops, dict):
            raise ValueError("data_from.attached_files must be an object with add/update/delete.")

        out = {
            "attached_files_added": 0,
            "attached_files_updated": 0,
            "attached_files_deleted": 0,
        }

        for row in FacultyProfileService._as_list(ops.get("add")):
            if not isinstance(row, dict):
                raise ValueError("attached_files.add items must be objects.")
            source_url = str(
                row.get("source_url") or row.get("additional_info_url") or ""
            ).strip()
            if not source_url:
                raise ValueError("attached_files.add requires source_url.")

            info = (
                sess.query(FacultyAdditionalInfo)
                .filter(
                    FacultyAdditionalInfo.faculty_id == int(faculty_id),
                    FacultyAdditionalInfo.additional_info_url == source_url,
                )
                .one_or_none()
            )
            if info is None:
                info = FacultyAdditionalInfo(
                    faculty_id=int(faculty_id),
                    additional_info_url=source_url,
                    extract_status="pending",
                )
                sess.add(info)
                out["attached_files_added"] += 1
            else:
                out["attached_files_updated"] += 1

            if "content_path" in row:
                info.content_path = str(row.get("content_path") or "").strip() or None
            if "extract_status" in row:
                info.extract_status = str(row.get("extract_status") or "").strip() or "pending"
            if "detected_type" in row:
                info.detected_type = str(row.get("detected_type") or "").strip() or None
            if "content_char_count" in row:
                info.content_char_count = FacultyProfileService._safe_int_or_none(row.get("content_char_count"))

        for row in FacultyProfileService._as_list(ops.get("update")):
            if not isinstance(row, dict):
                raise ValueError("attached_files.update items must be objects.")
            info_id = FacultyProfileService._safe_int_or_none(row.get("id"))
            if not info_id:
                raise ValueError("attached_files.update requires id.")

            info = (
                sess.query(FacultyAdditionalInfo)
                .filter(
                    FacultyAdditionalInfo.id == int(info_id),
                    FacultyAdditionalInfo.faculty_id == int(faculty_id),
                )
                .one_or_none()
            )
            if info is None:
                raise LookupError(f"Attached file not found: id={info_id}")

            if "source_url" in row or "additional_info_url" in row:
                new_url = str(
                    row.get("source_url") or row.get("additional_info_url") or ""
                ).strip()
                if not new_url:
                    raise ValueError("attached_files source_url cannot be empty.")
                info.additional_info_url = new_url
            if "content_path" in row:
                info.content_path = str(row.get("content_path") or "").strip() or None
            if "extract_status" in row:
                info.extract_status = str(row.get("extract_status") or "").strip() or "pending"
            if "detected_type" in row:
                info.detected_type = str(row.get("detected_type") or "").strip() or None
            if "content_char_count" in row:
                info.content_char_count = FacultyProfileService._safe_int_or_none(row.get("content_char_count"))
            out["attached_files_updated"] += 1

        delete_ids = [
            FacultyProfileService._safe_int_or_none(x)
            for x in FacultyProfileService._as_list(ops.get("delete"))
        ]
        delete_ids = [int(x) for x in delete_ids if x]
        if delete_ids:
            deleted = (
                sess.query(FacultyAdditionalInfo)
                .filter(
                    FacultyAdditionalInfo.faculty_id == int(faculty_id),
                    FacultyAdditionalInfo.id.in_(delete_ids),
                )
                .delete(synchronize_session=False)
            )
            out["attached_files_deleted"] += int(deleted or 0)

        return out

    def _regenerate_faculty_keywords(self, *, faculty_id: int) -> None:
        # Lazy import so retrieval path does not require keyword-stack deps.
        from services.context_retrieval.context_generator import ContextGenerator
        from services.keywords.keyword_generator import FacultyKeywordGenerator

        keyword_generator = FacultyKeywordGenerator(context_generator=ContextGenerator())
        keyword_generator.generate_faculty_keywords_for_id(
            int(faculty_id),
            force_regenerate=True,
        )

    def _rebuild_faculty_matches(self, *, faculty_id: int) -> int:
        # Recompute one-to-one rows for this faculty from current keyword/embedding state.
        from services.matching.faculty_grant_matcher import FacultyGrantMatcher

        matcher = FacultyGrantMatcher(session_factory=self.session_factory)
        return int(
            matcher.run_for_faculty(
                faculty_id=int(faculty_id),
                k=200,
                min_domain=0.0,
            )
        )

    @staticmethod
    def _apply_direct_keyword_update(
        *,
        dao: FacultyDAO,
        faculty_id: int,
        keywords: Dict[str, Any],
        source: Optional[str] = None,
    ) -> None:
        kw = FacultyProfileService._normalize_keywords_payload(keywords)
        source_name = str(source or "user_edit").strip() or "user_edit"

        dao.upsert_keywords_json(
            [
                {
                    "faculty_id": int(faculty_id),
                    "keywords": kw,
                    "raw_json": {
                        "edited": True,
                        "mode": "direct_update",
                    },
                    "source": source_name,
                }
            ]
        )

        # Keep embedding in sync with directly edited keywords.
        try:
            from utils.embedder import embed_domain_bucket
            from utils.keyword_utils import extract_domains_from_keywords

            r_domains, a_domains = extract_domains_from_keywords(kw)
            r_vec = embed_domain_bucket(r_domains)
            a_vec = embed_domain_bucket(a_domains)
            if r_vec is not None or a_vec is not None:
                dao.upsert_keyword_embedding(
                    {
                        "faculty_id": int(faculty_id),
                        "model": settings.bedrock_embed_model_id,
                        "research_domain_vec": r_vec,
                        "application_domain_vec": a_vec,
                    }
                )
        except Exception:
            # Keyword JSON update is still valid even if embedding refresh fails.
            pass

    @staticmethod
    def _normalize_keywords_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("all_keywords must be an object.")

        # Lazy import so retrieval path does not require keyword-stack deps.
        from utils.keyword_utils import coerce_keyword_sections

        kw = coerce_keyword_sections(dict(payload or {}))
        out: Dict[str, Any] = {}

        for section in ("research", "application"):
            sec = kw.get(section)
            if not isinstance(sec, dict):
                sec = {}

            raw_domains = sec.get("domain") or []
            if isinstance(raw_domains, str):
                raw_domains = [raw_domains]
            if not isinstance(raw_domains, list):
                raise ValueError(f"all_keywords.{section}.domain must be a list.")
            domains = [str(x).strip() for x in raw_domains if str(x).strip()]

            raw_specs = sec.get("specialization") or []
            if isinstance(raw_specs, str):
                raw_specs = [raw_specs]
            if not isinstance(raw_specs, list):
                raise ValueError(f"all_keywords.{section}.specialization must be a list.")

            specs: List[Dict[str, Any]] = []
            for item in raw_specs:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        specs.append({"t": text, "w": 1.0})
                    continue
                if isinstance(item, dict):
                    text = str(item.get("t") or item.get("text") or "").strip()
                    if not text:
                        continue
                    raw_w = item.get("w", item.get("weight", 1.0))
                    try:
                        weight = float(raw_w)
                    except Exception:
                        weight = 1.0
                    weight = max(0.0, min(weight, 1.0))
                    specs.append({"t": text, "w": weight})
                    continue
                raise ValueError(
                    f"all_keywords.{section}.specialization items must be string or object."
                )

            out[section] = {"domain": domains, "specialization": specs}

        return out

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------

    def _serialize_faculty(
        self,
        fac: Faculty,
        *,
        publication_year_from: Optional[int] = None,
        publication_year_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        publications_all = sorted(
            list(fac.publications or []),
            key=lambda p: ((p.year or 0), (p.title or "")),
            reverse=True,
        )

        publication_years_all = [int(p.year) for p in publications_all if p.year is not None]
        publication_year_options = self._build_publication_year_options(publication_years_all)
        publication_first_year = min(publication_years_all) if publication_years_all else None
        publication_latest_year = max(publication_years_all) if publication_years_all else None

        year_from = self._safe_int_or_none(publication_year_from)
        year_to = self._safe_int_or_none(publication_year_to)
        publications = publications_all
        if year_from is not None or year_to is not None:
            filtered: List[FacultyPublication] = []
            for p in publications_all:
                if p.year is None:
                    continue
                yy = int(p.year)
                if year_from is not None and yy < int(year_from):
                    continue
                if year_to is not None and yy > int(year_to):
                    continue
                filtered.append(p)
            publications = filtered

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
                "publication_first_year": publication_first_year,
                "publication_latest_year": publication_latest_year,
                "publication_year_options": publication_year_options,
                "publication_selected_from_year": year_from,
                "publication_selected_to_year": year_to,
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
            "id": int(row.id),
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
            "id": int(p.id),
            "openalex_work_id": p.openalex_work_id,
            "scholar_author_id": p.scholar_author_id,
            "title": p.title,
            "year": p.year,
            "abstract": p.abstract,
        }

    @staticmethod
    def _isoformat_or_none(dt_value: Optional[datetime]) -> Optional[str]:
        if dt_value is None:
            return None
        if isinstance(dt_value, datetime):
            return dt_value.isoformat()
        return str(dt_value)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _build_download_paths(self, content_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        parsed = self._parse_bucket_key(content_path)
        if not parsed:
            return None, None

        bucket, key = parsed
        s3_uri = f"s3://{bucket}/{key}"

        try:
            s3 = self._get_s3_client()
            if s3 is None:
                return s3_uri, None
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

        try:
            import boto3
        except Exception:
            return None

        session = (
            boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
            if settings.aws_profile
            else boto3.Session(region_name=settings.aws_region)
        )
        self._s3_client = session.client("s3")
        return self._s3_client

    @staticmethod
    def _safe_int_or_none(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    @staticmethod
    def _as_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    @staticmethod
    def _build_publication_year_options(publication_years: List[int]) -> List[int]:
        """
        Build a descending selector list: current year down to earliest publication year.
        Example: [2026, 2025, ..., 2011]
        """
        cleaned = [int(y) for y in publication_years if y is not None]
        if not cleaned:
            return []
        earliest = min(cleaned)
        current_year = datetime.now().year
        start = max(current_year, max(cleaned))
        if earliest > start:
            return []
        return list(range(int(start), int(earliest) - 1, -1))
