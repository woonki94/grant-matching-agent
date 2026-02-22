from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from services.context.context_generator import ContextGenerator
from services.justification.group_justification_generator import GroupJustificationGenerator
from services.keywords.keyword_generator import KeywordGenerator
from services.matching.faculty_grant_matcher import FacultyGrantMatcher


class MatchingExecutionAgent:
    # Query-mode relevance gates to avoid returning weak tail matches.
    MIN_QUERY_SCORE_ABS = 0.20
    QUERY_SCORE_REL_MARGIN = 0.12
    MIN_LLM_SCORE_WITH_QUERY = 0.0

    def __init__(self, *, session_factory=SessionLocal, context_generator: Optional[ContextGenerator] = None):
        self.session_factory = session_factory
        self.context_generator = context_generator or ContextGenerator()
        self.keyword_generator = KeywordGenerator(context_generator=self.context_generator)
        self.faculty_matcher = FacultyGrantMatcher(session_factory=session_factory)
        self.group_justification_generator = GroupJustificationGenerator(
            session_factory=session_factory,
            context_generator=self.context_generator,
        )

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _normalize_broad_category_filter(broad_category: Any) -> Optional[Set[str]]:
        allowed = {"basic_research", "applied_research", "educational"}
        out: Set[str] = set()
        if broad_category is None:
            return None
        if isinstance(broad_category, str):
            v = broad_category.strip().lower()
            if v in allowed:
                out.add(v)
            return out or None
        if isinstance(broad_category, (list, tuple, set)):
            for item in broad_category:
                v = str(item or "").strip().lower()
                if v in allowed:
                    out.add(v)
            return out or None
        return None

    @staticmethod
    def _normalize_broad_category_for_output(broad_category: Any) -> Any:
        filters = MatchingExecutionAgent._normalize_broad_category_filter(broad_category)
        if not filters:
            return None
        if len(filters) == 1:
            return next(iter(filters))
        return sorted(filters)

    def _build_query_similarity_map(
        self,
        *,
        query_text: Optional[str],
        k: int,
    ) -> Dict[str, float]:
        q = str(query_text or "").strip()
        if not q:
            return {}
        try:
            from services.search.search_grants import generate_query_keywords
            from utils.embedder import embed_domain_bucket
            from utils.keyword_utils import extract_domains_from_keywords

            query_keywords = generate_query_keywords(q, user_urls=None)
            r_domains, a_domains = extract_domains_from_keywords(query_keywords)
            r_vec = embed_domain_bucket(r_domains)
            a_vec = embed_domain_bucket(a_domains)
            if r_vec is None and a_vec is None:
                return {}
            with self.session_factory() as sess:
                mdao = MatchDAO(sess)
                pairs = mdao.topk_opps_for_query(
                    research_vec=r_vec,
                    application_vec=a_vec,
                    k=max(int(k), 50),
                )
            return {str(oid): float(score) for oid, score in pairs}
        except Exception:
            return {}

    def _apply_preference_filters_to_opportunities(
        self,
        *,
        rows: List[Dict[str, Any]],
        top_k: int,
        broad_category: Any,
        query_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        category_filters = self._normalize_broad_category_filter(broad_category)
        if not category_filters and not str(query_text or "").strip():
            return rows[: int(top_k)]
        query_map = self._build_query_similarity_map(
            query_text=query_text,
            k=max(int(top_k) * 30, 200),
        ) if query_text else {}

        out: List[Dict[str, Any]] = []
        try:
            with self.session_factory() as sess:
                odao = OpportunityDAO(sess)
                for r in rows:
                    opp_id = str(r.get("opportunity_id") or "")
                    if not opp_id:
                        continue
                    opp_ctx = odao.read_opportunity_context(opp_id) or {}
                    broad = str(opp_ctx.get("broad_category") or "").strip().lower() or None
                    if category_filters and broad not in category_filters:
                        continue
                    item = dict(r)
                    item["broad_category"] = broad
                    item["specific_categories"] = list(opp_ctx.get("specific_categories") or [])
                    item["query_score"] = float(query_map.get(opp_id, 0.0)) if query_map else None
                    out.append(item)
        except Exception:
            out = list(rows)

        if query_map:
            out.sort(
                key=lambda x: (
                    float(x.get("query_score") or 0.0),
                    float(x.get("llm_score") or 0.0),
                    float(x.get("domain_score") or 0.0),
                ),
                reverse=True,
            )
            best_query_score = float(out[0].get("query_score") or 0.0) if out else 0.0
            query_floor = max(
                float(self.MIN_QUERY_SCORE_ABS),
                float(best_query_score - self.QUERY_SCORE_REL_MARGIN),
            )
            out = [
                item
                for item in out
                if float(item.get("query_score") or 0.0) >= query_floor
                and float(item.get("llm_score") or 0.0) >= float(self.MIN_LLM_SCORE_WITH_QUERY)
            ]

        return out[: int(top_k)]

    def _apply_preference_filters_to_group_results(
        self,
        *,
        rows: List[Dict[str, Any]],
        broad_category: Any,
        query_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        category_filters = self._normalize_broad_category_filter(broad_category)
        if not category_filters and not str(query_text or "").strip():
            return rows
        query_map = self._build_query_similarity_map(
            query_text=query_text,
            k=max(len(rows) * 20, 200),
        ) if query_text else {}

        out: List[Dict[str, Any]] = []
        try:
            with self.session_factory() as sess:
                odao = OpportunityDAO(sess)
                for r in rows:
                    opp_id = str(r.get("grant_id") or r.get("opp_id") or "")
                    if not opp_id:
                        continue
                    opp_ctx = odao.read_opportunity_context(opp_id) or {}
                    broad = str(opp_ctx.get("broad_category") or "").strip().lower() or None
                    if category_filters and broad not in category_filters:
                        continue
                    item = dict(r)
                    item["broad_category"] = broad
                    item["specific_categories"] = list(opp_ctx.get("specific_categories") or [])
                    if not item.get("agency_name") and opp_ctx.get("agency"):
                        item["agency_name"] = opp_ctx.get("agency")
                    item["query_score"] = float(query_map.get(opp_id, 0.0)) if query_map else None
                    out.append(item)
        except Exception:
            out = list(rows)

        if query_map:
            out.sort(
                key=lambda x: (
                    float(x.get("query_score") or 0.0),
                    float(x.get("score") or 0.0),
                ),
                reverse=True,
            )
            best_query_score = float(out[0].get("query_score") or 0.0) if out else 0.0
            query_floor = max(
                float(self.MIN_QUERY_SCORE_ABS),
                float(best_query_score - self.QUERY_SCORE_REL_MARGIN),
            )
            out = [
                item
                for item in out
                if float(item.get("query_score") or 0.0) >= query_floor
            ]

        return out

    def generate_keywords_for_group(self, *, faculty_ids: List[int]) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.generate_keywords_for_group")
        generated = 0
        skipped_existing = 0
        try:
            with self.session_factory() as sess:
                fac_dao = FacultyDAO(sess)
                for fid in faculty_ids or []:
                    fid_int = int(fid)
                    if fac_dao.has_keyword_row(fid_int):
                        skipped_existing += 1
                        continue
                    if self.keyword_generator.generate_faculty_keywords_for_id(fid_int):
                        generated += 1
        except Exception as e:
            return {
                "next_action": "error_generate_keywords_group",
                "keywords_generated_count": generated,
                "keywords_skipped_existing_count": skipped_existing,
                "error": f"{type(e).__name__}: {e}",
            }
        return {
            "next_action": "generated_keywords_group",
            "keywords_generated_count": generated,
            "keywords_skipped_existing_count": skipped_existing,
        }

    def generate_keywords_for_group_specific_grant(
        self,
        *,
        faculty_ids: List[int],
        opportunity_id: str,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.generate_keywords_for_group_specific_grant")
        fac_generated = 0
        fac_skipped_existing = 0
        opp_generated = False
        opp_skipped_existing = False
        try:
            with self.session_factory() as sess:
                fac_dao = FacultyDAO(sess)
                opp_dao = OpportunityDAO(sess)
                for fid in faculty_ids or []:
                    fid_int = int(fid)
                    if fac_dao.has_keyword_row(fid_int):
                        fac_skipped_existing += 1
                        continue
                    if self.keyword_generator.generate_faculty_keywords_for_id(fid_int):
                        fac_generated += 1

                if opp_dao.has_keyword_row(opportunity_id):
                    opp_skipped_existing = True
                else:
                    opp_kw = self.keyword_generator.generate_opportunity_keywords_for_id(opportunity_id)
                    opp_generated = bool(opp_kw)
        except Exception as e:
            return {
                "next_action": "error_generate_keywords_group_specific_grant",
                "faculty_keywords_generated_count": fac_generated,
                "faculty_keywords_skipped_existing_count": fac_skipped_existing,
                "opportunity_keywords_generated": opp_generated,
                "opportunity_keywords_skipped_existing": opp_skipped_existing,
                "error": f"{type(e).__name__}: {e}",
            }
        return {
            "next_action": "generated_keywords_group_specific_grant",
            "faculty_keywords_generated_count": fac_generated,
            "faculty_keywords_skipped_existing_count": fac_skipped_existing,
            "opportunity_keywords_generated": opp_generated,
            "opportunity_keywords_skipped_existing": opp_skipped_existing,
        }

    def generate_keywords_and_matches_for_one_to_one_specific_grant(
        self,
        *,
        faculty_id: int,
        opportunity_id: str,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.generate_keywords_and_matches_for_one_to_one_specific_grant")
        out = self.generate_keywords_for_group_specific_grant(
            faculty_ids=[int(faculty_id)],
            opportunity_id=str(opportunity_id),
        )
        if str(out.get("next_action", "")).startswith("error_"):
            return {
                "next_action": "error_generate_keywords_one_to_one_specific_grant",
                **out,
            }
        try:
            upserted = int(
                self.faculty_matcher.run_for_opportunity(
                    opportunity_id=str(opportunity_id),
                    faculty_ids=[int(faculty_id)],
                    min_domain=0.0,
                )
            )
        except Exception as e:
            return {
                "next_action": "error_generate_keywords_one_to_one_specific_grant",
                **out,
                "error": f"{type(e).__name__}: {e}",
            }
        return {
            "next_action": "generated_keywords_one_to_one_specific_grant",
            **out,
            "match_rows_upserted": upserted,
        }

    def generate_keywords_and_matches_for_group_specific_grant(
        self,
        *,
        faculty_ids: List[int],
        opportunity_id: str,
        team_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.generate_keywords_and_matches_for_group_specific_grant")
        clean_faculty_ids = sorted({int(fid) for fid in (faculty_ids or [])})
        out = self.generate_keywords_for_group_specific_grant(
            faculty_ids=clean_faculty_ids,
            opportunity_id=str(opportunity_id),
        )
        if str(out.get("next_action", "")).startswith("error_"):
            return {
                "next_action": "error_generate_keywords_group_specific_grant",
                **out,
            }
        try:
            upserted_required = int(
                self.faculty_matcher.run_for_opportunity(
                    opportunity_id=str(opportunity_id),
                    faculty_ids=clean_faculty_ids,
                    min_domain=0.0,
                )
            )
            upserted_pool = 0
            requested_size = max(int(team_size or 0), len(clean_faculty_ids))
            if requested_size > len(clean_faculty_ids):
                # Build additional candidate pool so team can expand beyond required faculty.
                pool_k = max(requested_size * 25, 200)
                upserted_pool = int(
                    self.faculty_matcher.run_for_opportunity(
                        opportunity_id=str(opportunity_id),
                        faculty_ids=None,
                        k=pool_k,
                        min_domain=0.0,
                    )
                )
            upserted = upserted_required + upserted_pool
        except Exception as e:
            return {
                "next_action": "error_generate_keywords_group_specific_grant",
                **out,
                "error": f"{type(e).__name__}: {e}",
            }
        return {
            "next_action": "generated_keywords_group_specific_grant",
            **out,
            "match_rows_upserted": upserted,
            "team_size_requested": int(team_size) if team_size else None,
        }

    def run_one_to_one_matching(
        self,
        *,
        faculty_id: int,
        top_k: int = 10,
        broad_category: Any = None,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.run_one_to_one_matching")
        source = "match_results_cache"
        out: List[Dict[str, Any]] = []
        faculty_email: Optional[str] = None
        recommendation: Optional[Dict[str, Any]] = None
        recommendation_error: Optional[str] = None
        requested_k = max(int(top_k), 1)
        prefilter_k = max(requested_k * 5, requested_k)

        try:
            with self.session_factory() as sess:
                mdao = MatchDAO(sess)
                fac = sess.get(Faculty, int(faculty_id))
                faculty_email = getattr(fac, "email", None) if fac else None
                cached = mdao.top_matches_for_faculty(int(faculty_id), k=prefilter_k)
                if cached:
                    out = self.context_generator.build_top_match_payload(sess=sess, top_rows=cached)
                else:
                    source = "fresh_compute"

            if source == "fresh_compute":
                # Read from DB first. If empty, generate and read again.
                with self.session_factory() as sess:
                    fac_dao = FacultyDAO(sess)
                    has_faculty_keywords = fac_dao.has_keyword_row(int(faculty_id))
                if not has_faculty_keywords:
                    self.keyword_generator.generate_faculty_keywords_for_id(int(faculty_id))
                self.faculty_matcher.run_for_faculty(
                    faculty_id=int(faculty_id),
                    k=max(prefilter_k, 20),
                    min_domain=0.0,
                )

                with self.session_factory() as sess:
                    mdao = MatchDAO(sess)
                    top = mdao.top_matches_for_faculty(int(faculty_id), k=prefilter_k)
                    out = self.context_generator.build_top_match_payload(sess=sess, top_rows=top)

        except Exception as e:
            return {
                "next_action": "error_one_to_one_matching",
                "error": f"{type(e).__name__}: {e}",
                "matches": [],
            }

        out = self._apply_preference_filters_to_opportunities(
            rows=out,
            top_k=requested_k,
            broad_category=broad_category,
            query_text=query_text,
        )

        if faculty_email and recommendation is None and out:
            try:
                from services.justification.single_justification_generator import SingleJustificationGenerator

                rec_out = SingleJustificationGenerator().generate_faculty_recs(
                    email=faculty_email,
                    k=max(prefilter_k, requested_k),
                )
                # Keep recommendation output aligned with returned matches and top-k.
                order = {
                    str(item.get("opportunity_id") or ""): idx
                    for idx, item in enumerate(out)
                }
                filtered_recs = [
                    r
                    for r in (rec_out.recommendations or [])
                    if str(getattr(r, "opportunity_id", "")) in order
                ]
                filtered_recs.sort(
                    key=lambda r: order.get(str(getattr(r, "opportunity_id", "")), 10**9)
                )
                rec_out = rec_out.model_copy(update={"recommendations": filtered_recs[:requested_k]})
                recommendation = rec_out.model_dump()
            except Exception as e:
                recommendation_error = f"{type(e).__name__}: {e}"

        return {
            "next_action": "return_one_to_one_results",
            "source": source,
            "faculty_email": faculty_email,
            "top_k_grants": int(requested_k),
            "query_text": (str(query_text or "").strip() or None),
            "broad_category_filter": self._normalize_broad_category_for_output(broad_category),
            "matches": out,
            "recommendation": recommendation or {},
            "recommendation_error": recommendation_error,
        }

    def run_one_to_one_matching_with_specific_grant(
        self,
        *,
        faculty_id: int,
        opportunity_id: str,
        top_k_grants: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.run_one_to_one_matching_with_specific_grant")
        opp_id = str(opportunity_id or "").strip()
        requested_k = max(int(top_k_grants or 1), 1)
        if not opp_id:
            return {
                "next_action": "error_one_to_one_specific_grant_matching",
                "error": "Missing opportunity_id.",
                "matches": [],
            }

        source = "specific_grant_match"
        faculty_email: Optional[str] = None
        pair_row: Optional[Dict[str, Any]] = None

        try:
            with self.session_factory() as sess:
                fac = sess.get(Faculty, int(faculty_id))
                faculty_email = getattr(fac, "email", None) if fac else None
                if fac is None:
                    return {
                        "next_action": "error_one_to_one_specific_grant_matching",
                        "error": f"Faculty not found: {faculty_id}",
                        "matches": [],
                    }

                odao = OpportunityDAO(sess)
                opp_ctx = odao.read_opportunity_context(opp_id)
                if not opp_ctx:
                    return {
                        "next_action": "return_one_to_one_results",
                        "source": "specific_grant_not_in_db",
                        "faculty_email": faculty_email,
                        "opportunity_id": opp_id,
                        "matches": [],
                        "top_k_grants": int(requested_k),
                        "recommendation": {},
                        "recommendation_error": None,
                        "note": "Specific grant not found in DB after fetch/search.",
                    }

                mdao = MatchDAO(sess)
                pair_row = mdao.get_match_for_faculty_opportunity(
                    faculty_id=int(faculty_id),
                    opportunity_id=opp_id,
                )

            if pair_row is None:
                source = "fresh_compute_specific_grant"
                with self.session_factory() as sess:
                    fac_dao = FacultyDAO(sess)
                    opp_dao = OpportunityDAO(sess)
                    if not fac_dao.has_keyword_row(int(faculty_id)):
                        self.keyword_generator.generate_faculty_keywords_for_id(int(faculty_id))
                    if not opp_dao.has_keyword_row(opp_id):
                        self.keyword_generator.generate_opportunity_keywords_for_id(opp_id)

                # Generate match rows for this exact grant only.
                self.faculty_matcher.run_for_opportunity(
                    opportunity_id=opp_id,
                    faculty_ids=[int(faculty_id)],
                    min_domain=0.0,
                )

                with self.session_factory() as sess:
                    mdao = MatchDAO(sess)
                    pair_row = mdao.get_match_for_faculty_opportunity(
                        faculty_id=int(faculty_id),
                        opportunity_id=opp_id,
                    )

            if pair_row is None:
                return {
                    "next_action": "return_one_to_one_results",
                    "source": source,
                    "faculty_email": faculty_email,
                    "opportunity_id": opp_id,
                    "matches": [],
                    "top_k_grants": int(requested_k),
                    "recommendation": {},
                    "recommendation_error": None,
                    "note": "No one-to-one match found for the specific grant.",
                }

            with self.session_factory() as sess:
                payload = self.context_generator.build_top_match_payload(
                    sess=sess,
                    top_rows=[
                        (
                            str(pair_row.get("grant_id") or opp_id),
                            float(pair_row.get("domain_score") or 0.0),
                            float(pair_row.get("llm_score") or 0.0),
                        )
                    ],
                )
            payload = list(payload or [])[:requested_k]

            return {
                "next_action": "return_one_to_one_results",
                "source": source,
                "faculty_email": faculty_email,
                "opportunity_id": opp_id,
                "matches": payload,
                "top_k_grants": int(requested_k),
                "recommendation": {},
                "recommendation_error": None,
            }
        except Exception as e:
            return {
                "next_action": "error_one_to_one_specific_grant_matching",
                "error": f"{type(e).__name__}: {e}",
                "matches": [],
            }

    def run_one_to_one_specific_grant_justification(
        self,
        *,
        faculty_email: str,
        opportunity_id: str,
        base_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach single-opportunity justification after specific-grant match retrieval."""
        self._call("MatchingExecutionAgent.run_one_to_one_specific_grant_justification")
        out: Dict[str, Any] = dict(base_result or {})
        if not out:
            return out
        if out.get("next_action") != "return_one_to_one_results":
            return out
        if not list(out.get("matches") or []):
            return out

        email = str(faculty_email or "").strip()
        opp_id = str(opportunity_id or "").strip()
        if not email or not opp_id:
            return out

        try:
            from services.justification.single_justification_generator import SingleJustificationGenerator

            rec_out = SingleJustificationGenerator(
                context_generator=self.context_generator,
            ).generate_for_specific_grant(
                email=email,
                opportunity_id=opp_id,
            )
            out["recommendation"] = rec_out.model_dump()
            out["recommendation_error"] = None
            out["justification_source"] = "specific_grant_llm"
            return out
        except Exception as e:
            out["recommendation"] = out.get("recommendation") or {}
            out["recommendation_error"] = f"{type(e).__name__}: {e}"
            out["justification_source"] = "specific_grant_fallback"
            return out

    def run_group_matching(
        self,
        *,
        faculty_emails: List[str],
        team_size: int = 3,
        top_k_grants: Optional[int] = None,
        broad_category: Any = None,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.run_group_matching")
        try:
            results = self.group_justification_generator.run_justifications_from_group_results(
                faculty_emails=faculty_emails,
                team_size=team_size,
                limit_rows=500,
                include_trace=False,
            )
        except Exception as e:
            return {
                "next_action": "error_group_matching",
                "error": f"{type(e).__name__}: {e}",
                "matches": [],
            }
        results = self._apply_preference_filters_to_group_results(
            rows=results or [],
            broad_category=broad_category,
            query_text=query_text,
        )
        if top_k_grants is not None:
            try:
                k = max(int(top_k_grants), 1)
                results = list(results or [])[:k]
            except Exception:
                pass
        return {
            "next_action": "return_group_matching_results",
            "matches": results or [],
            "team_size": int(team_size),
            "top_k_grants": int(top_k_grants) if top_k_grants is not None else None,
            "query_text": (str(query_text or "").strip() or None),
            "broad_category_filter": self._normalize_broad_category_for_output(broad_category),
        }

    def run_group_matching_with_specific_grant(
        self,
        *,
        faculty_emails: List[str],
        opportunity_id: str,
        team_size: int = 3,
        desired_team_count: int = 3,
        top_k_grants: Optional[int] = None,
        broad_category: Any = None,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.run_group_matching_with_specific_grant")
        try:
            results = self.group_justification_generator.run_justifications_from_group_results(
                faculty_emails=faculty_emails,
                team_size=team_size,
                opp_ids=[opportunity_id],
                limit_rows=500,
                include_trace=False,
            )
        except Exception as e:
            return {
                "next_action": "error_group_specific_grant_matching",
                "error": f"{type(e).__name__}: {e}",
                "matches": [],
            }
        results = self._apply_preference_filters_to_group_results(
            rows=results or [],
            broad_category=broad_category,
            query_text=query_text,
        )
        if top_k_grants is not None:
            try:
                k = max(int(top_k_grants), 1)
                results = list(results or [])[:k]
            except Exception:
                pass
        return {
            "next_action": "return_group_specific_grant_results",
            "matches": results or [],
            "team_size": int(team_size),
            "top_k_grants": int(top_k_grants) if top_k_grants is not None else None,
            "query_text": (str(query_text or "").strip() or None),
            "broad_category_filter": self._normalize_broad_category_for_output(broad_category),
        }

    # ──────────────────────────────────────────────────────────────────
    # Collaborator / team-formation helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_faculty_details(self, faculty_id: int, is_existing: bool = False) -> Optional[Dict[str, Any]]:
        """Return a dict of faculty profile + keyword info for one faculty_id."""
        try:
            from db.models.faculty import Faculty as FacultyModel
            with self.session_factory() as sess:
                fac = sess.get(FacultyModel, int(faculty_id))
                if not fac:
                    return None
                kw_ctx = FacultyDAO(sess).get_faculty_keyword_context(int(faculty_id))
                keywords = (kw_ctx or {}).get("keywords", {})
                return {
                    "faculty_id": int(faculty_id),
                    "name": getattr(fac, "name", None),
                    "email": getattr(fac, "email", None),
                    "position": getattr(fac, "position", None),
                    "expertise": list(getattr(fac, "expertise", None) or []),
                    "research_domains": list((keywords.get("research") or {}).get("domain", [])),
                    "application_domains": list((keywords.get("application") or {}).get("domain", [])),
                    "is_existing_member": is_existing,
                }
        except Exception:
            return None

    def _ensure_opportunity_embedding(self, opportunity_id: str) -> None:
        """Generate opportunity keywords+embedding if not already present."""
        with self.session_factory() as sess:
            if not OpportunityDAO(sess).has_keyword_row(opportunity_id):
                self.keyword_generator.generate_opportunity_keywords_for_id(opportunity_id)

    # LLM pre-filter pool multiplier: score this many candidates via LLM, keep top-N
    LLM_POOL_MULTIPLIER = 4
    LLM_POOL_MIN = 15

    def _run_llm_scoring_for_candidates(
        self,
        opp_id: str,
        candidate_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run FacultyGrantMatcher for a batch of candidates against one opportunity,
        then read the stored MatchResult rows.

        Returns {faculty_id: {llm_score, domain_score, reason, covered, missing}}.
        """
        if not candidate_ids:
            return {}

        # FacultyGrantMatcher.run_for_opportunity computes + stores match rows.
        self.faculty_matcher.run_for_opportunity(
            opportunity_id=opp_id,
            faculty_ids=candidate_ids,
            min_domain=0.0,
        )

        scores: Dict[int, Dict[str, Any]] = {}
        with self.session_factory() as sess:
            mdao = MatchDAO(sess)
            for fid in candidate_ids:
                row = mdao.get_match_for_faculty_opportunity(
                    faculty_id=int(fid),
                    opportunity_id=opp_id,
                )
                if row:
                    scores[int(fid)] = {
                        "llm_score":    round(float(row.get("llm_score")    or 0), 3),
                        "domain_score": round(float(row.get("domain_score") or 0), 3),
                        "reason":   str(row.get("reason")  or ""),
                        "covered":  list(row.get("covered") or []),
                        "missing":  list(row.get("missing") or []),
                    }
        return scores

    def find_collaborators_for_grant(
        self,
        *,
        opportunity_id: str,
        existing_faculty_ids: Optional[List[int]] = None,
        additional_count: int = 3,
    ) -> Dict[str, Any]:
        """
        Find top `additional_count` faculty who would complement the existing team
        for a specific grant.

        Two-phase approach:
          1. Fast embedding pre-filter — topk_faculties_for_opportunity (pgvector)
          2. LLM scoring — FacultyGrantMatcher on the top-N*LLM_POOL candidates
          3. Re-rank by llm_score; return top additional_count with rich output.

        Each result includes: name, email, position, expertise, research/application
        domains, domain_score, llm_score, reason (one-line LLM justification),
        covered topics, and missing topics.
        """
        self._call("MatchingExecutionAgent.find_collaborators_for_grant")
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            return {
                "next_action": "error_find_collaborators",
                "error": "Missing opportunity_id.",
                "suggested_collaborators": [],
            }

        exclude_ids: Set[int] = set(map(int, existing_faculty_ids or []))

        try:
            # ── Phase 0: ensure opportunity has keyword embeddings ─────────
            self._ensure_opportunity_embedding(opp_id)

            from db.models.opportunity import Opportunity as OppModel
            with self.session_factory() as sess:
                opp = sess.get(OppModel, opp_id)
                opp_title = getattr(opp, "opportunity_title", None) if opp else None

            # ── Phase 1: fast embedding pre-filter ────────────────────────
            llm_pool_size = max(additional_count * self.LLM_POOL_MULTIPLIER, self.LLM_POOL_MIN)
            k_fetch = llm_pool_size + len(exclude_ids)  # fetch extra to cover exclusions
            with self.session_factory() as sess:
                top_pairs = MatchDAO(sess).topk_faculties_for_opportunity(
                    opportunity_id=opp_id, k=k_fetch
                )

            candidate_ids = [
                fid for fid, _ in top_pairs if fid not in exclude_ids
            ][:llm_pool_size]

            # ── Phase 2: LLM scoring for the candidate pool ───────────────
            scores = self._run_llm_scoring_for_candidates(opp_id, candidate_ids)

            # ── Phase 3: re-rank by llm_score ─────────────────────────────
            ranked = sorted(
                candidate_ids,
                key=lambda fid: (
                    scores.get(fid, {}).get("llm_score", 0),
                    scores.get(fid, {}).get("domain_score", 0),
                ),
                reverse=True,
            )
            top_n = ranked[:additional_count]

            # ── Phase 4: enrich with faculty profile ──────────────────────
            suggested = []
            for fid in top_n:
                details = self._get_faculty_details(int(fid), is_existing=False)
                if details:
                    details.update(scores.get(int(fid), {}))
                    suggested.append(details)

            return {
                "next_action": "return_collaborators",
                "opportunity_id": opp_id,
                "opportunity_title": opp_title,
                "additional_count": additional_count,
                "suggested_collaborators": suggested,
            }

        except Exception as e:
            return {
                "next_action": "error_find_collaborators",
                "error": f"{type(e).__name__}: {e}",
                "suggested_collaborators": [],
            }

    def find_team_for_grant(
        self,
        *,
        opportunity_id: str,
        team_size: int = 3,
        existing_faculty_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Suggest a team of `team_size` faculty for a grant.

        Two-phase approach:
          1. Fast embedding pre-filter — topk_faculties_for_opportunity
          2. LLM scoring — FacultyGrantMatcher on the top candidate pool
          3. Re-rank new candidates by llm_score; always keep existing members.

        Existing members provided by the caller are always included and marked
        ``is_existing_member: True``.  Remaining slots are filled by the
        top-scoring new candidates from the DB.

        Each result includes: name, email, position, expertise, research/application
        domains, domain_score, llm_score, reason, covered topics, missing topics,
        and is_existing_member flag.
        """
        self._call("MatchingExecutionAgent.find_team_for_grant")
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            return {
                "next_action": "error_find_team",
                "error": "Missing opportunity_id.",
                "suggested_team": [],
            }

        existing_ids: Set[int] = set(map(int, existing_faculty_ids or []))

        try:
            # ── Phase 0: ensure opportunity has keyword embeddings ─────────
            self._ensure_opportunity_embedding(opp_id)

            from db.models.opportunity import Opportunity as OppModel
            with self.session_factory() as sess:
                opp = sess.get(OppModel, opp_id)
                opp_title = getattr(opp, "opportunity_title", None) if opp else None

            # ── Phase 1: fast embedding pre-filter ────────────────────────
            remaining_slots = max(0, team_size - len(existing_ids))
            llm_pool_size = max(remaining_slots * self.LLM_POOL_MULTIPLIER, self.LLM_POOL_MIN)
            k_fetch = llm_pool_size + len(existing_ids)
            with self.session_factory() as sess:
                top_pairs = MatchDAO(sess).topk_faculties_for_opportunity(
                    opportunity_id=opp_id, k=k_fetch
                )

            # Separate: existing members in top results vs new candidates
            existing_in_top = [(fid, sc) for fid, sc in top_pairs if fid in existing_ids]
            new_candidates  = [fid for fid, _ in top_pairs if fid not in existing_ids]
            candidate_pool  = new_candidates[:llm_pool_size]

            # Existing members not returned by embedding query (no embedding yet)
            seen = {fid for fid, _ in existing_in_top}
            extra_existing_ids = [eid for eid in existing_ids if eid not in seen]

            # ── Phase 2: LLM scoring — new candidates + existing members ──
            all_to_score = candidate_pool + extra_existing_ids + [fid for fid, _ in existing_in_top]
            # also score existing members so we return their llm_score/reason
            scores = self._run_llm_scoring_for_candidates(opp_id, all_to_score)

            # ── Phase 3: re-rank new candidates by llm_score ──────────────
            ranked_new = sorted(
                candidate_pool,
                key=lambda fid: (
                    scores.get(fid, {}).get("llm_score", 0),
                    scores.get(fid, {}).get("domain_score", 0),
                ),
                reverse=True,
            )
            top_new = ranked_new[:remaining_slots]

            # ── Phase 4: assemble final team ──────────────────────────────
            # existing first (sorted by llm_score), then top new
            existing_order = sorted(
                list(existing_ids),
                key=lambda fid: scores.get(fid, {}).get("llm_score", 0),
                reverse=True,
            )
            final_team_ids = existing_order + top_new

            # ── Phase 5: enrich with faculty profile ──────────────────────
            suggested_team = []
            for fid in final_team_ids:
                is_existing = int(fid) in existing_ids
                details = self._get_faculty_details(int(fid), is_existing=is_existing)
                if details:
                    details.update(scores.get(int(fid), {}))
                    suggested_team.append(details)

            return {
                "next_action": "return_team",
                "opportunity_id": opp_id,
                "opportunity_title": opp_title,
                "team_size": team_size,
                "suggested_team": suggested_team,
            }

        except Exception as e:
            return {
                "next_action": "error_find_team",
                "error": f"{type(e).__name__}: {e}",
                "suggested_team": [],
            }
