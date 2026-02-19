from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config import get_llm_client
from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from services.context.context_generator import ContextGenerator
from services.justification.group_justification_generator import GroupJustificationGenerator
from services.keywords.keyword_generator import KeywordGenerator
from services.matching.faculty_grant_matcher import FacultyGrantMatcher
from services.opportunity.call_opportunity import OpportunitySearchService


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

    def ask_for_email(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_email")
        return {"next_action": "ask_email"}

    def ask_for_group_emails(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_group_emails")
        return {"next_action": "ask_group_emails"}

    def ask_for_user_reference_data(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_user_reference_data")
        return {"next_action": "ask_user_reference_data"}


class OpportunityContextAgent:
    UUID_RE = re.compile(r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b")

    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        search_service: Optional[OpportunitySearchService] = None,
    ):
        self.session_factory = session_factory
        self.search_service = search_service or OpportunitySearchService()

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    def _extract_opp_id_from_link(self, grant_link: str) -> Optional[str]:
        raw = (grant_link or "").strip()
        if not raw:
            return None
        if self.UUID_RE.fullmatch(raw):
            return raw
        m = self.UUID_RE.search(raw)
        if m:
            return m.group(1)
        m2 = re.search(r"/opportunity/([^/?#]+)", raw)
        if m2:
            return m2.group(1).strip()
        # Allow explicit opportunity_id style tokens if they are not UUID-like.
        if "/" not in raw and len(raw) <= 128:
            return raw
        return None

    def _upsert_opportunities(self, opps: List[Any]) -> Optional[str]:
        if not opps:
            return None
        with self.session_factory() as sess:
            odao = OpportunityDAO(sess)
            for opp in opps:
                odao.upsert_opportunity(opp)
                odao.upsert_attachments(opp.opportunity_id, opp.attachments or [])
                odao.upsert_additional_info(opp.opportunity_id, opp.additional_info or [])
            sess.commit()
        return getattr(opps[0], "opportunity_id", None)

    def ask_for_valid_grant_link(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.ask_for_valid_grant_link")
        return {"next_action": "ask_valid_grant_link"}

    def ask_for_grant_identifier(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.ask_for_grant_identifier")
        return {"next_action": "ask_grant_identifier"}

    def search_grant_by_link_in_db(self, *, grant_link: str) -> Dict[str, Any]:
        self._call("OpportunityContextAgent.search_grant_by_link_in_db")
        opp_id = self._extract_opp_id_from_link(grant_link)
        if not opp_id:
            return {"found": False, "opportunity_id": None, "opportunity_title": None}
        try:
            with self.session_factory() as sess:
                odao = OpportunityDAO(sess)
                ctx = odao.read_opportunity_context(opp_id)
                if not ctx:
                    return {"found": False, "opportunity_id": opp_id, "opportunity_title": None}
                return {
                    "found": True,
                    "opportunity_id": ctx.get("opportunity_id") or opp_id,
                    "opportunity_title": ctx.get("title"),
                }
        except Exception as e:
            return {"found": False, "opportunity_id": opp_id, "opportunity_title": None, "error": f"{type(e).__name__}: {e}"}

    def search_grant_by_title_in_db(self, *, grant_title: str) -> Dict[str, Any]:
        self._call("OpportunityContextAgent.search_grant_by_title_in_db")
        title = (grant_title or "").strip()
        if not title:
            return {"found": False, "opportunity_id": None, "opportunity_title": None}
        try:
            with self.session_factory() as sess:
                odao = OpportunityDAO(sess)
                opp = odao.find_opportunity_by_title(title)
                if not opp:
                    return {"found": False, "opportunity_id": None, "opportunity_title": None}
                return {
                    "found": True,
                    "opportunity_id": opp.opportunity_id,
                    "opportunity_title": opp.opportunity_title,
                }
        except Exception as e:
            return {"found": False, "opportunity_id": None, "opportunity_title": None, "error": f"{type(e).__name__}: {e}"}

    def fetch_grant_from_source(
        self,
        *,
        grant_identifier_type: Optional[str],
        grant_link: Optional[str] = None,
        grant_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._call("OpportunityContextAgent.fetch_grant_from_source")
        try:
            if grant_identifier_type == "link":
                opp_id = self._extract_opp_id_from_link(grant_link or "")
                if not opp_id:
                    return {"fetched": False, "opportunity_id": None, "opportunity_title": None}
                opps = self.search_service.run_search_pipeline(opportunity_id=opp_id)
                saved_opp_id = self._upsert_opportunities(opps)
                if not saved_opp_id:
                    return {"fetched": False, "opportunity_id": opp_id, "opportunity_title": None}
                return {
                    "fetched": True,
                    "opportunity_id": saved_opp_id,
                    "opportunity_title": getattr(opps[0], "opportunity_title", None) if opps else None,
                }

            title = (grant_title or "").strip()
            if not title:
                return {"fetched": False, "opportunity_id": None, "opportunity_title": None}
            opps = self.search_service.run_search_pipeline(q=title, page_size=20)
            if not opps:
                return {"fetched": False, "opportunity_id": None, "opportunity_title": None}
            target = None
            for opp in opps:
                opp_title = (getattr(opp, "opportunity_title", "") or "").strip().lower()
                if opp_title == title.lower():
                    target = opp
                    break
            target = target or opps[0]
            saved_opp_id = self._upsert_opportunities([target])
            return {
                "fetched": bool(saved_opp_id),
                "opportunity_id": saved_opp_id,
                "opportunity_title": getattr(target, "opportunity_title", None),
            }
        except Exception:
            return {"fetched": False, "opportunity_id": None, "opportunity_title": None}


class GeneralConversationAgent:
    def __init__(self):
        self.llm = None

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _as_text(resp: Any) -> str:
        content = getattr(resp, "content", resp)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    txt = str(item.get("text") or "").strip()
                    if txt:
                        parts.append(txt)
            return " ".join(parts).strip()
        return str(content).strip()

    def answer_briefly(self, *, user_input: str) -> Dict[str, Any]:
        self._call("GeneralConversationAgent.answer_briefly")
        header = "[Grant Match Assistant]"
        try:
            if self.llm is None:
                self.llm = get_llm_client().build()
            prompt = (
                "You are a specialized grant-matching assistant. "
                "If user asks non-grant/general question, answer very briefly in 1-2 sentences. "
                "Keep it helpful and concise."
            )
            resp = self.llm.invoke([("system", prompt), ("human", user_input or "")])
            msg = self._as_text(resp) or "I focus on grant matching. Ask me about grant search or faculty-grant fit."
            return {"next_action": "general_reply", "message": f"{header} {msg}"}
        except Exception:
            return {
                "next_action": "general_reply",
                "message": f"{header} I focus on grant matching. Please ask about grants or matching.",
            }


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
        broad_category: Optional[str],
        query_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        category_filter = str(broad_category or "").strip().lower() or None
        if not category_filter and not str(query_text or "").strip():
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
                    if category_filter and broad != category_filter:
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
        broad_category: Optional[str],
        query_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        category_filter = str(broad_category or "").strip().lower() or None
        if not category_filter and not str(query_text or "").strip():
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
                    if category_filter and broad != category_filter:
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
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.generate_keywords_and_matches_for_group_specific_grant")
        clean_faculty_ids = [int(fid) for fid in (faculty_ids or [])]
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
            upserted = int(
                self.faculty_matcher.run_for_opportunity(
                    opportunity_id=str(opportunity_id),
                    faculty_ids=clean_faculty_ids,
                    min_domain=0.0,
                )
            )
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
        }

    def run_one_to_one_matching(
        self,
        *,
        faculty_id: int,
        top_k: int = 10,
        broad_category: Optional[str] = None,
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
                if broad_category or query_text:
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
            "query_text": (str(query_text or "").strip() or None),
            "broad_category_filter": (str(broad_category or "").strip().lower() or None),
            "matches": out,
            "recommendation": recommendation or {},
            "recommendation_error": recommendation_error,
        }

    def run_one_to_one_matching_with_specific_grant(
        self,
        *,
        faculty_id: int,
        opportunity_id: str,
    ) -> Dict[str, Any]:
        self._call("MatchingExecutionAgent.run_one_to_one_matching_with_specific_grant")
        opp_id = str(opportunity_id or "").strip()
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

            return {
                "next_action": "return_one_to_one_results",
                "source": source,
                "faculty_email": faculty_email,
                "opportunity_id": opp_id,
                "matches": payload,
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
        broad_category: Optional[str] = None,
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
        return {
            "next_action": "return_group_matching_results",
            "matches": results or [],
            "team_size": int(team_size),
            "query_text": (str(query_text or "").strip() or None),
            "broad_category_filter": (str(broad_category or "").strip().lower() or None),
        }

    def run_group_matching_with_specific_grant(
        self,
        *,
        faculty_emails: List[str],
        opportunity_id: str,
        team_size: int = 3,
        desired_team_count: int = 3,
        broad_category: Optional[str] = None,
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
        return {
            "next_action": "return_group_specific_grant_results",
            "matches": results or [],
            "team_size": int(team_size),
            "query_text": (str(query_text or "").strip() or None),
            "broad_category_filter": (str(broad_category or "").strip().lower() or None),
        }
