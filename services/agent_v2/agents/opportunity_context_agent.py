from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.opportunity.call_opportunity import OpportunitySearchService


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


