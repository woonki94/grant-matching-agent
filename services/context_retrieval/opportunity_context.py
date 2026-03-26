from __future__ import annotations

from typing import Any, Dict, List, Tuple

from db.models import Opportunity
from services.context_retrieval.rag_chunk_retriever import retrieve_opportunity_supporting_chunks
from utils.content_extractor import load_extracted_content


class OpportunityContextBuilder:
    PROFILE_FIELDS: Dict[str, Tuple[str, ...]] = {
        "basic": (
            "opportunity_title",
            "agency_name",
            "category",
            "opportunity_status",
            "summary_description",
        ),
        "keyword": (
            "opportunity_id",
            "opportunity_title",
            "agency_name",
            "opportunity_link",
            "keywords",
        ),
        "explanation": (
            "opportunity_id",
            "opportunity_title",
            "agency_name",
            "category",
            "opportunity_status",
            "summary_description",
            "keywords",
        ),
    }

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _short_text(text: Any, *, max_chars: int) -> str:
        s = OpportunityContextBuilder._normalize_text(text)
        if not s:
            return ""
        cap = max(int(max_chars), 0)
        if cap == 0:
            return ""
        if len(s) <= cap:
            return s
        clipped = s[:cap].rstrip()
        sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if sentence_end >= int(cap * 0.5):
            return clipped[: sentence_end + 1].rstrip()
        word_end = clipped.rfind(" ")
        if word_end >= int(cap * 0.6):
            return clipped[:word_end].rstrip()
        return clipped

    @staticmethod
    def _brief_keyword_bucket(values: Any, *, limit: int) -> List[str]:
        out: List[str] = []
        for x in list(values or []):
            s = str(x).strip()
            if not s:
                continue
            out.append(s)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _select_fields(payload: Dict[str, Any], fields: Tuple[str, ...]) -> Dict[str, Any]:
        return {k: payload.get(k) for k in fields}

    @staticmethod
    def _first_additional_info_link(opp: Opportunity) -> Any:
        infos = list(getattr(opp, "additional_info", None) or [])
        if not infos:
            return None
        return getattr(infos[0], "additional_info_url", None)

    def build_opportunity_retrievable_context(self, opp: Opportunity) -> Dict[str, Any]:
        kw = (getattr(opp, "keyword", None) and getattr(opp.keyword, "keywords", None)) or {}
        return {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "opportunity_title": getattr(opp, "opportunity_title", None),
            "agency_name": getattr(opp, "agency_name", None),
            "category": getattr(opp, "category", None),
            "opportunity_status": getattr(opp, "opportunity_status", None),
            "summary_description": getattr(opp, "summary_description", None),
            "opportunity_link": self._first_additional_info_link(opp),
            "keywords": kw,
        }

    def build_opportunity_context(
        self,
        opp: Opportunity,
        *,
        profile: str = "basic",
        max_summary_chars: int = 420,
        max_item_chars: int = 240,
        max_additional_items: int = 2,
        max_attachment_items: int = 2,
        max_keywords_per_bucket: int = 6,
    ) -> Dict[str, Any]:
        _ = max_summary_chars
        full = self.build_opportunity_retrievable_context(opp)
        normalized = str(profile or "").strip().lower()
        fields = self.PROFILE_FIELDS.get(normalized)
        if not fields:
            raise ValueError(f"Unsupported opportunity context profile: {profile}")
        context = self._select_fields(full, fields)

        if normalized == "keyword":
            return context

        rag = retrieve_opportunity_supporting_chunks(
            opp,
            top_k_per_additional_source=4,
            top_k_per_attachment_source=4,
        )
        additional_blocks = list(rag.get("additional_info_chunks") or [])
        attachment_blocks = list(rag.get("attachment_chunks") or [])

        if not additional_blocks:
            additional_blocks = load_extracted_content(
                opp.additional_info,
                url_attr="additional_info_url",
                group_chunks=False,
                include_row_meta=True,
            )
        if not attachment_blocks:
            attachment_blocks = load_extracted_content(
                opp.attachments,
                url_attr="file_download_path",
                title_attr="file_name",
                group_chunks=False,
                include_row_meta=True,
            )

        if normalized == "basic":
            context["additional_info_extracted"] = additional_blocks
            context["attachments_extracted"] = attachment_blocks
            return context

        kw = context.get("keywords") if isinstance(context, dict) else {}
        research_kw = kw.get("research") if isinstance(kw, dict) else {}
        application_kw = kw.get("application") if isinstance(kw, dict) else {}

        additional_info_brief = []
        for item in list(additional_blocks or [])[:max_additional_items]:
            additional_info_brief.append(
                {
                    "url": item.get("url"),
                    "excerpt": self._short_text(item.get("content"), max_chars=max_item_chars),
                }
            )

        attachments_brief = []
        for item in list(attachment_blocks or [])[:max_attachment_items]:
            attachments_brief.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "excerpt": self._short_text(item.get("content"), max_chars=max_item_chars),
                }
            )

        context["summary_description"] = self._normalize_text(context.get("summary_description"))
        context["keywords"] = {
            "research": {
                "domain": self._brief_keyword_bucket(
                    (research_kw or {}).get("domain"),
                    limit=max_keywords_per_bucket,
                ),
                "specialization": self._brief_keyword_bucket(
                    (research_kw or {}).get("specialization"),
                    limit=max_keywords_per_bucket,
                ),
            },
            "application": {
                "domain": self._brief_keyword_bucket(
                    (application_kw or {}).get("domain"),
                    limit=max_keywords_per_bucket,
                ),
                "specialization": self._brief_keyword_bucket(
                    (application_kw or {}).get("specialization"),
                    limit=max_keywords_per_bucket,
                ),
            },
        }
        context["additional_info_brief"] = additional_info_brief
        context["attachments_brief"] = attachments_brief
        return context

    def build_opportunity_basic_context(self, opp: Opportunity) -> Dict[str, Any]:
        return self.build_opportunity_context(opp, profile="basic")

    def build_opportunity_keyword_context(self, opp: Opportunity) -> Dict[str, Any]:
        return self.build_opportunity_context(opp, profile="keyword")

    def build_opportunity_explanation_context(
        self,
        opp: Opportunity,
        *,
        max_summary_chars: int = 420,
        max_item_chars: int = 240,
        max_additional_items: int = 2,
        max_attachment_items: int = 2,
        max_keywords_per_bucket: int = 6,
    ) -> Dict[str, Any]:
        return self.build_opportunity_context(
            opp,
            profile="explanation",
            max_summary_chars=max_summary_chars,
            max_item_chars=max_item_chars,
            max_additional_items=max_additional_items,
            max_attachment_items=max_attachment_items,
            max_keywords_per_bucket=max_keywords_per_bucket,
        )
