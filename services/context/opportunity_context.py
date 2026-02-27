from __future__ import annotations

from typing import Any, Dict, List

from db.models import Opportunity
from utils.content_extractor import load_extracted_content


class OpportunityContextBuilder:
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
        kw = (getattr(opp, "keyword", None) and getattr(opp.keyword, "keywords", None)) or {}
        research_kw = kw.get("research") if isinstance(kw, dict) else {}
        application_kw = kw.get("application") if isinstance(kw, dict) else {}

        additional_blocks = load_extracted_content(
            opp.additional_info,
            url_attr="additional_info_url",
        )
        attachment_blocks = load_extracted_content(
            opp.attachments,
            url_attr="file_download_path",
            title_attr="file_name",
        )

        additional_short = []
        for item in list(additional_blocks or [])[:max_additional_items]:
            additional_short.append(
                {
                    "url": item.get("url"),
                    "excerpt": self._short_text(item.get("content"), max_chars=max_item_chars),
                }
            )

        attachment_short = []
        for item in list(attachment_blocks or [])[:max_attachment_items]:
            attachment_short.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "excerpt": self._short_text(item.get("content"), max_chars=max_item_chars),
                }
            )

        return {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "opportunity_title": getattr(opp, "opportunity_title", None),
            "agency_name": getattr(opp, "agency_name", None),
            "category": getattr(opp, "category", None),
            "opportunity_status": getattr(opp, "opportunity_status", None),
            # Keep summary untruncated; brevity is enforced by the LLM prompt.
            "summary_description": self._normalize_text(getattr(opp, "summary_description", None)),
            "keywords": {
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
            },
            "additional_info_brief": additional_short,
            "attachments_brief": attachment_short,
        }

    def build_opportunity_basic_context(self, opp: Opportunity) -> Dict[str, Any]:
        return {
            "opportunity_title": opp.opportunity_title,
            "agency_name": opp.agency_name,
            "category": opp.category,
            "opportunity_status": opp.opportunity_status,
            "summary_description": opp.summary_description,
            "additional_info_extracted": load_extracted_content(
                opp.additional_info,
                url_attr="additional_info_url",
            ),
            "attachments_extracted": load_extracted_content(
                opp.attachments,
                url_attr="file_download_path",
                title_attr="file_name",
            ),
        }

    def build_opportunity_keyword_context(self, opp: Opportunity) -> Dict[str, Any]:
        kw = (getattr(opp, "keyword", None) and getattr(opp.keyword, "keywords", None)) or {}
        first_link = None
        infos = list(getattr(opp, "additional_info", None) or [])
        if infos:
            first_link = getattr(infos[0], "additional_info_url", None)
        return {
            "opportunity_id": opp.opportunity_id,
            "opportunity_title": opp.opportunity_title,
            "agency_name": opp.agency_name,
            "opportunity_link": first_link,
            "keywords": kw,
        }
