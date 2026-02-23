from __future__ import annotations

from typing import Any, Dict

from db.models import Opportunity
from utils.content_extractor import load_extracted_content


class OpportunityContextBuilder:
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
