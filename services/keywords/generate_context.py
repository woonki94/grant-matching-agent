from typing import Dict, Any

from db.models import Faculty, Opportunity
from utils.content_extractor import load_extracted_content


def opportunity_to_keyword_context(opp: Opportunity,) -> Dict[str, Any]:
    return {
        "opportunity_title": opp.opportunity_title,
        "agency_name": opp.agency_name,
        "category": opp.category,
        "opportunity_status": opp.opportunity_status,
        "summary_description": opp.summary_description,
        # extracted text only when it exists on disk
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

def faculty_to_keyword_context(fac: Faculty) -> dict:
    pubs = sorted(
        fac.publications or [],
        key=lambda p: (p.year or 0),
        reverse=True
    )
    return {
        "name": fac.name,
        "position": fac.position,
        "organization": fac.organization,
        "email": fac.email,
        "biography": fac.biography,
        "expertise": fac.expertise or [],
        "degrees": fac.degrees or [],
        "additional_infos":  load_extracted_content(
            fac.additional_info,
            url_attr="additional_info_url",
        ),
        "publications": [
            {
                "title": p.title,
                "year": p.year,
                "abstract": p.abstract,
            }
            for p in pubs
        ],
    }
