from __future__ import annotations

from typing import Any, Dict

from db.models import Faculty
from utils.content_extractor import load_extracted_content


class FacultyContextBuilder:

    def build_faculty_basic_context(self, fac: Faculty) -> Dict[str, Any]:
        pubs = sorted(
            fac.publications or [],
            key=lambda p: (p.year or 0),
            reverse=True,
        )
        return {
            "name": fac.name,
            "position": fac.position,
            "organization": fac.organization,
            "email": fac.email,
            "biography": fac.biography,
            "expertise": fac.expertise or [],
            "degrees": fac.degrees or [],
            "additional_infos": load_extracted_content(
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

    def build_faculty_keyword_context(self, fac: Faculty) -> Dict[str, Any]:
        kw = (getattr(fac, "keyword", None) and getattr(fac.keyword, "keywords", None)) or {}
        return {
            "faculty_id": fac.faculty_id,
            "name": fac.name,
            "email": fac.email,
            "profile_url": fac.source_url,
            "keywords": kw,
        }
