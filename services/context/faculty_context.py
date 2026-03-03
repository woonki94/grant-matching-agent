from __future__ import annotations

from typing import Any, Dict, List, Tuple

from db.models import Faculty
from utils.content_extractor import load_extracted_content


class FacultyContextBuilder:
    PROFILE_FIELDS: Dict[str, Tuple[str, ...]] = {
        "basic": (
            "name",
            "position",
            "organization",
            "email",
            "biography",
            "expertise",
            "degrees",
            "additional_infos",
            "publications",
        ),
        "keyword": (
            "faculty_id",
            "name",
            "email",
            "profile_url",
            "keywords",
        ),
    }

    @staticmethod
    def _sorted_publications(fac: Faculty) -> List[Any]:
        return sorted(
            fac.publications or [],
            key=lambda p: (p.year or 0),
            reverse=True,
        )

    @staticmethod
    def _select_fields(payload: Dict[str, Any], fields: Tuple[str, ...]) -> Dict[str, Any]:
        return {k: payload.get(k) for k in fields}

    def build_faculty_retrievable_context(self, fac: Faculty) -> Dict[str, Any]:
        pubs = self._sorted_publications(fac)
        kw = (getattr(fac, "keyword", None) and getattr(fac.keyword, "keywords", None)) or {}
        context = {
            key: getattr(fac, key, None)
            for key in ("faculty_id", "name", "position", "organization", "email", "biography")
        }
        context["profile_url"] = fac.source_url
        context["expertise"] = fac.expertise or []
        context["degrees"] = fac.degrees or []
        context["additional_infos"] = load_extracted_content(
            fac.additional_info,
            url_attr="additional_info_url",
        )
        context["publications"] = [
            {
                "title": p.title,
                "year": p.year,
                "abstract": p.abstract,
            }
            for p in pubs
        ]
        context["keywords"] = kw
        return context

    def build_faculty_context(self, fac: Faculty, *, profile: str = "basic") -> Dict[str, Any]:
        full = self.build_faculty_retrievable_context(fac)
        normalized = str(profile or "").strip().lower()
        if normalized == "full":
            return full
        fields = self.PROFILE_FIELDS.get(normalized)
        if not fields:
            raise ValueError(f"Unsupported faculty context profile: {profile}")
        return self._select_fields(full, fields)

