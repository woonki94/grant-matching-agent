from __future__ import annotations

from typing import Any, Dict, List, Tuple

from db.models import Faculty
from services.context_retrieval.rag_chunk_retriever import retrieve_faculty_additional_info_chunks
from utils.content_extractor import load_extracted_content


class FacultyContextBuilder:
    DEFAULT_ADDITIONAL_INFO_CHUNKS_PER_SOURCE = 5
    DEFAULT_RECENT_PUB_TITLES_FOR_QUERY = 5

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

    def build_faculty_retrievable_context(
        self,
        fac: Faculty,
        *,
        include_additional_infos: bool = True,
    ) -> Dict[str, Any]:
        pubs = self._sorted_publications(fac)
        kw = (getattr(fac, "keyword", None) and getattr(fac.keyword, "keywords", None)) or {}
        context = {
            key: getattr(fac, key, None)
            for key in ("faculty_id", "name", "position", "organization", "email", "biography")
        }
        context["profile_url"] = fac.source_url
        context["expertise"] = fac.expertise or []
        context["degrees"] = fac.degrees or []
        additional_infos: List[Dict[str, Any]] = []
        if include_additional_infos:
            rag = retrieve_faculty_additional_info_chunks(
                fac,
                top_k_per_source=self.DEFAULT_ADDITIONAL_INFO_CHUNKS_PER_SOURCE,
                max_recent_pub_titles=self.DEFAULT_RECENT_PUB_TITLES_FOR_QUERY,
            )
            additional_infos = list(rag.get("additional_info_chunks") or [])
            if not additional_infos:
                additional_infos = load_extracted_content(
                    fac.additional_info,
                    url_attr="additional_info_url",
                    group_chunks=False,
                    include_row_meta=True,
                )
        context["additional_infos"] = additional_infos
        context["publications"] = [
            {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "abstract": p.abstract,
            }
            for p in pubs
        ]
        context["keywords"] = kw
        return context

    def build_faculty_context(self, fac: Faculty, *, profile: str = "basic") -> Dict[str, Any]:
        normalized = str(profile or "").strip().lower()
        include_additional_infos = normalized in ("basic", "full")
        full = self.build_faculty_retrievable_context(
            fac,
            include_additional_infos=include_additional_infos,
        )
        if normalized == "full":
            return full
        fields = self.PROFILE_FIELDS.get(normalized)
        if not fields:
            raise ValueError(f"Unsupported faculty context profile: {profile}")
        return self._select_fields(full, fields)
