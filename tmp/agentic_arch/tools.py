from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from tmp.agentic_arch.models import FacultyBasicInfo, FacultyPublication, GrantMetadata


class FacultyTools(Protocol):
    async def fetch_basic_info(self, email: str) -> FacultyBasicInfo: ...

    async def fetch_keywords(self, email: str) -> List[str]: ...

    async def fetch_additional_text(self, email: str, max_items: int = 3) -> List[str]: ...

    async def fetch_publications(self, email: str, max_items: int = 10) -> List[FacultyPublication]: ...


class GrantTools(Protocol):
    async def search_candidate_grants(self, profession_focus: List[str], top_k: int = 20) -> List[str]: ...

    async def fetch_metadata(self, grant_id: str) -> GrantMetadata: ...

    async def fetch_requirement_domains(self, grant_id: str) -> List[str]: ...

    async def fetch_requirement_specializations(self, grant_id: str) -> List[str]: ...

    async def fetch_requirement_eligibility(self, grant_id: str) -> List[str]: ...

    async def fetch_requirement_deliverables(self, grant_id: str) -> List[str]: ...


@dataclass(frozen=True)
class InMemoryFacultyRecord:
    basic_info: FacultyBasicInfo
    keywords: List[str]
    additional_text: List[str]
    publications: List[FacultyPublication]


@dataclass(frozen=True)
class InMemoryGrantRecord:
    metadata: GrantMetadata
    domains: List[str]
    specializations: List[str]
    eligibility: List[str]
    deliverables: List[str]


class InMemoryFacultyTools(FacultyTools):
    def __init__(self, records: Dict[str, InMemoryFacultyRecord]):
        self._records = {str(k).strip().lower(): v for k, v in (records or {}).items()}

    async def fetch_basic_info(self, email: str) -> FacultyBasicInfo:
        await asyncio.sleep(0)
        key = str(email or "").strip().lower()
        row = self._records.get(key)
        if not row:
            raise ValueError(f"Faculty not found for email={key}")
        return row.basic_info

    async def fetch_keywords(self, email: str) -> List[str]:
        await asyncio.sleep(0)
        key = str(email or "").strip().lower()
        row = self._records.get(key)
        return list(row.keywords) if row else []

    async def fetch_additional_text(self, email: str, max_items: int = 3) -> List[str]:
        await asyncio.sleep(0)
        key = str(email or "").strip().lower()
        row = self._records.get(key)
        if not row:
            return []
        return list(row.additional_text[: max(0, int(max_items or 0))])

    async def fetch_publications(self, email: str, max_items: int = 10) -> List[FacultyPublication]:
        await asyncio.sleep(0)
        key = str(email or "").strip().lower()
        row = self._records.get(key)
        if not row:
            return []
        return list(row.publications[: max(0, int(max_items or 0))])


class InMemoryGrantTools(GrantTools):
    def __init__(self, grants: Dict[str, InMemoryGrantRecord]):
        self._grants = {str(k).strip(): v for k, v in (grants or {}).items()}

    async def search_candidate_grants(self, profession_focus: List[str], top_k: int = 20) -> List[str]:
        await asyncio.sleep(0)
        top = max(1, int(top_k or 20))
        terms = [str(x).strip().lower() for x in (profession_focus or []) if str(x).strip()]
        if not terms:
            return list(self._grants.keys())[:top]

        def _score(grant_id: str, row: InMemoryGrantRecord) -> int:
            bag = " ".join(
                [
                    str(row.metadata.grant_name or "").lower(),
                    " ".join(str(x).lower() for x in (row.domains or [])),
                    " ".join(str(x).lower() for x in (row.specializations or [])),
                ]
            )
            return sum(1 for term in terms if term in bag)

        ranked = sorted(
            self._grants.items(),
            key=lambda item: (_score(item[0], item[1]), item[0]),
            reverse=True,
        )
        return [gid for gid, _ in ranked[:top]]

    async def fetch_metadata(self, grant_id: str) -> GrantMetadata:
        await asyncio.sleep(0)
        row = self._grants.get(str(grant_id or "").strip())
        if not row:
            raise ValueError(f"Grant not found: {grant_id}")
        return row.metadata

    async def fetch_requirement_domains(self, grant_id: str) -> List[str]:
        await asyncio.sleep(0)
        row = self._grants.get(str(grant_id or "").strip())
        return list(row.domains) if row else []

    async def fetch_requirement_specializations(self, grant_id: str) -> List[str]:
        await asyncio.sleep(0)
        row = self._grants.get(str(grant_id or "").strip())
        return list(row.specializations) if row else []

    async def fetch_requirement_eligibility(self, grant_id: str) -> List[str]:
        await asyncio.sleep(0)
        row = self._grants.get(str(grant_id or "").strip())
        return list(row.eligibility) if row else []

    async def fetch_requirement_deliverables(self, grant_id: str) -> List[str]:
        await asyncio.sleep(0)
        row = self._grants.get(str(grant_id or "").strip())
        return list(row.deliverables) if row else []
