from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from neo4j import GraphDatabase, RoutingControl

from tmp.agentic_arch.models import FacultyBasicInfo, FacultyPublication, GrantMetadata
from tmp.agentic_arch.tools import FacultyTools, GrantTools
from tmp.neo4j_common import Neo4jSettings

FACULTY_KEYWORD_RELATIONS = [
    "HAS_RESEARCH_DOMAIN",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_DOMAIN",
    "HAS_APPLICATION_SPECIALIZATION",
]

GRANT_KEYWORD_RELATIONS = [
    "HAS_RESEARCH_DOMAIN",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_DOMAIN",
    "HAS_APPLICATION_SPECIALIZATION",
]


class Neo4jFacultyTools(FacultyTools):
    def __init__(self, settings: Neo4jSettings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )

    def close(self) -> None:
        self.driver.close()

    async def fetch_basic_info(self, email: str) -> FacultyBasicInfo:
        def _run() -> FacultyBasicInfo:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (f:Faculty {email: $email})
                RETURN f {
                    .email,
                    .name,
                    .position,
                    .organizations
                } AS faculty
                LIMIT 1
                """,
                parameters_={"email": str(email or "").strip().lower()},
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            if not records:
                raise ValueError(f"Faculty not found: {email}")
            row = records[0].get("faculty") or {}
            return FacultyBasicInfo(
                email=str(row.get("email") or "").strip().lower(),
                faculty_name=row.get("name"),
                position=row.get("position"),
                organizations=list(row.get("organizations") or []),
            )

        return await asyncio.to_thread(_run)

    async def fetch_keywords(self, email: str) -> List[str]:
        def _run() -> List[str]:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (f:Faculty {email: $email})-[r]->(k:FacultyKeyword)
                WHERE type(r) IN $relations
                RETURN k.value AS value, coalesce(r.weight, 0.5) AS weight
                ORDER BY weight DESC, value ASC
                """,
                parameters_={
                    "email": str(email or "").strip().lower(),
                    "relations": FACULTY_KEYWORD_RELATIONS,
                },
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            out: List[str] = []
            seen = set()
            for row in records:
                value = str(row.get("value") or "").strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                out.append(value)
            return out

        return await asyncio.to_thread(_run)

    async def fetch_additional_text(self, email: str, max_items: int = 3) -> List[str]:
        def _run() -> List[str]:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (f:Faculty {email: $email})-[:HAS_ADDITIONAL_INFO]->(ai:FacultyAdditionalInfo)
                WHERE ai.extracted_text IS NOT NULL
                RETURN ai.extracted_text AS text
                ORDER BY ai.extracted_at DESC, ai.additional_info_id DESC
                LIMIT $max_items
                """,
                parameters_={
                    "email": str(email or "").strip().lower(),
                    "max_items": max(0, int(max_items or 0)),
                },
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            return [str(r.get("text") or "").strip() for r in records if str(r.get("text") or "").strip()]

        return await asyncio.to_thread(_run)

    async def fetch_publications(self, email: str, max_items: int = 10) -> List[FacultyPublication]:
        def _run() -> List[FacultyPublication]:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (f:Faculty {email: $email})-[:AUTHORED]->(p:FacultyPublication)
                RETURN p.title AS title, p.abstract AS abstract, p.year AS year
                ORDER BY p.year DESC, p.publication_id DESC
                LIMIT $max_items
                """,
                parameters_={
                    "email": str(email or "").strip().lower(),
                    "max_items": max(0, int(max_items or 0)),
                },
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            out: List[FacultyPublication] = []
            for row in records:
                title = str(row.get("title") or "").strip()
                if not title:
                    continue
                year = row.get("year")
                out.append(
                    FacultyPublication(
                        title=title,
                        abstract=(str(row.get("abstract") or "").strip() or None),
                        year=int(year) if year is not None else None,
                    )
                )
            return out

        return await asyncio.to_thread(_run)


class Neo4jGrantTools(GrantTools):
    def __init__(self, settings: Neo4jSettings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )

    def close(self) -> None:
        self.driver.close()

    async def search_candidate_grants(self, profession_focus: List[str], top_k: int = 20) -> List[str]:
        def _run() -> List[str]:
            terms = [str(x).strip().lower() for x in (profession_focus or []) if str(x).strip()]
            top = max(1, int(top_k or 20))

            if not terms:
                records, _, _ = self.driver.execute_query(
                    """
                    MATCH (g:Grant)
                    RETURN g.opportunity_id AS grant_id
                    ORDER BY g.close_date ASC, g.opportunity_id ASC
                    LIMIT $top_k
                    """,
                    parameters_={"top_k": top},
                    routing_=RoutingControl.READ,
                    database_=self.settings.database,
                )
                return [str(r.get("grant_id") or "").strip() for r in records if str(r.get("grant_id") or "").strip()]

            records, _, _ = self.driver.execute_query(
                """
                UNWIND $terms AS term
                MATCH (g:Grant)-[r]->(k:GrantKeyword)
                WHERE type(r) IN $relations
                  AND (toLower(k.value) CONTAINS term OR term CONTAINS toLower(k.value))
                WITH g, sum(coalesce(r.weight, 0.5)) AS score
                RETURN g.opportunity_id AS grant_id
                ORDER BY score DESC, g.close_date ASC, g.opportunity_id ASC
                LIMIT $top_k
                """,
                parameters_={
                    "terms": terms,
                    "relations": GRANT_KEYWORD_RELATIONS,
                    "top_k": top,
                },
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            return [str(r.get("grant_id") or "").strip() for r in records if str(r.get("grant_id") or "").strip()]

        return await asyncio.to_thread(_run)

    async def fetch_metadata(self, grant_id: str) -> GrantMetadata:
        def _run() -> GrantMetadata:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (g:Grant {opportunity_id: $grant_id})
                RETURN g {
                    .opportunity_id,
                    .opportunity_title,
                    .agency_name,
                    .close_date
                } AS grant
                LIMIT 1
                """,
                parameters_={"grant_id": str(grant_id or "").strip()},
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            if not records:
                raise ValueError(f"Grant not found: {grant_id}")
            row = records[0].get("grant") or {}
            return GrantMetadata(
                grant_id=str(row.get("opportunity_id") or "").strip(),
                grant_name=row.get("opportunity_title"),
                agency_name=row.get("agency_name"),
                close_date=row.get("close_date"),
            )

        return await asyncio.to_thread(_run)

    async def fetch_requirement_domains(self, grant_id: str) -> List[str]:
        return await self._fetch_keyword_bucket(grant_id=grant_id, bucket="domain")

    async def fetch_requirement_specializations(self, grant_id: str) -> List[str]:
        return await self._fetch_keyword_bucket(grant_id=grant_id, bucket="specialization")

    async def _fetch_keyword_bucket(self, *, grant_id: str, bucket: str) -> List[str]:
        def _run() -> List[str]:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (g:Grant {opportunity_id: $grant_id})-[r]->(k:GrantKeyword)
                WHERE type(r) IN $relations
                  AND k.bucket = $bucket
                RETURN k.value AS value, coalesce(r.weight, 0.5) AS weight
                ORDER BY weight DESC, value ASC
                """,
                parameters_={
                    "grant_id": str(grant_id or "").strip(),
                    "bucket": str(bucket or "").strip().lower(),
                    "relations": GRANT_KEYWORD_RELATIONS,
                },
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            out: List[str] = []
            seen = set()
            for row in records:
                value = str(row.get("value") or "").strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                out.append(value)
            return out

        return await asyncio.to_thread(_run)

    async def fetch_requirement_eligibility(self, grant_id: str) -> List[str]:
        def _run() -> List[str]:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (g:Grant {opportunity_id: $grant_id})-[:HAS_APPLICANT_TYPE]->(a:ApplicantType)
                RETURN DISTINCT a.name AS name
                ORDER BY name ASC
                """,
                parameters_={"grant_id": str(grant_id or "").strip()},
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            return [str(r.get("name") or "").strip() for r in records if str(r.get("name") or "").strip()]

        return await asyncio.to_thread(_run)

    async def fetch_requirement_deliverables(self, grant_id: str) -> List[str]:
        def _run() -> List[str]:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (g:Grant {opportunity_id: $grant_id})
                OPTIONAL MATCH (g)-[:HAS_FUNDING_INSTRUMENT]->(fi:FundingInstrument)
                OPTIONAL MATCH (g)-[:HAS_FUNDING_CATEGORY]->(fc:FundingCategory)
                WITH
                    [x IN collect(DISTINCT fi.name) WHERE x IS NOT NULL] +
                    [x IN collect(DISTINCT fc.name) WHERE x IS NOT NULL] AS raw_items
                UNWIND raw_items AS item
                RETURN DISTINCT item AS name
                ORDER BY name ASC
                """,
                parameters_={"grant_id": str(grant_id or "").strip()},
                routing_=RoutingControl.READ,
                database_=self.settings.database,
            )
            return [str(r.get("name") or "").strip() for r in records if str(r.get("name") or "").strip()]

        return await asyncio.to_thread(_run)
