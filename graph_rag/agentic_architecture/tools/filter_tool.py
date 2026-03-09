from __future__ import annotations

import re
from typing import Any, List, Set

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal



def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def hard_filter_open_grant_ids(
    *,
    opportunity_ids: List[str],
    include_closed: bool = False,
) -> List[str]:
    """
    Neo4j hard filter for grant status/deadline.

    - include_closed=True  -> return input ids as-is (deduped).
    - include_closed=False -> remove grants that are closed/archived/inactive/canceled
                              or have close_date in the past (if parseable).
    """
    deduped: List[str] = []
    seen = set()
    for raw_id in opportunity_ids or []:
        opp_id = _clean_text(raw_id)
        if not opp_id:
            continue
        key = opp_id.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opp_id)

    if include_closed or not deduped:
        return deduped

    try:
        from neo4j import GraphDatabase, RoutingControl
        from graph_rag.common import load_dotenv_if_present, read_neo4j_settings

        load_dotenv_if_present()
        neo4j_settings = read_neo4j_settings()
    except Exception:
        # Fail open when Neo4j is unavailable/config is missing.
        return deduped

    query = """
        UNWIND $opportunity_ids AS opportunity_id
        MATCH (g:Grant {opportunity_id: opportunity_id})
        WITH
            g,
            toLower(coalesce(g.opportunity_status, "")) AS status_token,
            coalesce(toString(g.close_date), "") AS close_token
        WITH
            g,
            status_token,
            CASE
                WHEN close_token =~ '^\\d{4}-\\d{2}-\\d{2}.*$' THEN date(substring(close_token, 0, 10))
                ELSE NULL
            END AS close_dt
        WHERE
            NONE(token IN ['closed', 'archived', 'inactive', 'canceled'] WHERE status_token CONTAINS token)
            AND (close_dt IS NULL OR close_dt >= date())
        RETURN g.opportunity_id AS opportunity_id
    """

    try:
        with GraphDatabase.driver(
            neo4j_settings.uri,
            auth=(neo4j_settings.username, neo4j_settings.password),
        ) as driver:
            records, _, _ = driver.execute_query(
                query,
                parameters_={"opportunity_ids": deduped},
                routing_=RoutingControl.READ,
                database_=neo4j_settings.database,
            )
    except Exception:
        # Fail open when query execution fails.
        return deduped

    allowed = {_clean_text(row.get("opportunity_id")) for row in records}
    return [opp_id for opp_id in deduped if opp_id in allowed]


def filter_faculty_ids_by_domain_threshold(
    *,
    opportunity_id: str,
    threshold: float,
) -> List[int]:
    """
    Return all faculty_ids from relational DB whose domain similarity against the
    opportunity embedding is >= threshold.
    """
    with SessionLocal() as session:
        match_dao = MatchDAO(session)
        return match_dao.faculty_ids_for_opportunity_above_domain_threshold(
            opportunity_id=str(opportunity_id),
            min_domain=float(threshold),
        )

def filter_grant_ids_by_domain_threshold(
    *,
    faculty_email: str,
    threshold: float,
) -> List[int]:
    """
    Return all faculty_ids from relational DB whose domain similarity against the
    opportunity embedding is >= threshold.
    """
    with SessionLocal() as session:
        match_dao = MatchDAO(session)
        fac_dao = FacultyDAO(session)

        return match_dao.grant_ids_for_faculty_above_domain_threshold(
            faculty_id=fac_dao.get_faculty_id_by_email(_clean_text(faculty_email)),
            min_domain=float(threshold),
        )
#TODO agency filter