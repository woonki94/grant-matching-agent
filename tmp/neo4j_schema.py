from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from tmp.neo4j_common import Neo4jSettings, load_dotenv_if_present, read_neo4j_settings

GRANT_SCHEMA_STATEMENTS: List[Tuple[str, str]] = [
    (
        "grant_opportunity_id_unique",
        "CREATE CONSTRAINT grant_opportunity_id_unique IF NOT EXISTS "
        "FOR (g:Grant) REQUIRE g.opportunity_id IS UNIQUE",
    ),
    (
        "agency_name_unique",
        "CREATE CONSTRAINT agency_name_unique IF NOT EXISTS "
        "FOR (a:Agency) REQUIRE a.name IS UNIQUE",
    ),
    (
        "grant_additional_info_id_unique",
        "CREATE CONSTRAINT grant_additional_info_id_unique IF NOT EXISTS "
        "FOR (ai:GrantAdditionalInfo) REQUIRE ai.additional_info_id IS UNIQUE",
    ),
    (
        "grant_attachment_id_unique",
        "CREATE CONSTRAINT grant_attachment_id_unique IF NOT EXISTS "
        "FOR (att:GrantAttachment) REQUIRE att.attachment_id IS UNIQUE",
    ),
    (
        "grant_keyword_unique",
        "CREATE CONSTRAINT grant_keyword_unique IF NOT EXISTS "
        "FOR (k:GrantKeyword) REQUIRE (k.value, k.section, k.bucket) IS UNIQUE",
    ),
    (
        "grant_broad_category_unique",
        "CREATE CONSTRAINT grant_broad_category_unique IF NOT EXISTS "
        "FOR (bc:GrantBroadCategory) REQUIRE bc.name IS UNIQUE",
    ),
    (
        "grant_specific_category_unique",
        "CREATE CONSTRAINT grant_specific_category_unique IF NOT EXISTS "
        "FOR (sc:GrantSpecificCategory) REQUIRE sc.name IS UNIQUE",
    ),
    (
        "grant_opportunity_category_unique",
        "CREATE CONSTRAINT grant_opportunity_category_unique IF NOT EXISTS "
        "FOR (oc:OpportunityCategory) REQUIRE oc.name IS UNIQUE",
    ),
    (
        "grant_applicant_type_unique",
        "CREATE CONSTRAINT grant_applicant_type_unique IF NOT EXISTS "
        "FOR (at:ApplicantType) REQUIRE at.name IS UNIQUE",
    ),
    (
        "grant_funding_category_unique",
        "CREATE CONSTRAINT grant_funding_category_unique IF NOT EXISTS "
        "FOR (fc:FundingCategory) REQUIRE fc.name IS UNIQUE",
    ),
    (
        "grant_funding_instrument_unique",
        "CREATE CONSTRAINT grant_funding_instrument_unique IF NOT EXISTS "
        "FOR (fi:FundingInstrument) REQUIRE fi.name IS UNIQUE",
    ),
    (
        "grant_close_date_index",
        "CREATE INDEX grant_close_date IF NOT EXISTS "
        "FOR (g:Grant) ON (g.close_date)",
    ),
    (
        "grant_agency_name_index",
        "CREATE INDEX grant_agency_name IF NOT EXISTS "
        "FOR (g:Grant) ON (g.agency_name)",
    ),
    (
        "grant_category_index",
        "CREATE INDEX grant_category IF NOT EXISTS "
        "FOR (g:Grant) ON (g.category)",
    ),
    (
        "grant_status_index",
        "CREATE INDEX grant_status IF NOT EXISTS "
        "FOR (g:Grant) ON (g.opportunity_status)",
    ),
    (
        "grant_keyword_value_index",
        "CREATE INDEX grant_keyword_value IF NOT EXISTS "
        "FOR (k:GrantKeyword) ON (k.value)",
    ),
]

FACULTY_SCHEMA_STATEMENTS: List[Tuple[str, str]] = [
    (
        "faculty_email_unique",
        "CREATE CONSTRAINT faculty_email_unique IF NOT EXISTS "
        "FOR (f:Faculty) REQUIRE f.email IS UNIQUE",
    ),
    (
        "faculty_id_unique",
        "CREATE CONSTRAINT faculty_id_unique IF NOT EXISTS "
        "FOR (f:Faculty) REQUIRE f.faculty_id IS UNIQUE",
    ),
    (
        "faculty_additional_info_id_unique",
        "CREATE CONSTRAINT faculty_additional_info_id_unique IF NOT EXISTS "
        "FOR (ai:FacultyAdditionalInfo) REQUIRE ai.additional_info_id IS UNIQUE",
    ),
    (
        "faculty_publication_id_unique",
        "CREATE CONSTRAINT faculty_publication_id_unique IF NOT EXISTS "
        "FOR (p:FacultyPublication) REQUIRE p.publication_id IS UNIQUE",
    ),
    (
        "faculty_keyword_unique",
        "CREATE CONSTRAINT faculty_keyword_unique IF NOT EXISTS "
        "FOR (k:FacultyKeyword) REQUIRE (k.value, k.section, k.bucket) IS UNIQUE",
    ),
    (
        "faculty_publication_year_index",
        "CREATE INDEX faculty_publication_year IF NOT EXISTS "
        "FOR (p:FacultyPublication) ON (p.year)",
    ),
    (
        "faculty_keyword_value_index",
        "CREATE INDEX faculty_keyword_value IF NOT EXISTS "
        "FOR (k:FacultyKeyword) ON (k.value)",
    ),
]


def init_neo4j_schema(
    settings: Neo4jSettings,
    *,
    include_grant: bool = True,
    include_faculty: bool = True,
) -> None:
    statements: List[Tuple[str, str]] = []
    if include_grant:
        statements.extend(GRANT_SCHEMA_STATEMENTS)
    if include_faculty:
        statements.extend(FACULTY_SCHEMA_STATEMENTS)

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()
        for name, statement in statements:
            driver.execute_query(
                statement,
                database_=settings.database,
            )
            print(f"Applied: {name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize shared Neo4j schema for Grant/Faculty GraphRAG prototypes.")
    parser.add_argument("--grants-only", action="store_true", help="Apply only grant schema statements.")
    parser.add_argument("--faculty-only", action="store_true", help="Apply only faculty schema statements.")
    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    include_grant = True
    include_faculty = True
    if args.grants_only and not args.faculty_only:
        include_faculty = False
    if args.faculty_only and not args.grants_only:
        include_grant = False

    init_neo4j_schema(
        settings,
        include_grant=include_grant,
        include_faculty=include_faculty,
    )
    print("Shared Neo4j schema ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
