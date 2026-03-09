from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings
from graph_rag.faculty.sync_keyword_links_v2 import _load_faculties, sync_faculty_keyword_links_to_neo4j
from logging_setup import setup_logging
from services.keywords.faculty_keyword_generator_v2 import FacultyKeywordGeneratorV2
from utils.thread_pool import parallel_map

setup_logging()
logger = logging.getLogger(__name__)


class FacultyKeywordLinkNeo4jSink(FacultyKeywordGeneratorV2):
    def __init__(
        self,
        *,
        driver,
        database: str,
        faculty_email_by_id: Dict[int, str],
        embedding_batch_size: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._driver = driver
        self._database = str(database or "").strip() or "neo4j"
        self._faculty_email_by_id = {int(k): str(v).strip().lower() for k, v in dict(faculty_email_by_id or {}).items()}
        self._embedding_batch_size = max(1, int(embedding_batch_size or 1))
        self._lock = threading.Lock()
        self.synced: List[Dict[str, Any]] = []

    def save_faculty_keywords(
        self,
        *,
        faculty_id: int,
        keywords: Dict[str, Any],
        raw_json: Dict[str, Any] | None = None,
        source_model: str | None = None,
    ) -> None:
        _ = raw_json, source_model
        fid = int(faculty_id)
        email = str(self._faculty_email_by_id.get(fid, "") or "").strip().lower()
        if not email:
            raise ValueError(f"Missing email mapping for faculty_id={fid}")

        result = sync_faculty_keyword_links_to_neo4j(
            driver=self._driver,
            database=self._database,
            faculty_row={
                "faculty_id": fid,
                "email": email,
                "keywords": dict(keywords or {}),
            },
            embedding_batch_size=self._embedding_batch_size,
        )
        with self._lock:
            self.synced.append(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FacultyKeywordGeneratorV2 and sync keyword linkage edges directly to Neo4j "
            "(no keyword persistence dependency in Postgres)."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Run one faculty_id.")
    parser.add_argument("--all", action="store_true", help="Run all faculty rows.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for --all.")
    parser.add_argument("--max-context-chars", type=int, default=40000)
    parser.add_argument("--max-neo4j-chunks", type=int, default=4000)
    parser.add_argument("--reserve-prompt-chars", type=int, default=3000)
    parser.add_argument("--embedding-batch-size", type=int, default=12, help="Embedding batch size for keyword nodes.")
    parser.add_argument("--use-llm-merge", action="store_true", help="Enable domain merge LLM chain.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first failure.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only.")
    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()

    settings_neo4j = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    faculty_rows = _load_faculties(
        faculty_id=int(args.faculty_id or 0),
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )
    if not faculty_rows:
        raise RuntimeError("No faculty rows found for requested scope.")

    target_ids = [int(x["faculty_id"]) for x in faculty_rows]
    email_map = {int(x["faculty_id"]): str(x.get("email") or "").strip().lower() for x in faculty_rows}

    max_workers = max(1, int(args.max_workers or 1))
    max_context_chars = max(1000, int(args.max_context_chars or 40000))
    max_neo4j_chunks = max(10, int(args.max_neo4j_chunks or 4000))
    reserve_prompt_chars = max(0, int(args.reserve_prompt_chars or 0))
    embedding_batch_size = max(1, int(args.embedding_batch_size or 1))

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()
        generator = FacultyKeywordLinkNeo4jSink(
            driver=driver,
            database=settings_neo4j.database,
            faculty_email_by_id=email_map,
            embedding_batch_size=embedding_batch_size,
            max_context_chars=max_context_chars,
            max_neo4j_chunks=max_neo4j_chunks,
            reserve_prompt_chars=reserve_prompt_chars,
        )

        if len(target_ids) == 1:
            fid = int(target_ids[0])
            generator.run_faculty_keyword_pipeline(
                faculty_id=fid,
                max_context_chars=max_context_chars,
                persist=True,
                use_llm_merge=bool(args.use_llm_merge),
            )
            summary = {
                "total": 1,
                "succeeded": 1,
                "failed": 0,
                "failed_faculty_ids": [],
            }
            run_results = [{"faculty_id": fid, "ok": True}]
        else:
            def _run_one(fid: int) -> Dict[str, Any]:
                generator.run_faculty_keyword_pipeline(
                    faculty_id=int(fid),
                    max_context_chars=max_context_chars,
                    persist=True,
                    use_llm_merge=bool(args.use_llm_merge),
                )
                return {"faculty_id": int(fid), "ok": True}

            def _on_error(_idx: int, fid: int, e: Exception) -> Dict[str, Any]:
                logger.exception("[sync_keyword_links_direct_v2] faculty_id=%s failed", fid)
                return {
                    "faculty_id": int(fid),
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }

            run_results = parallel_map(
                target_ids,
                max_workers=max_workers,
                run_item=_run_one,
                on_error=_on_error,
            )
            failed_ids = [int(x["faculty_id"]) for x in run_results if not bool(x.get("ok"))]
            summary = {
                "total": len(target_ids),
                "succeeded": len(target_ids) - len(failed_ids),
                "failed": len(failed_ids),
                "failed_faculty_ids": failed_ids,
            }
            if bool(args.stop_on_error) and failed_ids:
                summary["stopped_on_error"] = True

    synced = list(generator.synced)
    totals = {
        "domain_keywords": sum(int(x.get("counts", {}).get("domain_keywords", 0)) for x in synced),
        "specialization_keywords": sum(int(x.get("counts", {}).get("specialization_keywords", 0)) for x in synced),
        "embedded_keywords": sum(int(x.get("counts", {}).get("embedded_keywords", 0)) for x in synced),
        "domain_gate_nodes": sum(int(x.get("counts", {}).get("domain_gate_node", 0)) for x in synced),
        "domain_gate_terms": sum(int(x.get("counts", {}).get("domain_gate_terms", 0)) for x in synced),
        "domain_gate_embedded": sum(int(x.get("counts", {}).get("domain_gate_embedded", 0)) for x in synced),
        "domain_chunk_linked": sum(int(x.get("counts", {}).get("domain_chunk_linked", 0)) for x in synced),
        "domain_publication_linked": sum(int(x.get("counts", {}).get("domain_publication_linked", 0)) for x in synced),
        "domain_specialization_linked": sum(int(x.get("counts", {}).get("domain_specialization_linked", 0)) for x in synced),
        "chunk_specialization_linked": sum(int(x.get("counts", {}).get("chunk_specialization_linked", 0)) for x in synced),
        "publication_specialization_linked": sum(
            int(x.get("counts", {}).get("publication_specialization_linked", 0)) for x in synced
        ),
    }

    payload = {
        "scope": {
            "faculty_id": int(args.faculty_id or 0),
            "all": bool(args.all),
            "limit": max(0, int(args.limit or 0)),
            "offset": max(0, int(args.offset or 0)),
            "max_workers": max_workers,
            "max_context_chars": max_context_chars,
            "max_neo4j_chunks": max_neo4j_chunks,
            "reserve_prompt_chars": reserve_prompt_chars,
            "embedding_batch_size": embedding_batch_size,
            "use_llm_merge": bool(args.use_llm_merge),
        },
        "summary": summary,
        "totals": totals,
        "run_results": run_results,
        "synced": synced,
    }

    if not bool(args.json_only):
        print("Faculty direct keyword->link sync complete.")
        print(f"  total                             : {summary['total']}")
        print(f"  succeeded                         : {summary['succeeded']}")
        print(f"  failed                            : {summary['failed']}")
        print(f"  domain keywords                   : {totals['domain_keywords']}")
        print(f"  specialization keywords           : {totals['specialization_keywords']}")
        print(f"  embedded keywords                 : {totals['embedded_keywords']}")
        print(f"  domain gate nodes                 : {totals['domain_gate_nodes']}")
        print(f"  domain gate terms                 : {totals['domain_gate_terms']}")
        print(f"  domain gate embedded              : {totals['domain_gate_embedded']}")
        print(f"  domain->chunk linked              : {totals['domain_chunk_linked']}")
        print(f"  domain->publication linked        : {totals['domain_publication_linked']}")
        print(f"  domain->specialization linked     : {totals['domain_specialization_linked']}")
        print(f"  chunk->specialization linked      : {totals['chunk_specialization_linked']}")
        print(f"  publication->specialization linked: {totals['publication_specialization_linked']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
