from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context.context_generator import ContextGenerator
from services.justification.group_justification_engine import GroupJustificationEngine
from services.matching.team_grant_matcher import TeamGrantMatcher


class GroupJustificationGenerator:
    DEFAULT_JUSTIFICATION_WORKERS = 4

    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        context_generator: Optional[ContextGenerator] = None,
        team_grant_matcher: Optional[TeamGrantMatcher] = None,
    ):
        self.session_factory = session_factory
        self.context_generator = context_generator or ContextGenerator()
        self.team_grant_matcher = team_grant_matcher or TeamGrantMatcher(
            session_factory=session_factory,
            context_generator=self.context_generator,
        )

    @staticmethod
    def _expand_group_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        for row in rows:
            if "selected_teams" in row:
                opp_id = row.get("opp_id") or row.get("grant_id")
                for cand in row.get("selected_teams") or []:
                    expanded.append(
                        {
                            "opp_id": opp_id,
                            "team": cand.get("team"),
                            "final_coverage": cand.get("final_coverage"),
                            "score": cand.get("score"),
                        }
                    )
            else:
                expanded.append(
                    {
                        "opp_id": row.get("opp_id") or row.get("grant_id"),
                        "team": row.get("team"),
                        "final_coverage": row.get("final_coverage"),
                        "score": row.get("score"),
                    }
                )
        return expanded

    @staticmethod
    def _parse_team_ids(raw_team: Any) -> List[int]:
        out: List[int] = []
        for fid in list(raw_team or []):
            try:
                out.append(int(fid))
            except Exception:
                continue
        return out

    @staticmethod
    def _ordered_unique(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for v in values:
            if v and v not in seen:
                out.append(v)
                seen.add(v)
        return out

    @staticmethod
    def _build_team_members(fac_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "faculty_id": f.get("faculty_id") or f.get("id"),
                "faculty_name": f.get("name"),
                "faculty_email": f.get("email"),
            }
            for f in fac_ctxs
        ]

    def run_justifications_from_group_results(
        self,
        *,
        faculty_emails: Union[str, List[str]],
        team_size: int,
        opp_ids: Optional[List[str]] = None,
        limit_rows: int = 500,
        include_trace: bool = False,
    ) -> List[Dict[str, Any]]:
        with self.session_factory() as sess:
            odao = OpportunityDAO(sess)
            fdao = FacultyDAO(sess)

            email_list = [faculty_emails] if isinstance(faculty_emails, str) else list(faculty_emails)
            group_results = self.team_grant_matcher.run_group_match(
                faculty_emails=email_list,
                team_size=team_size,
                limit_rows=limit_rows,
                opp_ids=opp_ids,
            )
            normalized_rows = self._expand_group_results(group_results)
            if not normalized_rows:
                raise ValueError(f"No group matches found for {faculty_emails}")

            opp_cache: Dict[str, Dict[str, Any]] = {}
            fac_cache: Dict[int, Dict[str, Any]] = {}
            opp_member_cov_cache: Dict[str, Dict[int, Dict[str, Dict[int, float]]]] = {}
            results_by_index: Dict[int, Dict[str, Any]] = {}
            work_items: List[Dict[str, Any]] = []

            opp_ids = self._ordered_unique(
                [
                    str(row.get("opp_id") or row.get("grant_id") or "").strip()
                    for row in normalized_rows
                    if str(row.get("opp_id") or row.get("grant_id") or "").strip()
                ]
            )
            team_fids = sorted(
                {
                    int(fid)
                    for row in normalized_rows
                    for fid in self._parse_team_ids(row.get("team"))
                }
            )

            for opp_id in opp_ids:
                opp_cache[opp_id] = odao.read_opportunity_context(opp_id) or {}
            for fid in team_fids:
                fac_cache[fid] = fdao.get_faculty_keyword_context(fid) or {}
            for opp_id in opp_ids:
                opp_member_cov_cache[opp_id] = self.context_generator.build_member_coverages_for_opportunity(
                    sess=sess,
                    opportunity_id=opp_id,
                    limit_rows=limit_rows,
                )

            for idx, row in enumerate(normalized_rows):
                opp_id = str(row.get("opp_id") or row.get("grant_id") or "").strip()
                team = self._parse_team_ids(row.get("team"))
                coverage = row.get("final_coverage")
                grant_link = f"https://simpler.grants.gov/opportunity/{opp_id}" if opp_id else None

                if not opp_id:
                    results_by_index[idx] = {
                        "index": idx,
                        "grant_id": None,
                        "grant_title": None,
                        "agency_name": None,
                        "grant_link": None,
                        "team": team,
                        "error": "Missing opp_id in group match row.",
                    }
                    continue

                opp_ctx = opp_cache[opp_id]
                grant_title = opp_ctx.get("title") or opp_ctx.get("opportunity_title")
                agency_name = opp_ctx.get("agency") or opp_ctx.get("agency_name")

                if not opp_ctx:
                    results_by_index[idx] = {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": None,
                        "agency_name": None,
                        "grant_link": grant_link,
                        "team": team,
                        "error": "Opportunity not found",
                    }
                    continue

                fac_ctxs: List[Dict[str, Any]] = []
                for fid in team:
                    if fac_cache[fid]:
                        fac_ctxs.append(fac_cache[fid])

                member_coverages = {
                    int(fid): opp_member_cov_cache.get(opp_id, {}).get(int(fid), {"application": {}, "research": {}})
                    for fid in team
                }

                if not fac_ctxs:
                    results_by_index[idx] = {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": grant_title,
                        "agency_name": agency_name,
                        "grant_link": grant_link,
                        "team": team,
                        "error": "No faculty contexts found for team",
                    }
                    continue

                group_meta = {
                    "group_id": row.get("group_id") or row.get("id"),
                    "lambda": row.get("lambda"),
                    "k": row.get("k"),
                    "objective": row.get("objective"),
                    "redundancy": row.get("redundancy"),
                    "meta": row.get("meta"),
                }

                work_items.append(
                    {
                        "index": idx,
                        "row": row,
                        "opp_id": opp_id,
                        "opp_ctx": dict(opp_ctx),
                        "grant_title": grant_title,
                        "agency_name": agency_name,
                        "team": team,
                        "fac_ctxs": [dict(f) for f in fac_ctxs],
                        "coverage": coverage,
                        "member_coverages": member_coverages,
                        "group_meta": group_meta,
                        "grant_link": grant_link,
                    }
                )

            thread_local = threading.local()

            def _get_engine() -> GroupJustificationEngine:
                engine = getattr(thread_local, "engine", None)
                if engine is None:
                    # odao/fdao are not required by run_one; pass None to keep per-thread engine isolated.
                    engine = GroupJustificationEngine(
                        odao=None,  # type: ignore[arg-type]
                        fdao=None,  # type: ignore[arg-type]
                        context_generator=self.context_generator,
                    )
                    thread_local.engine = engine
                return engine

            def _run_item(item: Dict[str, Any]) -> Dict[str, Any]:
                idx = int(item["index"])
                opp_id = str(item["opp_id"])
                team = list(item["team"])
                row = dict(item["row"])
                try:
                    justification, trace = _get_engine().run_one(
                        opp_ctx=dict(item["opp_ctx"]),
                        fac_ctxs=[dict(f) for f in list(item["fac_ctxs"])],
                        coverage=item["coverage"],
                        member_coverages=dict(item["member_coverages"] or {}),
                        group_meta=dict(item["group_meta"] or {}),
                        trace={"index": idx, "opp_id": opp_id, "team": team},
                    )
                    out = {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": item["grant_title"],
                        "agency_name": item["agency_name"],
                        "team": team,
                        "team_members": self._build_team_members(list(item["fac_ctxs"])),
                        "team_score": float(row.get("score") or 0.0),
                        "justification": justification.model_dump(),
                    }
                    if include_trace:
                        out["trace"] = trace
                    return out
                except Exception as e:
                    return {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": item["grant_title"],
                        "agency_name": item["agency_name"],
                        "grant_link": item["grant_link"],
                        "team": team,
                        "error": f"{type(e).__name__}: {e}",
                    }

            max_workers = max(
                1,
                int(
                    os.getenv(
                        "GROUP_JUSTIFICATION_WORKERS",
                        str(self.DEFAULT_JUSTIFICATION_WORKERS),
                    )
                ),
            )
            pool_size = min(max_workers, len(work_items))

            if pool_size <= 1:
                for item in work_items:
                    out = _run_item(item)
                    results_by_index[int(item["index"])] = out
            else:
                with ThreadPoolExecutor(max_workers=pool_size) as ex:
                    futures = {
                        ex.submit(_run_item, item): int(item["index"])
                        for item in work_items
                    }
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            results_by_index[idx] = fut.result()
                        except Exception as e:
                            item = next(w for w in work_items if int(w["index"]) == idx)
                            results_by_index[idx] = {
                                "index": idx,
                                "grant_id": item["opp_id"],
                                "grant_title": item["grant_title"],
                                "agency_name": item["agency_name"],
                                "grant_link": item["grant_link"],
                                "team": item["team"],
                                "error": f"{type(e).__name__}: {e}",
                            }

            return [
                results_by_index[i]
                for i in range(len(normalized_rows))
                if i in results_by_index
            ]
