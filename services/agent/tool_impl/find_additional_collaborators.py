from __future__ import annotations

from typing import List, Dict, Any

from db.db_conn import SessionLocal
from dao.faculty_dao import FacultyDAO
import re

from db.models.faculty import Faculty
from dao.opportunity_dao import OpportunityDAO
from services.matching.group_match_super_faculty import run_group_match
from services.justification.generate_group_justification import (
    run_justifications_from_group_results,
)


def find_additional_collaborators(
    faculty_emails: list[str],
    opp_ids: list[str] | None,
    team_size: int,
) -> dict:
    # Normalize inputs
    faculty_emails = [str(e).strip() for e in (faculty_emails or []) if str(e).strip()]
    opp_ids_str = [str(x).strip() for x in (opp_ids or []) if str(x).strip()] if opp_ids else None

    sess = SessionLocal()
    try:
        # Resolve opportunity names to ids if needed
        resolved_opp_ids: list[str] | None = None
        if opp_ids_str:
            uuid_re = re.compile(r"^[0-9a-fA-F-]{32,36}$")
            ids: list[str] = []
            names: list[str] = []
            for v in opp_ids_str:
                if uuid_re.match(v):
                    ids.append(v)
                else:
                    names.append(v)

            if names:
                odao = OpportunityDAO(sess)
                for name in names:
                    ids.extend(odao.find_opportunity_ids_by_title(name, limit=5))

            resolved_opp_ids = list(dict.fromkeys(ids)) if ids else None

        results = run_group_match(
            faculty_emails=faculty_emails,
            opp_ids=resolved_opp_ids,
            team_size=int(team_size),
            desired_team_count=1,
            use_llm_selection=False,
        )
        id_set = set()
        for row in results or []:
            for team in row.get("selected_teams", []):
                for fid in team.get("team", []):
                    id_set.add(int(fid))

        id_to_email: Dict[int, str] = {}
        if id_set:
            rows = (
                sess.query(Faculty.faculty_id, Faculty.email)
                .filter(Faculty.faculty_id.in_(list(id_set)))
                .all()
            )
            for fid, email in rows:
                if email:
                    id_to_email[int(fid)] = email

        per_opp = []
        selected_additional_emails: List[str] = []
        selected_additional_names: List[str] = []

        for row in results or []:
            opp_id = row.get("opp_id")
            teams = row.get("selected_teams", [])
            if not teams:
                continue
            team_ids = teams[0].get("team", [])
            additional_ids = [
                int(fid) for fid in team_ids if id_to_email.get(int(fid)) not in faculty_emails
            ]
            additional_emails = [id_to_email.get(fid) for fid in additional_ids if id_to_email.get(fid)]
            per_opp.append(
                {
                    "opp_id": opp_id,
                    "additional_emails": list(additional_emails),
                    "score": teams[0].get("score"),
                }
            )

            if not selected_additional_emails:
                selected_additional_emails = list(additional_emails)

        if selected_additional_emails:
            dao = FacultyDAO(sess)
            name_map = dao.get_names_by_emails(selected_additional_emails)
            if all(name_map.get(e) for e in selected_additional_emails):
                selected_additional_names = [name_map[e] for e in selected_additional_emails]

        try:
            report = run_justifications_from_group_results(
                group_results=results,
                limit_rows=500,
                include_trace=False,
            )
            return report
        except Exception as exc:
            return f"Error generating report: {type(exc).__name__}: {exc}"
    finally:
        sess.close()
