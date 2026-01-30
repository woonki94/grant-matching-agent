from __future__ import annotations
from typing import Any, Dict, List
import json

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm_client
from dao.group_match_dao import GroupMatchDAO
from dao.opportunity_dao import OpportunityDAO
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import GroupJustificationOut
from services.prompts.group_justification_prompts import GROUP_JUSTIFY_PROMPT


def build_group_justify_chain(justify_prompt: ChatPromptTemplate):
    llm = get_llm_client().build()
    return justify_prompt | llm.with_structured_output(GroupJustificationOut)


def build_group_justification_prompt_payload(
    *,
    opp_ctx: Dict[str, Any],
    fac_ctxs: List[Dict[str, Any]],
    group_row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "grant": {
            "id": opp_ctx.get("opportunity_id"),
            "title": opp_ctx.get("title"),
            "agency": opp_ctx.get("agency"),
            "summary": opp_ctx.get("summary"),
            "keywords": opp_ctx.get("keywords"),
        },
        "team": [
            {
                "faculty_id": f.get("faculty_id"),
                "name": f.get("name"),
                "email": f.get("email"),
                "keywords": f.get("keywords"),
            }
            for f in fac_ctxs
        ],
        "group_match": {
            "group_id": group_row.get("group_id") or group_row.get("id"),
            "lambda": group_row.get("lambda"),
            "k": group_row.get("k"),
            "objective": group_row.get("objective"),
            "redundancy": group_row.get("redundancy"),
            "meta": group_row.get("meta"),
        },
    }


def run_justifications_from_faculty_email(email: str, limit: int = 200) -> str:
    """
    Generate justifications for ALL group matches for a faculty email.
    Returns JSON string of list[ {group_id, grant_id, justification...} ].
    """
    justify_chain = build_group_justify_chain(GROUP_JUSTIFY_PROMPT)

    with SessionLocal() as sess:
        gdao = GroupMatchDAO(sess)
        odao = OpportunityDAO(sess)
        fdao = FacultyDAO(sess)

        groups = gdao.list_groups_for_faculty_email(email=email, limit=limit)
        if not groups:
            raise ValueError(f"No group matches found for email={email}")

        results: List[Dict[str, Any]] = []

        # optional small caches to reduce DB calls
        opp_cache: Dict[str, Dict[str, Any]] = {}
        fac_cache: Dict[int, Dict[str, Any]] = {}

        for idx, group_row in enumerate(groups):
            group_id = int(group_row.get("group_id") or group_row.get("id"))
            grant_id = str(group_row["grant_id"])

            # group members
            members = gdao.read_group_members(group_id=group_id)
            member_fids = [int(m["faculty_id"]) for m in members]
            if not member_fids:
                results.append({
                    "index": idx,
                    "group_id": group_id,
                    "grant_id": grant_id,
                    "error": f"group_id={group_id} has no members",
                })
                continue

            # opportunity context (cached)
            if grant_id not in opp_cache:
                opp_ctx = odao.get_opportunity_context(grant_id)
                opp_cache[grant_id] = opp_ctx
            else:
                opp_ctx = opp_cache[grant_id]

            if not opp_ctx:
                results.append({
                    "index": idx,
                    "group_id": group_id,
                    "grant_id": grant_id,
                    "error": f"Opportunity not found: {grant_id}",
                })
                continue

            # faculty contexts (cached)
            fac_ctxs: List[Dict[str, Any]] = []
            for fid in member_fids:
                if fid not in fac_cache:
                    fac_cache[fid] = fdao.get_faculty_keyword_context(fid) or {}
                if fac_cache[fid]:
                    fac_ctxs.append(fac_cache[fid])

            payload = build_group_justification_prompt_payload(
                opp_ctx=opp_ctx,
                fac_ctxs=fac_ctxs,
                group_row=group_row,
            )

            try:
                out: GroupJustificationOut = justify_chain.invoke({
                    "input_json": json.dumps(payload, ensure_ascii=False)
                })
                results.append({
                    "index": idx,
                    "group_id": group_id,
                    "grant_id": grant_id,
                    "justification": out.model_dump(),
                })
            except Exception as e:
                # don’t fail the whole batch on one bad group
                results.append({
                    "index": idx,
                    "group_id": group_id,
                    "grant_id": grant_id,
                    "error": f"{type(e).__name__}: {e}",
                })

        return json.dumps(results, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    email = "AbbasiB@oregonstate.edu"
    print(run_justifications_from_faculty_email(email))