from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm_client
from tmp.agentic_arch_v2.models import PlannedQueryOut, QueryItem, QueryPlanOut

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan inter-agent questions for a faculty-to-grant one-to-one matcher.\n"
            "Output concise, high-value queries only.\n"
            "Rules:\n"
            "1) Start from grant-side queries first.\n"
            "2) Include at least one grant_metadata query and one grant_requirement query.\n"
            "3) Include faculty-targeted query only when grant-side evidence likely needs faculty clarification.\n"
            "4) Keep intent short snake_case.\n"
            "5) expected_fields should be concrete keys.\n",
        ),
        (
            "human",
            "Faculty profile JSON:\n{faculty_profile_json}\n\n"
            "Candidate grant ids:\n{candidate_grant_ids}\n\n"
            "Return query plan.",
        ),
    ]
)


class LLMQueryPlanner:
    def __init__(self, *, enable_llm: bool = True):
        self.enable_llm = bool(enable_llm)
        self._chain = None
        # Keep fallback failures quiet when Bedrock is unreachable.
        logging.getLogger("langchain_aws").setLevel(logging.CRITICAL)
        logging.getLogger("botocore").setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    def _get_chain(self):
        if self._chain is None:
            llm = get_llm_client().build()
            self._chain = PLAN_PROMPT | llm.with_structured_output(QueryPlanOut)
        return self._chain

    @staticmethod
    def _fallback_queries() -> List[PlannedQueryOut]:
        return [
            PlannedQueryOut(
                target_agent="grant",
                intent="grant_metadata",
                question="What are the grant name, agency, and close date?",
                expected_fields=["grant_name", "agency_name", "close_date"],
                priority=0.95,
                confidence_threshold=0.8,
            ),
            PlannedQueryOut(
                target_agent="grant",
                intent="grant_requirement",
                question="What domains, required specializations, eligibility, and deliverables are required?",
                expected_fields=["domains", "specializations", "eligibility", "deliverables"],
                priority=0.95,
                confidence_threshold=0.75,
            ),
            PlannedQueryOut(
                target_agent="grant",
                intent="grant_profession_fit_probe",
                question="Given the faculty profession focus, what requirement gaps are likely?",
                expected_fields=["matched_terms", "missing_terms"],
                priority=0.8,
                confidence_threshold=0.7,
            ),
        ]

    def plan_initial_queries(
        self,
        *,
        faculty_profile: Dict[str, Any],
        candidate_grant_ids: List[str],
    ) -> List[PlannedQueryOut]:
        if not self.enable_llm:
            return self._fallback_queries()
        try:
            out: QueryPlanOut = self._get_chain().invoke(
                {
                    "faculty_profile_json": json.dumps(faculty_profile, ensure_ascii=False),
                    "candidate_grant_ids": json.dumps(candidate_grant_ids, ensure_ascii=False),
                }
            )
            queries = list(out.queries or [])
            if not queries:
                return self._fallback_queries()

            # guardrails: enforce minimum required grant intents
            intents = {str(q.intent or "").strip().lower() for q in queries}
            if "grant_metadata" not in intents:
                queries.append(self._fallback_queries()[0])
            if "grant_requirement" not in intents:
                queries.append(self._fallback_queries()[1])
            return queries
        except Exception:
            return self._fallback_queries()


def expand_queries_for_grants(
    *,
    plan: List[PlannedQueryOut],
    candidate_grant_ids: List[str],
) -> List[QueryItem]:
    queries: List[QueryItem] = []
    counter = 0

    for q in plan or []:
        if q.target_agent == "grant":
            for grant_id in candidate_grant_ids:
                counter += 1
                queries.append(
                    QueryItem(
                        query_id=f"q{counter:04d}",
                        target_agent="grant",
                        intent=str(q.intent or "").strip().lower(),
                        question=str(q.question or "").strip(),
                        expected_fields=list(q.expected_fields or []),
                        priority=float(q.priority),
                        confidence_threshold=float(q.confidence_threshold),
                        grant_id=str(grant_id),
                        context={},
                    )
                )
        else:
            counter += 1
            queries.append(
                QueryItem(
                    query_id=f"q{counter:04d}",
                    target_agent="faculty",
                    intent=str(q.intent or "").strip().lower(),
                    question=str(q.question or "").strip(),
                    expected_fields=list(q.expected_fields or []),
                    priority=float(q.priority),
                    confidence_threshold=float(q.confidence_threshold),
                    grant_id=None,
                    context={},
                )
            )

    queries.sort(key=lambda x: (x.priority, x.query_id), reverse=True)
    return queries
