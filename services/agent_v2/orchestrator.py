from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.agent_v2.router import IntentRouter
from services.agent_v2.state import GrantMatchRequest, GrantMatchWorkflowState
from services.agent_v2.tool_agents import (
    FacultyContextAgent,
    MatchingExecutionAgent,
    OpportunityContextAgent,
)


def build_memory_checkpointer():
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except Exception as e:  # pragma: no cover - dependency guard
        raise ImportError("langgraph is required. Install with: pip install langgraph") from e
    return MemorySaver()


class GrantMatchOrchestrator:
    def __init__(
        self,
        *,
        router: Optional[IntentRouter] = None,
        faculty_agent: Optional[FacultyContextAgent] = None,
        opportunity_agent: Optional[OpportunityContextAgent] = None,
        matching_agent: Optional[MatchingExecutionAgent] = None,
        checkpointer: Any = None,
    ):
        self.router = router or IntentRouter()
        self.faculty_agent = faculty_agent or FacultyContextAgent()
        self.opportunity_agent = opportunity_agent or OpportunityContextAgent()
        self.matching_agent = matching_agent or MatchingExecutionAgent()
        self.graph = self._build_graph(checkpointer=checkpointer)

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _merge_emails(primary: Optional[str], existing: List[str], inferred: List[str]) -> List[str]:
        out: List[str] = []
        for e in [primary, *(existing or []), *(inferred or [])]:
            if e and e not in out:
                out.append(e)
        return out

    @staticmethod
    def _resolve_grant_identifier_type(state: GrantMatchWorkflowState) -> Optional[str]:
        if state.get("grant_identifier_type") in {"link", "title"}:
            return state.get("grant_identifier_type")
        if state.get("grant_link"):
            return "link"
        if state.get("grant_title"):
            return "title"
        return None

    @staticmethod
    def _faculty_in_db(state: GrantMatchWorkflowState) -> Optional[bool]:
        if state.get("faculty_in_db") is not None:
            return bool(state.get("faculty_in_db"))
        if state.get("email_in_db") is not None:
            return bool(state.get("email_in_db"))
        return None

    def _node_route(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_route")
        inferred = self.router.infer(state.get("user_input") or "")
        scenario = inferred.get("scenario") or "one_to_one"

        inferred_emails = list(inferred.get("emails") or [])
        emails = self._merge_emails(
            primary=state.get("email"),
            existing=list(state.get("emails") or []),
            inferred=inferred_emails,
        )
        email = state.get("email") or inferred.get("email") or (emails[0] if emails else None)
        grant_link = state.get("grant_link") or inferred.get("grant_link")
        grant_title = state.get("grant_title") or inferred.get("grant_title")
        grant_identifier_type = (
            state.get("grant_identifier_type")
            or inferred.get("grant_identifier_type")
            or ("link" if grant_link else ("title" if grant_title else None))
        )

        return {
            "scenario": scenario,
            "email": email,
            "emails": emails,
            "grant_link": grant_link,
            "grant_title": grant_title,
            "grant_identifier_type": grant_identifier_type,
            "has_faculty_signal": bool(inferred.get("has_faculty_signal")),
            "has_group_signal": bool(inferred.get("has_group_signal")),
            "has_grant_signal": bool(inferred.get("has_grant_signal")),
            "email_detected": email,
            "emails_detected": emails,
            "grant_link_detected": grant_link,
            "grant_title_detected": grant_title,
        }

    def _edge_from_route(self, state: GrantMatchWorkflowState) -> str:
        scenario = state.get("scenario")
        if scenario == "group_specific_grant":
            return "decide_group_specific_grant"
        if scenario == "group":
            return "decide_group"
        return "decide_one_to_one"

    def _node_decide_one_to_one(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_decide_one_to_one")
        if not state.get("email"):
            return {"decision": "ask_email"}
        if self._faculty_in_db(state) is False:
            return {"decision": "ask_user_reference_data"}
        return {"decision": "generate_keywords_one_to_one"}

    def _node_decide_group(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_decide_group")
        emails = list(state.get("emails") or [])
        if len(emails) < 2:
            return {"decision": "ask_group_emails"}
        if self._faculty_in_db(state) is False:
            return {"decision": "ask_user_reference_data"}
        return {"decision": "generate_keywords_group"}

    def _node_decide_group_specific_grant(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_decide_group_specific_grant")
        emails = list(state.get("emails") or [])
        if len(emails) < 2:
            return {"decision": "ask_group_emails"}
        if self._faculty_in_db(state) is False:
            return {"decision": "ask_user_reference_data"}

        ident = self._resolve_grant_identifier_type(state)
        if ident == "link":
            if not state.get("grant_link"):
                return {"decision": "ask_grant_identifier"}
            return {"decision": "search_grant_by_link_in_db"}
        if ident == "title":
            if not state.get("grant_title"):
                return {"decision": "ask_grant_identifier"}
            return {"decision": "search_grant_by_title_in_db"}
        return {"decision": "ask_grant_identifier"}

    @staticmethod
    def _edge_from_decision(state: GrantMatchWorkflowState) -> str:
        return state.get("decision") or "ask_email"

    def _node_ask_email(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_ask_email")
        return {"result": self.faculty_agent.ask_for_email()}

    def _node_ask_group_emails(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_ask_group_emails")
        return {"result": self.faculty_agent.ask_for_group_emails()}

    def _node_ask_user_reference_data(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_ask_user_reference_data")
        return {"result": self.faculty_agent.ask_for_user_reference_data()}

    def _node_ask_grant_identifier(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_ask_grant_identifier")
        return {"result": self.opportunity_agent.ask_for_grant_identifier()}

    def _node_search_grant_by_link_in_db(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_search_grant_by_link_in_db")
        self.opportunity_agent.search_grant_by_link_in_db()
        if state.get("grant_in_db") is True:
            return {"decision": "generate_keywords_group_specific_grant"}
        return {"decision": "fetch_grant_from_source"}

    def _node_search_grant_by_title_in_db(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_search_grant_by_title_in_db")
        self.opportunity_agent.search_grant_by_title_in_db()
        if state.get("grant_in_db") is True:
            return {"decision": "generate_keywords_group_specific_grant"}
        return {"decision": "fetch_grant_from_source"}

    def _node_fetch_grant_from_source(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_fetch_grant_from_source")
        self.opportunity_agent.fetch_grant_from_source()
        return {
            "grant_in_db": True,
            "decision": "generate_keywords_group_specific_grant",
            "result": {"next_action": "fetched_grant_from_source"},
        }

    def _node_generate_keywords_one_to_one(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_generate_keywords_one_to_one")
        self.matching_agent.generate_keywords_for_one_to_one()
        return {"decision": "run_one_to_one_matching"}

    def _node_generate_keywords_group(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_generate_keywords_group")
        self.matching_agent.generate_keywords_for_group()
        return {"decision": "run_group_matching"}

    def _node_generate_keywords_group_specific_grant(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_generate_keywords_group_specific_grant")
        self.matching_agent.generate_keywords_for_group_specific_grant()
        return {"decision": "run_group_matching_with_specific_grant"}

    def _node_run_one_to_one_matching(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_one_to_one_matching")
        return {"result": self.matching_agent.run_one_to_one_matching()}

    def _node_run_group_matching(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_group_matching")
        return {"result": self.matching_agent.run_group_matching()}

    def _node_run_group_matching_with_specific_grant(self, _: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_group_matching_with_specific_grant")
        return {"result": self.matching_agent.run_group_matching_with_specific_grant()}

    def _build_graph(self, *, checkpointer: Any = None):
        try:
            from langgraph.graph import END, StateGraph
        except Exception as e:  # pragma: no cover - dependency guard
            raise ImportError("langgraph is required. Install with: pip install langgraph") from e

        graph = StateGraph(GrantMatchWorkflowState)

        graph.add_node("route", self._node_route)
        graph.add_node("decide_one_to_one", self._node_decide_one_to_one)
        graph.add_node("decide_group", self._node_decide_group)
        graph.add_node("decide_group_specific_grant", self._node_decide_group_specific_grant)

        graph.add_node("ask_email", self._node_ask_email)
        graph.add_node("ask_group_emails", self._node_ask_group_emails)
        graph.add_node("ask_user_reference_data", self._node_ask_user_reference_data)
        graph.add_node("ask_grant_identifier", self._node_ask_grant_identifier)

        graph.add_node("search_grant_by_link_in_db", self._node_search_grant_by_link_in_db)
        graph.add_node("search_grant_by_title_in_db", self._node_search_grant_by_title_in_db)
        graph.add_node("fetch_grant_from_source", self._node_fetch_grant_from_source)

        graph.add_node("generate_keywords_one_to_one", self._node_generate_keywords_one_to_one)
        graph.add_node("generate_keywords_group", self._node_generate_keywords_group)
        graph.add_node(
            "generate_keywords_group_specific_grant",
            self._node_generate_keywords_group_specific_grant,
        )

        graph.add_node("run_one_to_one_matching", self._node_run_one_to_one_matching)
        graph.add_node("run_group_matching", self._node_run_group_matching)
        graph.add_node(
            "run_group_matching_with_specific_grant",
            self._node_run_group_matching_with_specific_grant,
        )

        graph.set_entry_point("route")
        graph.add_conditional_edges("route", self._edge_from_route)
        graph.add_conditional_edges("decide_one_to_one", self._edge_from_decision)
        graph.add_conditional_edges("decide_group", self._edge_from_decision)
        graph.add_conditional_edges("decide_group_specific_grant", self._edge_from_decision)
        graph.add_conditional_edges("search_grant_by_link_in_db", self._edge_from_decision)
        graph.add_conditional_edges("search_grant_by_title_in_db", self._edge_from_decision)
        graph.add_conditional_edges("fetch_grant_from_source", self._edge_from_decision)
        graph.add_conditional_edges("generate_keywords_one_to_one", self._edge_from_decision)
        graph.add_conditional_edges("generate_keywords_group", self._edge_from_decision)
        graph.add_conditional_edges("generate_keywords_group_specific_grant", self._edge_from_decision)

        graph.add_edge("ask_email", END)
        graph.add_edge("ask_group_emails", END)
        graph.add_edge("ask_user_reference_data", END)
        graph.add_edge("ask_grant_identifier", END)
        graph.add_edge("run_one_to_one_matching", END)
        graph.add_edge("run_group_matching", END)
        graph.add_edge("run_group_matching_with_specific_grant", END)

        if checkpointer is not None:
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()

    def run(self, request: GrantMatchRequest, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        self._call("GrantMatchOrchestrator.run")
        state: GrantMatchWorkflowState = {
            "user_input": request.user_input,
            "email": request.email,
            "emails": list(request.emails or []),
            "faculty_in_db": request.faculty_in_db,
            "email_in_db": request.email_in_db,
            "grant_link": request.grant_link,
            "grant_title": request.grant_title,
            "grant_identifier_type": request.grant_identifier_type,
            "grant_in_db": request.grant_in_db,
            "grant_link_valid": request.grant_link_valid,
            "grant_title_confirmed": request.grant_title_confirmed,
        }

        if thread_id:
            out = self.graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        else:
            out = self.graph.invoke(state)

        return {
            "scenario": out.get("scenario") or "one_to_one",
            "email_detected": out.get("email_detected"),
            "emails_detected": out.get("emails_detected") or [],
            "grant_link_detected": out.get("grant_link_detected"),
            "grant_title_detected": out.get("grant_title_detected"),
            "result": out.get("result") or {},
        }

