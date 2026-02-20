from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.agent_v2.router import IntentRouter
from services.agent_v2.state import GrantMatchRequest, GrantMatchWorkflowState
from services.agent_v2.agents import (
    FacultyContextAgent,
    GeneralConversationAgent,
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
        general_agent: Optional[GeneralConversationAgent] = None,
        checkpointer: Any = None,
    ):
        self.router = router or IntentRouter()
        self.faculty_agent = faculty_agent or FacultyContextAgent()
        self.opportunity_agent = opportunity_agent or OpportunityContextAgent()
        self.matching_agent = matching_agent or MatchingExecutionAgent()
        self.general_agent = general_agent or GeneralConversationAgent()
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
    def _resolve_team_size(state: GrantMatchWorkflowState, emails: List[str]) -> int:
        min_size = max(2, len(emails))
        requested = state.get("requested_team_size")
        if requested is None:
            return min_size
        try:
            return max(min_size, int(requested))
        except Exception:
            return min_size

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
        desired_broad_category = (
            state.get("desired_broad_category")
            or inferred.get("desired_broad_category")
        )
        topic_query = state.get("topic_query") or inferred.get("topic_query")
        requested_team_size = state.get("requested_team_size") or inferred.get("requested_team_size")
        requested_top_k_grants = state.get("requested_top_k_grants") or inferred.get("requested_top_k_grants")
        grant_identifier_type = (
            state.get("grant_identifier_type")
            or inferred.get("grant_identifier_type")
            or ("link" if grant_link else ("title" if grant_title else None))
        )

        # Business rule: if user provides 2+ faculty emails, force group routing.
        if len(emails) >= 2:
            scenario = "group_specific_grant" if (grant_link or grant_title or grant_identifier_type) else "group"

        return {
            "scenario": scenario,
            "email": email,
            "emails": emails,
            "grant_link": grant_link,
            "grant_title": grant_title,
            "desired_broad_category": desired_broad_category,
            "topic_query": topic_query,
            "requested_team_size": requested_team_size,
            "requested_top_k_grants": requested_top_k_grants,
            "grant_identifier_type": grant_identifier_type,
            "has_faculty_signal": bool(inferred.get("has_faculty_signal")),
            "has_group_signal": bool(inferred.get("has_group_signal")),
            "has_grant_signal": bool(inferred.get("has_grant_signal")),
            "email_detected": email,
            "emails_detected": emails,
            "grant_link_detected": grant_link,
            "grant_title_detected": grant_title,
            "desired_broad_category_detected": desired_broad_category,
            "topic_query_detected": topic_query,
            "requested_team_size_detected": requested_team_size,
            "requested_top_k_grants_detected": requested_top_k_grants,
        }

    def _edge_from_route(self, state: GrantMatchWorkflowState) -> str:
        scenario = state.get("scenario")
        if scenario == "general":
            return "run_general_response"
        if scenario == "group_specific_grant":
            return "decide_group_specific_grant"
        if scenario == "group":
            return "decide_group"
        return "decide_one_to_one"

    def _node_decide_one_to_one(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_decide_one_to_one")
        if not state.get("email"):
            return {"decision": "ask_email"}
        resolved = self.faculty_agent.resolve_faculties(emails=[str(state.get("email"))])
        if not resolved.get("all_in_db"):
            return {"decision": "ask_user_reference_data"}
        faculty_ids = list(resolved.get("faculty_ids") or [])
        resolved_state: GrantMatchWorkflowState = {
            "faculty_ids": faculty_ids,
            "missing_emails": list(resolved.get("missing_emails") or []),
        }
        ident = self._resolve_grant_identifier_type(state)
        if ident == "link":
            if not state.get("grant_link"):
                return {"decision": "ask_grant_identifier"}
            return {**resolved_state, "decision": "search_grant_by_link_in_db"}
        if ident == "title":
            if not state.get("grant_title"):
                return {"decision": "ask_grant_identifier"}
            return {**resolved_state, "decision": "search_grant_by_title_in_db"}
        return {**resolved_state, "decision": "run_one_to_one_matching"}

    def _node_decide_group(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_decide_group")
        emails = list(state.get("emails") or [])
        if len(emails) < 2:
            return {"decision": "ask_group_emails"}
        resolved = self.faculty_agent.resolve_faculties(emails=emails)
        if not resolved.get("all_in_db"):
            return {"decision": "ask_user_reference_data"}
        return {
            "emails": list(resolved.get("emails") or emails),
            "faculty_ids": list(resolved.get("faculty_ids") or []),
            "missing_emails": list(resolved.get("missing_emails") or []),
            "decision": "generate_keywords_group",
        }

    def _node_decide_group_specific_grant(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_decide_group_specific_grant")
        emails = list(state.get("emails") or [])
        if len(emails) < 2:
            return {"decision": "ask_group_emails"}
        resolved = self.faculty_agent.resolve_faculties(emails=emails)
        if not resolved.get("all_in_db"):
            return {"decision": "ask_user_reference_data"}

        resolved_state: GrantMatchWorkflowState = {
            "emails": list(resolved.get("emails") or emails),
            "faculty_ids": list(resolved.get("faculty_ids") or []),
            "missing_emails": list(resolved.get("missing_emails") or []),
        }

        ident = self._resolve_grant_identifier_type(state)
        if ident == "link":
            if not state.get("grant_link"):
                return {"decision": "ask_grant_identifier"}
            return {**resolved_state, "decision": "search_grant_by_link_in_db"}
        if ident == "title":
            if not state.get("grant_title"):
                return {"decision": "ask_grant_identifier"}
            return {**resolved_state, "decision": "search_grant_by_title_in_db"}
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

    def _node_run_general_response(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_general_response")
        return {
            "result": self.general_agent.answer_briefly(
                user_input=str(state.get("user_input") or ""),
            )
        }

    def _node_search_grant_by_link_in_db(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_search_grant_by_link_in_db")
        found = self.opportunity_agent.search_grant_by_link_in_db(
            grant_link=str(state.get("grant_link") or ""),
        )
        next_step = (
            "generate_keywords_group_specific_grant"
            if state.get("scenario") == "group_specific_grant"
            else "generate_keywords_one_to_one_specific_grant"
        )
        if found.get("found"):
            return {
                "opportunity_id": found.get("opportunity_id"),
                "opportunity_title": found.get("opportunity_title"),
                "decision": next_step,
            }
        return {"decision": "fetch_grant_from_source"}

    def _node_search_grant_by_title_in_db(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_search_grant_by_title_in_db")
        found = self.opportunity_agent.search_grant_by_title_in_db(
            grant_title=str(state.get("grant_title") or ""),
        )
        next_step = (
            "generate_keywords_group_specific_grant"
            if state.get("scenario") == "group_specific_grant"
            else "generate_keywords_one_to_one_specific_grant"
        )
        if found.get("found"):
            return {
                "opportunity_id": found.get("opportunity_id"),
                "opportunity_title": found.get("opportunity_title"),
                "decision": next_step,
            }
        return {"decision": "fetch_grant_from_source"}

    def _node_fetch_grant_from_source(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_fetch_grant_from_source")
        fetched = self.opportunity_agent.fetch_grant_from_source(
            grant_identifier_type=self._resolve_grant_identifier_type(state),
            grant_link=state.get("grant_link"),
            grant_title=state.get("grant_title"),
        )
        if not fetched.get("fetched"):
            return {"decision": "ask_grant_identifier"}
        next_step = (
            "generate_keywords_group_specific_grant"
            if state.get("scenario") == "group_specific_grant"
            else "generate_keywords_one_to_one_specific_grant"
        )
        return {
            "grant_in_db": bool(fetched.get("fetched")),
            "opportunity_id": fetched.get("opportunity_id"),
            "opportunity_title": fetched.get("opportunity_title"),
            "decision": next_step,
            "result": {"next_action": "fetched_grant_from_source"},
        }

    def _node_generate_keywords_one_to_one_specific_grant(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_generate_keywords_one_to_one_specific_grant")
        faculty_ids = [int(x) for x in (state.get("faculty_ids") or [])]
        opp_id = str(state.get("opportunity_id") or "")
        if not faculty_ids:
            return {"decision": "ask_user_reference_data"}
        if not opp_id:
            return {"decision": "ask_grant_identifier"}
        out = self.matching_agent.generate_keywords_and_matches_for_one_to_one_specific_grant(
            faculty_id=int(faculty_ids[0]),
            opportunity_id=opp_id,
        )
        if str(out.get("next_action", "")).startswith("error_"):
            return {"result": out, "decision": "return_error"}
        return {"decision": "run_one_to_one_matching_with_specific_grant"}

    def _node_generate_keywords_group(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_generate_keywords_group")
        faculty_ids = [int(x) for x in (state.get("faculty_ids") or [])]
        if not faculty_ids:
            return {"decision": "ask_user_reference_data"}
        out = self.matching_agent.generate_keywords_for_group(faculty_ids=faculty_ids)
        if str(out.get("next_action", "")).startswith("error_"):
            return {"result": out, "decision": "return_error"}
        return {"decision": "run_group_matching"}

    def _node_generate_keywords_group_specific_grant(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_generate_keywords_group_specific_grant")
        faculty_ids = [int(x) for x in (state.get("faculty_ids") or [])]
        opp_id = str(state.get("opportunity_id") or "")
        emails = list(state.get("emails") or [])
        if not faculty_ids:
            return {"decision": "ask_user_reference_data"}
        if not opp_id:
            return {"decision": "ask_grant_identifier"}
        team_size = self._resolve_team_size(state, emails)
        out = self.matching_agent.generate_keywords_and_matches_for_group_specific_grant(
            faculty_ids=faculty_ids,
            opportunity_id=opp_id,
            team_size=team_size,
        )
        if str(out.get("next_action", "")).startswith("error_"):
            return {"result": out, "decision": "return_error"}
        return {"decision": "run_group_matching_with_specific_grant"}

    def _node_run_one_to_one_matching(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_one_to_one_matching")
        faculty_ids = list(state.get("faculty_ids") or [])
        if not faculty_ids:
            return {"result": self.faculty_agent.ask_for_user_reference_data()}
        return {
            "result": self.matching_agent.run_one_to_one_matching(
                faculty_id=int(faculty_ids[0]),
                top_k=int(state.get("requested_top_k_grants") or 10),
                broad_category=state.get("desired_broad_category"),
                query_text=state.get("topic_query"),
            )
        }

    def _node_run_one_to_one_matching_with_specific_grant(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_one_to_one_matching_with_specific_grant")
        faculty_ids = list(state.get("faculty_ids") or [])
        opp_id = str(state.get("opportunity_id") or "")
        if not faculty_ids:
            return {"result": self.faculty_agent.ask_for_user_reference_data()}
        if not opp_id:
            return {"result": self.opportunity_agent.ask_for_grant_identifier()}
        return {
            "result": self.matching_agent.run_one_to_one_matching_with_specific_grant(
                faculty_id=int(faculty_ids[0]),
                opportunity_id=opp_id,
                top_k_grants=state.get("requested_top_k_grants"),
            )
        }

    def _node_run_one_to_one_specific_grant_justification(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_one_to_one_specific_grant_justification")
        result = dict(state.get("result") or {})
        if not result:
            return {"result": result}
        if str(result.get("next_action", "")).startswith("error_"):
            return {"result": result}
        if result.get("next_action") != "return_one_to_one_results":
            return {"result": result}
        if not list(result.get("matches") or []):
            return {"result": result}

        faculty_email = str(result.get("faculty_email") or "").strip()
        opp_id = str(result.get("opportunity_id") or "").strip()
        if not faculty_email or not opp_id:
            return {"result": result}

        return {
            "result": self.matching_agent.run_one_to_one_specific_grant_justification(
                faculty_email=faculty_email,
                opportunity_id=opp_id,
                base_result=result,
            )
        }

    def _node_run_group_matching(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_group_matching")
        emails = list(state.get("emails") or [])
        if len(emails) < 2:
            return {"result": self.faculty_agent.ask_for_group_emails()}
        team_size = self._resolve_team_size(state, emails)
        return {
            "result": self.matching_agent.run_group_matching(
                faculty_emails=emails,
                team_size=team_size,
                top_k_grants=state.get("requested_top_k_grants"),
                broad_category=state.get("desired_broad_category"),
                query_text=state.get("topic_query"),
            )
        }

    def _node_run_group_matching_with_specific_grant(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_run_group_matching_with_specific_grant")
        emails = list(state.get("emails") or [])
        opp_id = str(state.get("opportunity_id") or "")
        if len(emails) < 2:
            return {"result": self.faculty_agent.ask_for_group_emails()}
        if not opp_id:
            return {"result": self.opportunity_agent.ask_for_grant_identifier()}
        team_size = self._resolve_team_size(state, emails)
        return {
            "result": self.matching_agent.run_group_matching_with_specific_grant(
                faculty_emails=emails,
                opportunity_id=opp_id,
                team_size=team_size,
                top_k_grants=state.get("requested_top_k_grants"),
                broad_category=state.get("desired_broad_category"),
                query_text=state.get("topic_query"),
            )
        }

    def _node_return_error(self, state: GrantMatchWorkflowState) -> GrantMatchWorkflowState:
        self._call("GrantMatchOrchestrator._node_return_error")
        return {"result": state.get("result") or {"next_action": "error"}}

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
        graph.add_node("run_general_response", self._node_run_general_response)

        graph.add_node("search_grant_by_link_in_db", self._node_search_grant_by_link_in_db)
        graph.add_node("search_grant_by_title_in_db", self._node_search_grant_by_title_in_db)
        graph.add_node("fetch_grant_from_source", self._node_fetch_grant_from_source)

        graph.add_node("generate_keywords_group", self._node_generate_keywords_group)
        graph.add_node(
            "generate_keywords_group_specific_grant",
            self._node_generate_keywords_group_specific_grant,
        )
        graph.add_node(
            "generate_keywords_one_to_one_specific_grant",
            self._node_generate_keywords_one_to_one_specific_grant,
        )

        graph.add_node("run_one_to_one_matching", self._node_run_one_to_one_matching)
        graph.add_node(
            "run_one_to_one_matching_with_specific_grant",
            self._node_run_one_to_one_matching_with_specific_grant,
        )
        graph.add_node(
            "run_one_to_one_specific_grant_justification",
            self._node_run_one_to_one_specific_grant_justification,
        )
        graph.add_node("run_group_matching", self._node_run_group_matching)
        graph.add_node(
            "run_group_matching_with_specific_grant",
            self._node_run_group_matching_with_specific_grant,
        )
        graph.add_node("return_error", self._node_return_error)

        graph.set_entry_point("route")
        graph.add_conditional_edges("route", self._edge_from_route)
        graph.add_conditional_edges("decide_one_to_one", self._edge_from_decision)
        graph.add_conditional_edges("decide_group", self._edge_from_decision)
        graph.add_conditional_edges("decide_group_specific_grant", self._edge_from_decision)
        graph.add_conditional_edges("search_grant_by_link_in_db", self._edge_from_decision)
        graph.add_conditional_edges("search_grant_by_title_in_db", self._edge_from_decision)
        graph.add_conditional_edges("fetch_grant_from_source", self._edge_from_decision)
        graph.add_conditional_edges("generate_keywords_group", self._edge_from_decision)
        graph.add_conditional_edges("generate_keywords_group_specific_grant", self._edge_from_decision)
        graph.add_conditional_edges("generate_keywords_one_to_one_specific_grant", self._edge_from_decision)

        graph.add_edge("ask_email", END)
        graph.add_edge("ask_group_emails", END)
        graph.add_edge("ask_user_reference_data", END)
        graph.add_edge("ask_grant_identifier", END)
        graph.add_edge("run_general_response", END)
        graph.add_edge("run_one_to_one_matching", END)
        graph.add_edge("run_one_to_one_matching_with_specific_grant", "run_one_to_one_specific_grant_justification")
        graph.add_edge("run_one_to_one_specific_grant_justification", END)
        graph.add_edge("run_group_matching", END)
        graph.add_edge("run_group_matching_with_specific_grant", END)
        graph.add_edge("return_error", END)

        if checkpointer is not None:
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()

    def _build_initial_state(self, request: GrantMatchRequest) -> GrantMatchWorkflowState:
        return {
            "user_input": request.user_input,
            "email": request.email,
            "emails": list(request.emails or []),
            "faculty_in_db": request.faculty_in_db,
            "email_in_db": request.email_in_db,
            "grant_link": request.grant_link,
            "grant_title": request.grant_title,
            "desired_broad_category": request.desired_broad_category,
            "topic_query": request.topic_query,
            "requested_team_size": request.requested_team_size,
            "requested_top_k_grants": request.requested_top_k_grants,
            "grant_identifier_type": request.grant_identifier_type,
            "grant_in_db": request.grant_in_db,
            "grant_link_valid": request.grant_link_valid,
            "grant_title_confirmed": request.grant_title_confirmed,
        }

    @staticmethod
    def _format_output(out: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(out.get("result") or {})
        top_k = result.get("top_k_grants")
        if top_k is None:
            top_k = out.get("requested_top_k_grants_detected")
        try:
            k = int(top_k) if top_k is not None else None
        except Exception:
            k = None
        if k is not None and k > 0:
            if isinstance(result.get("matches"), list):
                result["matches"] = list(result["matches"])[:k]
            recommendation = result.get("recommendation")
            if isinstance(recommendation, dict) and isinstance(recommendation.get("recommendations"), list):
                recommendation = dict(recommendation)
                recommendation["recommendations"] = list(recommendation["recommendations"])[:k]
                result["recommendation"] = recommendation
        return {
            "scenario": out.get("scenario") or "one_to_one",
            "email_detected": out.get("email_detected"),
            "emails_detected": out.get("emails_detected") or [],
            "grant_link_detected": out.get("grant_link_detected"),
            "grant_title_detected": out.get("grant_title_detected"),
            "desired_broad_category_detected": out.get("desired_broad_category_detected"),
            "topic_query_detected": out.get("topic_query_detected"),
            "requested_team_size_detected": out.get("requested_team_size_detected"),
            "requested_top_k_grants_detected": out.get("requested_top_k_grants_detected"),
            "result": result,
        }

    def stream(self, request: GrantMatchRequest, *, thread_id: Optional[str] = None):
        self._call("GrantMatchOrchestrator.stream")
        state = self._build_initial_state(request)
        final_state: Dict[str, Any] = dict(state)

        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        chunks = self.graph.stream(state, config=config, stream_mode="updates")

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                if isinstance(update, dict):
                    final_state.update(update)
                yield {
                    "type": "step",
                    "node": node_name,
                    "update": update if isinstance(update, dict) else {},
                }

        yield {
            "type": "final",
            "output": self._format_output(final_state),
        }

    def run(self, request: GrantMatchRequest, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        self._call("GrantMatchOrchestrator.run")
        state = self._build_initial_state(request)

        if thread_id:
            out = self.graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        else:
            out = self.graph.invoke(state)

        return self._format_output(out)
