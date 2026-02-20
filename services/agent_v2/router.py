from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class IntentRouter:
    ALLOWED_BROAD_CATEGORIES = {"basic_research", "applied_research", "educational"}

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _extract_json_object(text: str) -> str:
        s = (text or "").strip()
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return s[i : j + 1]
        return ""

    @staticmethod
    def _normalize_emails(parsed: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        raw_emails = parsed.get("emails")
        if isinstance(raw_emails, list):
            for x in raw_emails:
                e = str(x or "").strip()
                if e and e not in out:
                    out.append(e)
        email = str(parsed.get("email") or "").strip()
        if email and email not in out:
            out.insert(0, email)
        return out

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "scenario": "one_to_one",
            "email": None,
            "emails": [],
            "grant_link": None,
            "grant_title": None,
            "grant_identifier_type": None,
            "desired_broad_category": None,
            "topic_query": None,
            "requested_team_size": None,
            "requested_top_k_grants": None,
            "has_faculty_signal": False,
            "has_group_signal": False,
            "has_grant_signal": False,
            "has_grant_intent": False,
        }

    @classmethod
    def _normalize_broad_category_filter(cls, raw: Any) -> Optional[List[str]]:
        if raw is None:
            return None

        out: List[str] = []
        if isinstance(raw, str):
            v = raw.strip().lower()
            if v in cls.ALLOWED_BROAD_CATEGORIES:
                out.append(v)
            return out or None

        if isinstance(raw, list):
            for item in raw:
                v = str(item or "").strip().lower()
                if v in cls.ALLOWED_BROAD_CATEGORIES and v not in out:
                    out.append(v)
            return out or None

        return None

    def _validate_general_route(
        self,
        *,
        llm: Any,
        user_text: str,
        routed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM-only guard: prevent grant requests from being mislabeled as general."""
        prompt = (
            "You are validating a router decision for a grant-matching assistant.\n"
            "Return ONLY JSON with keys: keep_general, corrected_scenario.\n"
            "corrected_scenario must be one_to_one, group, group_specific_grant, or general.\n"
            "Rule: If the user is asking about grants/funding/opportunity matching, keep_general=false.\n"
            "If keep_general=false, provide corrected_scenario as one_to_one/group/group_specific_grant.\n"
            "No markdown or explanations."
        )
        resp = llm.invoke(
            [
                ("system", prompt),
                (
                    "human",
                    f"USER_TEXT:\n{user_text or ''}\n\nROUTER_OUTPUT:\n{json.dumps(routed, ensure_ascii=False)}",
                ),
            ]
        )
        content = getattr(resp, "content", resp)
        parsed = json.loads(self._extract_json_object(str(content)) or "{}")
        if not isinstance(parsed, dict):
            return routed
        keep_general = bool(parsed.get("keep_general"))
        corrected = str(parsed.get("corrected_scenario") or "").strip()
        if corrected not in {"one_to_one", "group", "group_specific_grant", "general"}:
            corrected = routed.get("scenario") or "general"
        if keep_general:
            return routed
        out = dict(routed)
        out["scenario"] = corrected if corrected != "general" else "one_to_one"
        out["has_grant_intent"] = True
        return out

    def infer(self, text: str) -> Dict[str, Any]:
        self._call("IntentRouter.infer")
        try:
            from config import get_llm_client

            llm = get_llm_client().build()
            prompt = (
                "You are an intent router for a grant matching system.\n"
                "Return ONLY a JSON object with keys:\n"
                "scenario,email,emails,grant_link,grant_title,grant_identifier_type,"
                "desired_broad_category,topic_query,requested_team_size,requested_top_k_grants,"
                "has_faculty_signal,has_group_signal,has_grant_signal,has_grant_intent.\n"
                "Allowed scenario values: one_to_one, group, group_specific_grant, general.\n"
                "Never output scenario='general' for grant search/match/funding requests.\n"
                "grant_identifier_type must be one of: link, title, null.\n"
                "desired_broad_category must be one of: basic_research, applied_research, educational, "
                "a list of those values, or null.\n"
                "requested_team_size must be an integer >= 2 or null.\n"
                "requested_top_k_grants must be an integer >= 1 or null.\n"
                "FIELD-SEPARATION RULES:\n"
                "- desired_broad_category is ONLY for broad policy classes: basic_research, applied_research, educational.\n"
                "- topic_query is ONLY for topical semantic search phrases like agriculture, robotics, climate resilience.\n"
                "- Do NOT infer desired_broad_category from topical words (e.g., agriculture, robotics).\n"
                "- If the user asks broad type only (e.g., educational grants) and provides no topical theme, set topic_query to null.\n"
                "- If the user asks a topical theme only (e.g., related to agriculture), set desired_broad_category to null.\n"
                "- If user asks about a specific grant by title/link/id, set topic_query to null.\n"
                "- For exclusion requests (e.g., 'not educational'), set desired_broad_category to the INCLUDED list "
                "['basic_research','applied_research'] and keep topic_query null unless a separate topic exists.\n"
                "EXAMPLES:\n"
                "- Input: 'find matching grant that is related to agriculture' -> desired_broad_category=null, topic_query='agriculture'.\n"
                "- Input: 'find me educational grant' -> desired_broad_category='educational', topic_query=null.\n"
                "- Input: 'find grants that are not educational' -> desired_broad_category=['basic_research','applied_research'], topic_query=null.\n"
                "- Input: 'find group match with team size 4 including me' -> requested_team_size=4.\n"
                "- Input: 'show top 5 grants for me' -> requested_top_k_grants=5.\n"
                "For booleans, return true or false.\n"
                "Do not include markdown, code fences, or explanations."
            )
            resp = llm.invoke([("system", prompt), ("human", text or "")])
            content = getattr(resp, "content", resp)
            parsed = json.loads(self._extract_json_object(str(content)) or "{}")
            if not isinstance(parsed, dict):
                return self._empty_result()

            scenario = str(parsed.get("scenario") or "").strip()
            if scenario not in {"one_to_one", "group", "group_specific_grant", "general"}:
                return self._empty_result()

            emails = self._normalize_emails(parsed)
            email = str(parsed.get("email") or "").strip() or (emails[0] if emails else None)
            grant_link = str(parsed.get("grant_link") or "").strip() or None
            grant_title = str(parsed.get("grant_title") or "").strip() or None

            grant_identifier_type = str(parsed.get("grant_identifier_type") or "").strip().lower() or None
            if grant_identifier_type not in {"link", "title"}:
                grant_identifier_type = None

            desired_broad_category_list = self._normalize_broad_category_filter(parsed.get("desired_broad_category"))
            if desired_broad_category_list is None:
                desired_broad_category: Optional[str | List[str]] = None
            elif len(desired_broad_category_list) == 1:
                desired_broad_category = desired_broad_category_list[0]
            else:
                desired_broad_category = desired_broad_category_list

            topic_query = str(parsed.get("topic_query") or "").strip() or None
            requested_team_size = None
            raw_team_size = parsed.get("requested_team_size")
            if raw_team_size is not None:
                try:
                    requested_team_size = int(raw_team_size)
                except Exception:
                    requested_team_size = None
            if requested_team_size is not None and requested_team_size < 2:
                requested_team_size = None

            requested_top_k_grants = None
            raw_top_k = parsed.get("requested_top_k_grants")
            if raw_top_k is not None:
                try:
                    requested_top_k_grants = int(raw_top_k)
                except Exception:
                    requested_top_k_grants = None
            if requested_top_k_grants is not None and requested_top_k_grants < 1:
                requested_top_k_grants = None

            routed = {
                "scenario": scenario,
                "email": email,
                "emails": emails,
                "grant_link": grant_link,
                "grant_title": grant_title,
                "grant_identifier_type": grant_identifier_type,
                "desired_broad_category": desired_broad_category,
                "topic_query": topic_query,
                "requested_team_size": requested_team_size,
                "requested_top_k_grants": requested_top_k_grants,
                "has_faculty_signal": bool(parsed.get("has_faculty_signal")),
                "has_group_signal": bool(parsed.get("has_group_signal")),
                "has_grant_signal": bool(parsed.get("has_grant_signal")),
                "has_grant_intent": bool(parsed.get("has_grant_intent")),
            }
            if routed.get("scenario") == "general":
                routed = self._validate_general_route(llm=llm, user_text=text or "", routed=routed)
            return routed
        except Exception:
            return self._empty_result()
