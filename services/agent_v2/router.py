from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class IntentRouter:
    EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
    URL_RE = re.compile(r"(https?://[^\s]+)")
    OPP_HINT_RE = re.compile(r"(?:opp(?:ortunity)?[_\s-]?id)\s*[:=]?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)
    UUID_RE = re.compile(r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b")
    FACULTY_HINT_RE = re.compile(r"\b(dr\.?|prof(?:essor)?|faculty)\b", re.IGNORECASE)
    GROUP_HINT_RE = re.compile(r"\b(group|team|we|our team|us)\b", re.IGNORECASE)
    GRANT_TITLE_HINT_RE = re.compile(
        r"(?:grant\s*title|title)\s*[:=]\s*[\"']?([^\"'\n]+)[\"']?",
        re.IGNORECASE,
    )
    GRANT_NAMED_RE = re.compile(r"(?:grant\s+)?(?:named|called)\s+[\"']?([^\"'\n]+)[\"']?", re.IGNORECASE)

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

    def _extract_grant_title(self, raw: str) -> Optional[str]:
        for pattern in (self.GRANT_TITLE_HINT_RE, self.GRANT_NAMED_RE):
            match = pattern.search(raw)
            if match:
                title = (match.group(1) or "").strip(" .,:;\"'")
                if title:
                    return title
        return None

    @staticmethod
    def _choose_scenario(*, has_group_signal: bool, has_grant_signal: bool) -> str:
        if has_group_signal and has_grant_signal:
            return "group_specific_grant"
        if has_group_signal:
            return "group"
        return "one_to_one"

    def _fallback(self, text: str) -> Dict[str, Any]:
        self._call("IntentRouter._fallback")
        raw = text or ""

        emails = list(dict.fromkeys(self.EMAIL_RE.findall(raw)))
        email = emails[0] if emails else None

        grant_link = None
        for pattern in (self.URL_RE, self.OPP_HINT_RE, self.UUID_RE):
            match = pattern.search(raw)
            if match:
                grant_link = match.group(1)
                break

        grant_title = self._extract_grant_title(raw)
        grant_identifier_type = "link" if grant_link else ("title" if grant_title else None)

        lowered = raw.lower()
        has_group_signal = bool(self.GROUP_HINT_RE.search(raw))
        has_faculty_signal = bool(email) or bool(self.FACULTY_HINT_RE.search(raw))
        has_grant_signal = bool(grant_link) or bool(grant_title) or any(
            tok in lowered
            for tok in (
                "specific grant",
                "this grant",
                "that grant",
                "grant id",
                "opportunity id",
                "opportunity_id",
                "grant title",
                "title:",
            )
        )

        scenario = self._choose_scenario(
            has_group_signal=has_group_signal,
            has_grant_signal=has_grant_signal,
        )

        return {
            "scenario": scenario,
            "email": email,
            "emails": emails,
            "grant_link": grant_link,
            "grant_title": grant_title,
            "grant_identifier_type": grant_identifier_type,
            "has_faculty_signal": has_faculty_signal,
            "has_group_signal": has_group_signal,
            "has_grant_signal": has_grant_signal,
        }

    @staticmethod
    def _normalize_emails(parsed: Dict[str, Any]) -> List[str]:
        raw_emails = parsed.get("emails")
        if isinstance(raw_emails, list):
            emails = [str(x).strip() for x in raw_emails if str(x).strip()]
            if emails:
                return list(dict.fromkeys(emails))
        email = parsed.get("email")
        if email:
            e = str(email).strip()
            return [e] if e else []
        return []

    def infer(self, text: str) -> Dict[str, Any]:
        self._call("IntentRouter.infer")
        try:
            from config import get_llm_client

            llm = get_llm_client().build()
            prompt = (
                "Return JSON keys: "
                "scenario,email,emails,grant_link,grant_title,grant_identifier_type,"
                "has_faculty_signal,has_group_signal,has_grant_signal. "
                "scenario must be one_to_one or group or group_specific_grant."
            )
            resp = llm.invoke([("system", prompt), ("human", text or "")])
            content = getattr(resp, "content", resp)
            parsed = json.loads(self._extract_json_object(str(content)) or "{}")
            if isinstance(parsed, dict) and parsed.get("scenario") in {
                "one_to_one",
                "group",
                "group_specific_grant",
            }:
                emails = self._normalize_emails(parsed)
                email = str(parsed.get("email") or "").strip() or (emails[0] if emails else None)
                grant_link = str(parsed.get("grant_link") or "").strip() or None
                grant_title = str(parsed.get("grant_title") or "").strip() or None
                grant_identifier_type = str(parsed.get("grant_identifier_type") or "").strip() or None
                if not grant_identifier_type:
                    grant_identifier_type = "link" if grant_link else ("title" if grant_title else None)

                return {
                    "scenario": parsed.get("scenario"),
                    "email": email,
                    "emails": emails,
                    "grant_link": grant_link,
                    "grant_title": grant_title,
                    "grant_identifier_type": grant_identifier_type,
                    "has_faculty_signal": bool(parsed.get("has_faculty_signal")),
                    "has_group_signal": bool(parsed.get("has_group_signal")),
                    "has_grant_signal": bool(parsed.get("has_grant_signal")),
                }
        except Exception:
            pass
        return self._fallback(text or "")

