from __future__ import annotations

from typing import Any, Dict, List

from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal


class FacultyContextAgent:
    def __init__(self, *, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _normalize_emails(emails: List[str]) -> List[str]:
        out: List[str] = []
        for e in emails or []:
            x = str(e or "").strip().lower()
            if x and x not in out:
                out.append(x)
        return out

    def resolve_faculties(self, *, emails: List[str]) -> Dict[str, Any]:
        self._call("FacultyContextAgent.resolve_faculties")
        normalized = self._normalize_emails(emails)
        faculty_ids: List[int] = []
        missing: List[str] = []
        try:
            with self.session_factory() as sess:
                dao = FacultyDAO(sess)
                for email in normalized:
                    fid = dao.get_faculty_id_by_email(email)
                    if fid is None:
                        missing.append(email)
                    else:
                        faculty_ids.append(int(fid))
        except Exception as e:
            return {
                "emails": normalized,
                "faculty_ids": [],
                "missing_emails": normalized,
                "all_in_db": False,
                "error": f"{type(e).__name__}: {e}",
            }

        return {
            "emails": normalized,
            "faculty_ids": faculty_ids,
            "missing_emails": missing,
            "all_in_db": len(missing) == 0 and len(faculty_ids) == len(normalized),
        }

    def ask_for_email(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_email")
        return {"next_action": "ask_email"}

    def ask_for_group_emails(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_group_emails")
        return {"next_action": "ask_group_emails"}

    def ask_for_user_reference_data(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_user_reference_data")
        return {"next_action": "ask_user_reference_data"}


