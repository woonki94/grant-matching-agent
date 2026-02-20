from __future__ import annotations

from typing import Any, Dict, List

from config import get_llm_client


class GeneralConversationAgent:
    def __init__(self):
        self.llm = None

    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _as_text(resp: Any) -> str:
        content = getattr(resp, "content", resp)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    txt = str(item.get("text") or "").strip()
                    if txt:
                        parts.append(txt)
            return " ".join(parts).strip()
        return str(content).strip()

    def answer_briefly(self, *, user_input: str) -> Dict[str, Any]:
        self._call("GeneralConversationAgent.answer_briefly")
        header = "[Grant Match Assistant]"
        try:
            if self.llm is None:
                self.llm = get_llm_client().build()
            prompt = (
                "You are a specialized grant-matching assistant. "
                "If user asks non-grant/general question, answer very briefly in 1-2 sentences. "
                "Keep it helpful and concise."
            )
            resp = self.llm.invoke([("system", prompt), ("human", user_input or "")])
            msg = self._as_text(resp) or "I focus on grant matching. Ask me about grant search or faculty-grant fit."
            return {"next_action": "general_reply", "message": f"{header} {msg}"}
        except Exception:
            return {
                "next_action": "general_reply",
                "message": f"{header} I focus on grant matching. Please ask about grants or matching.",
            }


