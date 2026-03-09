from __future__ import annotations

import asyncio
import re
from typing import Dict, List

from tmp.agentic_arch.tools import FacultyTools
from tmp.agentic_arch.models import FacultyProfessionProfile


class FacultyProfessionAgent:
    """
    Faculty agent:
    1) parallel tool calls for faculty context
    2) infer profession focus phrases from profile + publications + keywords
    """

    def __init__(self, tools: FacultyTools):
        self.tools = tools

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @staticmethod
    def _dedupe(values: List[str], max_items: int = 20) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in values:
            token = FacultyProfessionAgent._clean(raw)
            if not token:
                continue
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(token)
            if len(out) >= max_items:
                break
        return out

    def _infer_profession_focus(
        self,
        *,
        position: str,
        organizations: List[str],
        keywords: List[str],
        publication_abstracts: List[str],
        additional_text: List[str],
    ) -> List[str]:
        candidates: List[str] = []

        pos = self._clean(position)
        if pos:
            candidates.append(pos)
            m = re.search(r"\bof\b\s+(.+)$", pos, flags=re.IGNORECASE)
            if m:
                candidates.append(self._clean(m.group(1)))

        candidates.extend(self._clean(x) for x in organizations)
        candidates.extend(self._clean(x) for x in keywords)

        # Lightweight signal mining: keep noun-ish key snippets from abstracts/text.
        mined_text = " ".join(publication_abstracts + additional_text).lower()
        mining_phrases = [
            "machine learning",
            "artificial intelligence",
            "robotics",
            "computer vision",
            "data science",
            "reinforcement learning",
            "autonomous systems",
            "human-robot interaction",
            "systems engineering",
            "cybersecurity",
            "materials science",
            "biomedical",
            "agriculture",
        ]
        for phrase in mining_phrases:
            if phrase in mined_text:
                candidates.append(phrase)

        return self._dedupe(candidates, max_items=24)

    async def profile_profession(self, *, email: str) -> FacultyProfessionProfile:
        basic_task = self.tools.fetch_basic_info(email)
        keywords_task = self.tools.fetch_keywords(email)
        additional_task = self.tools.fetch_additional_text(email)
        pubs_task = self.tools.fetch_publications(email)

        basic_info, keywords, additional_text, publications = await asyncio.gather(
            basic_task,
            keywords_task,
            additional_task,
            pubs_task,
        )

        pub_abstracts = [self._clean(p.abstract or "") for p in publications if self._clean(p.abstract or "")]

        profession_focus = self._infer_profession_focus(
            position=basic_info.position or "",
            organizations=list(basic_info.organizations or []),
            keywords=list(keywords or []),
            publication_abstracts=pub_abstracts,
            additional_text=list(additional_text or []),
        )

        return FacultyProfessionProfile(
            email=str(email or "").strip().lower(),
            basic_info=basic_info,
            profession_focus=profession_focus,
            keywords=self._dedupe(list(keywords or []), max_items=40),
            evidence={
                "organizations": list(basic_info.organizations or []),
                "publication_abstracts": pub_abstracts[:8],
                "additional_text": [self._clean(x) for x in (additional_text or [])][:5],
            },
        )
