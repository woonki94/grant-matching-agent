from __future__ import annotations

import re
from typing import List

from tmp.agentic_arch.models import FacultyProfessionProfile, GrantSnapshot, OneToOneMatch


class OneToOneProfessionMatcher:
    """Simple one-to-one matcher between faculty profession focus and grant requirements."""

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = OneToOneProfessionMatcher._clean(text)
        return [t for t in re.split(r"[^a-z0-9]+", cleaned) if t]

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _score(
        self,
        *,
        profession_focus: List[str],
        grant_domains: List[str],
        grant_specs: List[str],
    ) -> tuple[float, List[str]]:
        matched: List[str] = []
        total = 0.0

        requirement_phrases = list(grant_domains or []) + list(grant_specs or [])
        if not profession_focus or not requirement_phrases:
            return 0.0, matched

        for prof in profession_focus:
            prof_clean = self._clean(prof)
            if not prof_clean:
                continue

            prof_tokens = self._tokenize(prof_clean)
            best_local = 0.0
            for req in requirement_phrases:
                req_clean = self._clean(req)
                if not req_clean:
                    continue

                if prof_clean in req_clean or req_clean in prof_clean:
                    local = 1.0
                else:
                    local = self._jaccard(prof_tokens, self._tokenize(req_clean))

                if local > best_local:
                    best_local = local

            if best_local >= 0.35:
                matched.append(prof)
            total += best_local

        score = total / max(len(profession_focus), 1)
        return float(score), matched

    def rank(
        self,
        *,
        faculty_profile: FacultyProfessionProfile,
        grants: List[GrantSnapshot],
        top_k: int = 5,
    ) -> List[OneToOneMatch]:
        rows: List[OneToOneMatch] = []

        for grant in grants or []:
            score, matched_prof = self._score(
                profession_focus=list(faculty_profile.profession_focus or []),
                grant_domains=list(grant.requirement.domains or []),
                grant_specs=list(grant.requirement.specializations or []),
            )

            rows.append(
                OneToOneMatch(
                    faculty_email=faculty_profile.email,
                    grant_id=grant.metadata.grant_id,
                    score=score,
                    reason=(
                        f"Profession overlap with grant requirements: "
                        f"{len(matched_prof)} matched focus terms"
                    ),
                    matched_professions=matched_prof,
                    grant_name=grant.metadata.grant_name,
                    agency_name=grant.metadata.agency_name,
                    close_date=grant.metadata.close_date,
                )
            )

        rows.sort(key=lambda x: (x.score, x.grant_id), reverse=True)
        return rows[: max(1, int(top_k or 5))]
