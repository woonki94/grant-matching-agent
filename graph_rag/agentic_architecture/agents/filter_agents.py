from __future__ import annotations

from typing import Any, Dict


class PreFilterAgent:
    def __init__(
        self,
        *,
        domain_cosine_threshold: float = 0.2,
    ):
        self.domain_cosine_threshold = float(domain_cosine_threshold)



