from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class GroupQualityDTO:
    cov_app: float
    cov_res: float
    cov_total: float

    breadth_app: float
    breadth_res: float
    breadth_total: float

    critical_hit_app: float
    critical_hit_res: float
    critical_hit_total: float


@dataclass
class GroupMatchMetaDTO:
    algo: str
    penalty_source: str
    lambda_grid: List[float]
    quality: GroupQualityDTO


@dataclass
class GroupMatchResultDTO:
    group_id: int
    grant_id: str

    lambda_: float
    k: int
    top_n: int

    objective: float
    redundancy: float
    status: Optional[str]

    alpha: Dict[str, float]
    meta: GroupMatchMetaDTO