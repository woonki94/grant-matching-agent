from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from tmp.agentic_arch.orchestrator import ProfessionGrantMatchOrchestrator
from tmp.agentic_arch.tools import (
    InMemoryFacultyRecord,
    InMemoryFacultyTools,
    InMemoryGrantRecord,
    InMemoryGrantTools,
)
from tmp.agentic_arch.models import FacultyBasicInfo, FacultyPublication, GrantMetadata


def build_demo_orchestrator() -> ProfessionGrantMatchOrchestrator:
    faculty_tools = InMemoryFacultyTools(
        {
            "alan.fern@oregonstate.edu": InMemoryFacultyRecord(
                basic_info=FacultyBasicInfo(
                    email="alan.fern@oregonstate.edu",
                    faculty_name="Alan Fern",
                    position="Professor",
                    organizations=[
                        "Electrical Engineering and Computer Science",
                        "Collaborative Robotics and Intelligent Systems Institute",
                    ],
                ),
                keywords=[
                    "robotics",
                    "reinforcement learning",
                    "autonomous systems",
                    "planning",
                ],
                additional_text=[
                    "Research in robot learning, planning, and decision making.",
                    "Focus on embodied AI and autonomous robotics systems.",
                ],
                publications=[
                    FacultyPublication(
                        title="Learning and planning for dynamic robots",
                        abstract="We study reinforcement learning and model-based planning for legged robots.",
                        year=2023,
                    )
                ],
            )
        }
    )

    grant_tools = InMemoryGrantTools(
        {
            "g-001": InMemoryGrantRecord(
                metadata=GrantMetadata(
                    grant_id="g-001",
                    grant_name="Embodied AI and Robotics",
                    agency_name="U.S. National Science Foundation",
                    close_date="2026-10-15",
                ),
                domains=["robotics", "autonomous systems"],
                specializations=["reinforcement learning", "planning"],
                eligibility=["universities"],
                deliverables=["prototype", "evaluation report"],
            ),
            "g-002": InMemoryGrantRecord(
                metadata=GrantMetadata(
                    grant_id="g-002",
                    grant_name="Biomedical Imaging Program",
                    agency_name="NIH",
                    close_date="2026-07-30",
                ),
                domains=["biomedical", "imaging"],
                specializations=["signal processing", "clinical validation"],
                eligibility=["universities"],
                deliverables=["dataset", "publication"],
            ),
        }
    )

    return ProfessionGrantMatchOrchestrator(
        faculty_tools=faculty_tools,
        grant_tools=grant_tools,
    )


def main() -> int:
    orchestrator = build_demo_orchestrator()
    out = orchestrator.run_one_to_one_sync(
        faculty_email="alan.fern@oregonstate.edu",
        candidate_grant_k=10,
        result_top_k=3,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
