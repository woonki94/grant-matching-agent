from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings

# ------------------------------------------------------------
# Edit these manually
# ------------------------------------------------------------
KEYWORD_SET_TEXT = """
- multimodal robot learning
- sim-to-real transfer
- humanoid locomotion control
- reinforcement learning for contact-rich manipulation
""".strip()

FAC_SPEC_TEXT = """
This work focuses on path planning and obstacle avoidance for autonomous mobile robots operating in structured indoor environments. We develop efficient graph-based algorithms for navigation under dynamic constraints, without incorporating learning-based control or multimodal perception.
""".strip()

SYSTEM_PROMPT = """
You are scoring relevance between:
1) a specialization keyword set
2) a candidate faculty specialization text

Return ONLY strict JSON:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one short sentence>"
}

Scoring rubric:
- 0.9-1.0: direct and strong semantic match
- 0.7-0.89: clearly relevant with good overlap
- 0.4-0.69: partial/indirect relevance
- 0.1-0.39: weak relevance
- 0.0-0.09: unrelated
""".strip()

USER_PROMPT_TEMPLATE = """
Keyword set:
{keyword_set}

Candidate faculty specialization:
{fac_spec}
""".strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = str(item.get("text") or "").strip()
                if txt:
                    parts.append(txt)
            else:
                txt = str(item or "").strip()
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def main() -> int:
    model_id = (settings.haiku or "").strip()
    if not model_id:
        raise RuntimeError("settings.haiku is empty. Set BEDROCK_CLAUDE_HAIKU in .env.")

    llm = get_llm_client(model_id=model_id).build()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        keyword_set=KEYWORD_SET_TEXT,
        fac_spec=FAC_SPEC_TEXT,
    )
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    raw_response = _content_to_text(getattr(response, "content", response))
    parsed = _extract_json_object(raw_response) or {}

    score = _coerce_score(parsed.get("score"))
    reason = str(parsed.get("reason") or "").strip() or "No reason returned."

    print(f"model_id={model_id}")
    print("")
    print("keyword_set:")
    print(KEYWORD_SET_TEXT)
    print("")
    print("faculty_specialization:")
    print(FAC_SPEC_TEXT)
    print("")
    print(f"score={score:.4f}")
    print(f"reason={reason}")
    print("")
    print("raw_response:")
    print(raw_response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
