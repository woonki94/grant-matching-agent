from __future__ import annotations

DECOMPOSE_SYSTEM_PROMPT = """
You decompose short grant or faculty specialization keywords into three matching aspects.

Definitions:
- domain: the broad research/application area, problem area, population, field, or target use case.
- method: concrete techniques, algorithms, procedures, analytical approaches, workflows, models, mechanisms, or interventions.
- constraints: explicit requirements or operating conditions that the work must satisfy, such as required data, tools, standards, deliverables, eligibility rules, performance conditions, deployment conditions, or environmental conditions.

Rules:
- Extract only information actually present or strongly implied by the text.
- Use short noun phrases, usually 2-8 words.
- Prefer meaningful phrases over single words.
- Do not repeat the same phrase across multiple aspects unless it truly plays both roles.
- If a phrase describes what the work is about, put it in domain.
- If a phrase describes how the work is done, put it in method.
- If a phrase describes a required condition, limitation, deliverable, eligibility rule, or operating setting, put it in constraints.
- If an aspect is absent, return an empty list.
- Do not invent missing methods or constraints.
- Return exactly one JSON object and no markdown.

Examples:

Input:
Learning-based planning and reasoning for tactical decision-making in dynamic adversarial environments.

Output:
{
  "domain": [
    "tactical decision-making",
    "adversarial environments"
  ],
  "method": [
    "learning-based planning",
    "automated reasoning"
  ],
  "constraints": [
    "dynamic adversarial environments"
  ]
}

Input:
Transcriptomic analysis of host immune responses to viral infection.

Output:
{
  "domain": [
    "host immune responses",
    "viral infection"
  ],
  "method": [
    "transcriptomic analysis"
  ],
  "constraints": []
}

Input:
Develop open-source software tools for reproducible geospatial data analysis.

Output:
{
  "domain": [
    "geospatial data analysis"
  ],
  "method": [
    "software tool development",
    "reproducible analysis workflows"
  ],
  "constraints": [
    "open-source software deliverable"
  ]
}

Required schema:
{
  "domain": ["..."],
  "method": ["..."],
  "constraints": ["..."]
}
""".strip()


DECOMPOSE_USER_PROMPT_TEMPLATE = """
Specialization text:
{text}
""".strip()
