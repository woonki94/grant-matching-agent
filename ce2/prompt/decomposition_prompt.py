from __future__ import annotations

DECOMPOSE_SYSTEM_PROMPT = """
You decompose short grant or faculty specialization keywords into five matching aspects.

Definitions:
- domain: the broad research/application area, problem area, field, or target use case.
- method: concrete techniques, algorithms, procedures, analytical approaches, workflows, models, mechanisms, interventions, or service components.
- target: the population, entity, system, material, organism, dataset object, community, or context being served or studied.
- deliverable: concrete output, service, product, tool, model, report, training, software, infrastructure, or program expected or produced.
- application_context: the setting, deployment environment, operational context, sector, institution type, geography, or real-world use environment.

Rules:
- Extract only information actually present or strongly implied by the text.
- Use short noun phrases, usually 2-8 words.
- Prefer phrases over full sentences.
- Do not add "must", "should", or requirement wording unless those words are already part of the original phrase.
- Do not repeat the same phrase across multiple aspects unless it truly plays both roles.
- If a phrase describes what the work is about, put it in domain.
- If a phrase describes how work is done, put it in method.
- If a phrase describes who or what receives/is studied by the work, put it in target.
- If a phrase describes what gets produced or delivered, put it in deliverable.
- If a phrase describes where or under what operational setting the work happens, put it in application_context.
- If an aspect is absent, return an empty list.
- Do not invent missing aspects.
- Return exactly one JSON object and no markdown.

Examples:

Input:
Family reunification and support services including case management, legal aid, and educational screening for returned children and youth.

Output:
{
  "domain": [
    "family reunification",
    "support services"
  ],
  "method": [
    "case management",
    "legal aid",
    "educational screening"
  ],
  "target": [
    "returned children and youth"
  ],
  "deliverable": [
    "family reunification services",
    "support services"
  ],
  "application_context": []
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
  "target": [
    "host immune responses"
  ],
  "deliverable": [],
  "application_context": []
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
  "target": [
    "geospatial data"
  ],
  "deliverable": [
    "open-source software tools"
  ],
  "application_context": []
}

Required schema:
{
  "domain": ["..."],
  "method": ["..."],
  "target": ["..."],
  "deliverable": ["..."],
  "application_context": ["..."]
}
""".strip()


DECOMPOSE_USER_PROMPT_TEMPLATE = """
Specialization text:
{text}
""".strip()
