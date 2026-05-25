from __future__ import annotations


DECOMPOSITION_BASE_RULES = """
Rules:
- Extract only information actually present or strongly implied by the text.
- Use short keyword phrases only.
- Prefer phrases over full sentences.
- Do not explain your reasoning.
- Do not output chain-of-thought or <think> blocks.
- Do not add "must", "should", or requirement wording unless present in source.
- If the aspect is absent, return an empty list.
- Do not invent missing aspects.
- Lowercase unless proper nouns.
- Return exactly one JSON object and no markdown.

Required schema:
{
  "items": ["..."]
}
""".strip()


DOMAIN_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract DOMAIN phrases from a short grant or faculty specialization keyword.

Domain means the broad research/application area, problem area, field, or target use case.

Length rule:
- each domain item must be 1-3 words.

Include phrases about:
- research or application topic
- problem area
- scientific/technical field
- high-level service or use area

Do not include:
- concrete methods or procedures
- populations/entities unless they define the topic area itself

{DECOMPOSITION_BASE_RULES}
""".strip()


METHOD_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract METHOD phrases from a short grant or faculty specialization keyword.

Method means concrete techniques, algorithms, procedures, analytical approaches, workflows, models, mechanisms, interventions, or service components.

Length rule:
- each method item must be 1-4 words.

Include phrases about:
- concrete methods or techniques
- algorithms, models, instruments, mechanisms, or procedures
- service components such as case management, legal aid, screening, counseling, training, or assessment
- analytical or implementation workflows
- action/process phrases such as managing, identifying, evaluating, measuring, designing, developing, creating, distributing, implementing
- "experience/expertise/skill in X" patterns: extract X as method phrase when X is a concrete process

Do not include:
- broad topic/domain phrases
- populations/entities unless they are part of a method phrase
- settings or application environments unless central to the method phrase

{DECOMPOSITION_BASE_RULES}
""".strip()


TARGET_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract TARGET phrases from a short grant or faculty specialization keyword.

Target means the population, entity, system, material, organism, dataset object, community, or context being served or studied.

Length rule:
- each target item must be 1-4 words.
- if target is unclear or implicit only, return [].

Include phrases about:
- populations, communities, beneficiaries, stakeholders, or study subjects
- entities, systems, organisms, materials, infrastructure, or data objects
- what the work is applied to, measured on, protecting, improving, or serving

Do not include:
- methods used on the target
- deliverables produced for the target
- broad domain labels unless they identify the served/studied entity
- pure setting/institution context when no served/studied entity is specified

{DECOMPOSITION_BASE_RULES}
""".strip()


DECOMPOSE_SYSTEM_PROMPTS_BY_ASPECT = {
    "domain": DOMAIN_DECOMPOSE_SYSTEM_PROMPT,
    "method": METHOD_DECOMPOSE_SYSTEM_PROMPT,
    "target": TARGET_DECOMPOSE_SYSTEM_PROMPT,
}


DECOMPOSE_USER_PROMPT_TEMPLATE = """
/no_think

Aspect to extract:
{aspect}

Specialization text:
{text}
""".strip()
