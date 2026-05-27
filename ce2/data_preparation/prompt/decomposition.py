from __future__ import annotations


DECOMPOSITION_BASE_RULES = """
Rules:
- Extract only information actually present or strongly implied by the text.
- Use short keyword phrases only.
- Prefer phrases over full sentences.
- Do not explain your reasoning.
- Do not output chain-of-thought or <think> blocks.
- Do not add "must", "should", or requirement wording unless present in source.
- Return [] when this aspect is not explicitly present or clearly implied.
- Empty is better than a fake or generic label.
- Do not invent concepts not grounded in the source text.
- Lowercase unless proper nouns.
- Each phrase should appear in only one aspect unless absolutely necessary.
- Prefer the most specific role:
  - method beats domain for techniques/actions
  - target beats domain for populations/systems/objects
  - domain is only the topic/problem area left after method/target phrases are removed
- Preserve meaningful multi-word technical phrases.
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
- each domain item should usually be 1-6 words.

Include phrases about:
- research or application topic
- problem area
- scientific/technical field
- high-level service or use area

Do not include:
- concrete methods or procedures
- populations/entities unless they define the topic area itself
- institution/entity/object lists (e.g., universities, institutes, entities, patients, fish) when they are targets
- deliverable/object nouns (e.g., textbooks, resources, platforms, datasets, tools)
- objectives or outcomes unless they define the problem area

{DECOMPOSITION_BASE_RULES}
""".strip()


METHOD_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract METHOD phrases from a short grant or faculty specialization keyword.

Method means concrete techniques, algorithms, procedures, analytical approaches, workflows, models, mechanisms, interventions, or service components.

Length rule:
- each method item should usually be 1-8 words.

Include phrases about:
- concrete methods or techniques
- algorithms, models, instruments, measurements, mechanisms, or procedures
- service components such as case management, legal aid, screening, counseling, training, or assessment
- analytical or implementation workflows
- action/process phrases such as managing, identifying, evaluating, measuring, designing, developing, creating, distributing, implementing, combining, characterizing, optimizing
- techniques such as spectroscopy, characterization, statistics, integration, fabrication, synthesis, testing, simulation, modeling, analysis
- "experience/expertise/skill in X" patterns: extract X as method phrase when X is a concrete process

Do not include:
- broad topic/domain phrases
- populations/entities unless they are part of a method phrase
- settings or application environments unless central to the method phrase
- generic single-word method labels when a more specific phrase is available
- phenomena, materials, populations, or objects by themselves
- a method item when the text only names a topic/problem and gives no procedure

{DECOMPOSITION_BASE_RULES}
""".strip()


TARGET_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract TARGET phrases from a short grant or faculty specialization keyword.

Target means the population, entity, system, material, organism, dataset object, community, or context being served or studied.

Length rule:
- each target item should usually be 1-6 words.
- if target is unclear, infer the acted-on entity/object from the text when possible; return [] if no target is present.

Include phrases about:
- populations, communities, beneficiaries, stakeholders, or study subjects
- entities, systems, organisms, materials, infrastructure, or data objects
- what the work is applied to, measured on, protecting, improving, or serving
- outcomes, conditions, or settings when they are the object being optimized, reduced, measured, or improved
- institution/entity/system types (e.g., hospitals, schools, institutes, agencies, fish populations) when they are the acted-on object

Do not include:
- methods used on the target
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
