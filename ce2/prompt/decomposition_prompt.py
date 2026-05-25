from __future__ import annotations


DECOMPOSITION_BASE_RULES = """
Rules:
- Extract only information actually present or strongly implied by the text.
- Use short noun phrases, usually 2-8 words.
- Prefer phrases over full sentences.
- Do not add "must", "should", or requirement wording unless those words are already part of the original phrase.
- If the aspect is absent, return an empty list.
- Do not invent missing aspects.
- Return exactly one JSON object and no markdown.

Required schema:
{
  "items": ["..."]
}
""".strip()


DOMAIN_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract DOMAIN phrases from a short grant or faculty specialization keyword.

Domain means the broad research/application area, problem area, field, or target use case.

Include phrases about:
- research or application topic
- problem area
- scientific/technical field
- high-level service or use area

Do not include:
- concrete methods or procedures
- populations/entities unless they define the topic area
- deliverables/products
- deployment settings unless they define the domain

{DECOMPOSITION_BASE_RULES}
""".strip()


METHOD_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract METHOD phrases from a short grant or faculty specialization keyword.

Method means concrete techniques, algorithms, procedures, analytical approaches, workflows, models, mechanisms, interventions, or service components.

Include phrases about:
- concrete methods or techniques
- algorithms, models, instruments, mechanisms, or procedures
- service components such as case management, legal aid, screening, counseling, training, or assessment
- analytical or implementation workflows

Do not include:
- broad topic/domain phrases
- populations/entities unless they are part of a method phrase
- deliverables/products unless they describe how work is done
- settings or application environments

{DECOMPOSITION_BASE_RULES}
""".strip()


TARGET_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract TARGET phrases from a short grant or faculty specialization keyword.

Target means the population, entity, system, material, organism, dataset object, community, or context being served or studied.

Include phrases about:
- populations, communities, beneficiaries, stakeholders, or study subjects
- entities, systems, organisms, materials, infrastructure, or data objects
- what the work is applied to, measured on, protecting, improving, or serving

Do not include:
- methods used on the target
- deliverables produced for the target
- broad domain labels unless they identify the served/studied entity

{DECOMPOSITION_BASE_RULES}
""".strip()


DELIVERABLE_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract DELIVERABLE phrases from a short grant or faculty specialization keyword.

Deliverable means a concrete output, service, product, tool, model, report, training, software, infrastructure, or program expected or produced.

Include phrases about:
- software tools, datasets, models, reports, protocols, platforms, systems, infrastructure, or programs
- services, interventions, training, assistance, or operational capabilities delivered
- concrete outputs expected by a grant or produced by faculty work

Do not include:
- methods unless the method itself is the delivered service/component
- broad domain phrases
- targets unless they are part of the deliverable phrase
- application settings

{DECOMPOSITION_BASE_RULES}
""".strip()


APPLICATION_CONTEXT_DECOMPOSE_SYSTEM_PROMPT = f"""
You extract APPLICATION CONTEXT phrases from a short grant or faculty specialization keyword.

Application context means the setting, deployment environment, operational context, sector, institution type, geography, or real-world use environment.

Include phrases about:
- operational or deployment settings
- sectors, institution types, geographies, field settings, or use environments
- implementation or practice contexts
- environmental, clinical, educational, industrial, community, or policy settings

Do not include:
- the target entity alone unless it describes a setting
- methods or deliverables
- broad domain phrases unless they clearly name an application setting

{DECOMPOSITION_BASE_RULES}
""".strip()


DECOMPOSE_SYSTEM_PROMPTS_BY_ASPECT = {
    "domain": DOMAIN_DECOMPOSE_SYSTEM_PROMPT,
    "method": METHOD_DECOMPOSE_SYSTEM_PROMPT,
    "target": TARGET_DECOMPOSE_SYSTEM_PROMPT,
    "deliverable": DELIVERABLE_DECOMPOSE_SYSTEM_PROMPT,
    "application_context": APPLICATION_CONTEXT_DECOMPOSE_SYSTEM_PROMPT,
}


DECOMPOSE_USER_PROMPT_TEMPLATE = """
Aspect to extract:
{aspect}

Specialization text:
{text}
""".strip()
