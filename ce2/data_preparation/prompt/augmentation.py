from __future__ import annotations

AUGMENT_OUTPUT_SCHEMA = """
Return exactly one JSON object and no markdown.
Do not output chain-of-thought or <think> blocks.

Required schema:
{
  "faculty_text": "<one concise specialization phrase, 8-26 words>",
  "decomposition": {
    "domain": ["..."],
    "method": ["..."],
    "target": ["..."]
  },
  "note": "<short note>"
}
""".strip()

AUGMENT_BASE_RULES = """
Rules:
- Generate realistic faculty specialization style text.
- Do not copy long spans from the grant text.
- Use short keyword phrases in decomposition.
- Keep decomposition aspect boundaries clean.
- Use lower-case unless proper nouns.
- Keep domain items 1-3 words, method 1-4 words, target 1-4 words.
""".strip()

DOMAIN_AUGMENT_SYSTEM_PROMPT = f"""
You are generating synthetic faculty specialization candidates to hit a target DOMAIN band.

Optimize primary control on DOMAIN only:
- high: same/very-close topic-problem area
- mid: related topic with meaningful gap
- low: weakly related or different topic

Secondary guidance:
- method/target can vary naturally, but keep text plausible.

{AUGMENT_BASE_RULES}

{AUGMENT_OUTPUT_SCHEMA}
""".strip()

METHOD_AUGMENT_SYSTEM_PROMPT = f"""
You are generating synthetic faculty specialization candidates to hit a target METHOD band.

Optimize primary control on METHOD only:
- high: same/very-close techniques/workflows
- mid: partial overlap with missing/replaced method parts
- low: weakly related or different techniques

Secondary guidance:
- domain/target can vary naturally, but keep text plausible.

{AUGMENT_BASE_RULES}

{AUGMENT_OUTPUT_SCHEMA}
""".strip()

TARGET_AUGMENT_SYSTEM_PROMPT = f"""
You are generating synthetic faculty specialization candidates to hit a target TARGET band.

Optimize primary control on TARGET only:
- high: same/very-close served population/entity/system/object
- mid: related but not the same target
- low: weakly related or different target

Secondary guidance:
- domain/method can vary naturally, but keep text plausible.

{AUGMENT_BASE_RULES}

{AUGMENT_OUTPUT_SCHEMA}
""".strip()

AUGMENT_SYSTEM_PROMPTS_BY_ASPECT = {
    "domain": DOMAIN_AUGMENT_SYSTEM_PROMPT,
    "method": METHOD_AUGMENT_SYSTEM_PROMPT,
    "target": TARGET_AUGMENT_SYSTEM_PROMPT,
}

AUGMENT_USER_PROMPT_TEMPLATE = """
/no_think

Target aspect band to hit:
- aspect: {aspect}
- target_band: {target_band}

Grant specialization text:
{grant_text}

Grant decomposition:
{grant_decomposition_json}

Grant {aspect} phrases:
{grant_aspect_items_json}

Generate one faculty specialization candidate whose {aspect} match to this grant is likely {target_band}.
""".strip()
