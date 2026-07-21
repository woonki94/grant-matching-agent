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
- Do not mention scores, bands, augmentation, synthetic data, or the grant.
- Use short keyword phrases in decomposition.
- Keep decomposition aspect boundaries clean.
- Make the decomposition faithful to the generated faculty text, not just to the desired band.
- Use lower-case unless proper nouns.
- Keep domain items 1-3 words, method 1-4 words, target 1-4 words.
""".strip()

AUGMENT_BAND_RULES = """
The candidate will be judged later by an independent strict scorer.
Target the requested band for the requested aspect only:
- high means expected score 0.70-1.00: very close or near-exact aspect match.
- mid means expected score 0.30-0.69: meaningful related overlap, but with a clear missing or changed piece.
- low means expected score 0.00-0.29: weak, incidental, or different aspect match.

For mid, avoid both extremes: do not make it a near-exact match, and do not make it unrelated.
For low, avoid exact phrases, close synonyms, and obvious same-thing paraphrases for the target aspect.
For high, include concrete overlap or close synonyms for the target aspect, not only broad umbrella terms.
""".strip()

DOMAIN_AUGMENT_SYSTEM_PROMPT = f"""
You are generating synthetic faculty specialization candidates to hit a target DOMAIN band.

Optimize primary control on DOMAIN only:
- high: same or very-close topic/problem/application area
- mid: adjacent topic/problem/application area with a meaningful gap
- low: weakly related or different topic/problem/application area

Secondary guidance:
- method/target can vary naturally, but keep text plausible.

{AUGMENT_BASE_RULES}

{AUGMENT_BAND_RULES}

{AUGMENT_OUTPUT_SCHEMA}
""".strip()

METHOD_AUGMENT_SYSTEM_PROMPT = f"""
You are generating synthetic faculty specialization candidates to hit a target METHOD band.

Optimize primary control on METHOD only:
- high: same or very-close concrete techniques, workflows, models, instruments, or interventions
- mid: partial method overlap with missing, narrowed, broadened, or replaced method parts
- low: weakly related or different techniques, workflows, models, instruments, or interventions

Secondary guidance:
- domain/target can vary naturally, but keep text plausible.

{AUGMENT_BASE_RULES}

{AUGMENT_BAND_RULES}

{AUGMENT_OUTPUT_SCHEMA}
""".strip()

TARGET_AUGMENT_SYSTEM_PROMPT = f"""
You are generating synthetic faculty specialization candidates to hit a target TARGET band.

Optimize primary control on TARGET only:
- high: same or very-close served population, entity, system, object, data type, or beneficiary
- mid: related target with a clear subtype, supertype, setting, or entity gap
- low: weakly related or different served population, entity, system, object, data type, or beneficiary

Secondary guidance:
- domain/method can vary naturally, but keep text plausible.

{AUGMENT_BASE_RULES}

{AUGMENT_BAND_RULES}

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
Silently check that the generated faculty text and its decomposition would land in the requested band for {aspect}.
""".strip()
