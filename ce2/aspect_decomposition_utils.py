from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ce2.aspect_common import ASPECTS, ASPECT_MAX_ITEMS, ASPECT_WORD_LIMITS, SpecItem, append_jsonl
from ce2.llm_runtime import batched, build_prompt, extract_json_object, generate_responses_batch, normalize_ws
from ce2.prompt.decomposition_prompt import DECOMPOSE_SYSTEM_PROMPTS_BY_ASPECT, DECOMPOSE_USER_PROMPT_TEMPLATE

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_ws(x) for x in value if normalize_ws(x)]
    if isinstance(value, str) and normalize_ws(value):
        return [normalize_ws(value)]
    return []


def _parse_decomposition_items(obj: Optional[Dict[str, Any]], aspect: str) -> Tuple[List[str], bool]:
    if not isinstance(obj, dict):
        return [], False
    if "items" in obj:
        return _as_list(obj.get("items")), True
    if aspect in obj:
        return _as_list(obj.get(aspect)), True
    return [], False


_METHOD_CUE_RE = re.compile(
    r"\b("
    r"manag(?:e|es|ed|ing|ment)?|"
    r"identif(?:y|ies|ied|ying|ication)?|"
    r"evaluat(?:e|es|ed|ing|ion)?|"
    r"measur(?:e|es|ed|ing|ement)?|"
    r"assess(?:ment|e|es|ed|ing)?|"
    r"screen(?:ing|ed|s)?|survey(?:ing|ed|s)?|"
    r"model(?:ing|led|s)?|analy(?:sis|ze|zes|zed|zing|tical)?|"
    r"map(?:ping|ped|s)?|monitor(?:ing|ed|s)?|"
    r"implement(?:ation|ing|ed|s)?|"
    r"develop(?:ing|ed|s)?|design(?:ing|ed|s)?|"
    r"creat(?:e|es|ed|ing|ion)?|distribut(?:e|es|ed|ing|ion)?|"
    r"train(?:ing|ed|s)?|optimiz(?:e|es|ed|ing|ation)?|simulate(?:d|s|ing)?|validate(?:d|s|ing)?|"
    r"case management|legal aid|workflow|protocol|algorithm|method\w*|technique\w*"
    r")\b"
)

_TRAILING_DROP_WORDS = {
    "and",
    "or",
    "for",
    "with",
    "to",
    "in",
    "on",
    "via",
    "through",
    "including",
    "involving",
}

_METHOD_SINGLETON_WHITELIST = {
    "screening",
    "surveying",
    "triage",
    "simulation",
    "optimization",
    "modeling",
    "implementation",
    "management",
    "evaluation",
    "assessment",
    "analysis",
    "monitoring",
    "design",
    "development",
    "training",
}

_NEAR_DUP_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "with",
    "to",
    "in",
    "on",
    "and",
    "or",
    "by",
    "via",
    "through",
}

_TARGET_LIKE_RE = re.compile(
    r"\b("
    r"patient\w*|famil\w*|youth|children|child|student\w*|participant\w*|"
    r"institution\w*|university|college|school|institute\w*|entity|entities|"
    r"community|communities|clinic\w*|hospital\w*|agency|agencies|"
    r"service\w*|program\w*|initiative\w*|platform\w*|resource\w*|textbook\w*|"
    r"dataset\w*|tool\w*|fish|species|system\w*|infrastructure"
    r")\b"
)

_DELIVERABLE_LIKE_RE = re.compile(
    r"\b("
    r"textbook\w*|resource\w*|platform\w*|dataset\w*|tool\w*|software|"
    r"report\w*|protocol\w*|model\w*|content|materials?"
    r")\b"
)


def _dedupe_phrases(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        phrase = normalize_ws(v)
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _strip_capability_prefix(phrase: str) -> str:
    p = normalize_ws(phrase)
    p = re.sub(
        r"^(experience|expertise|skill|ability|proficiency|competence)\s+(with|in|for)\s+",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(r"^(experience|expertise|skill|ability)\s+", "", p, flags=re.IGNORECASE)
    return normalize_ws(p)


def _limit_words(phrase: str, *, max_words: int) -> str:
    text = normalize_ws(phrase)
    if not text:
        return ""
    words = text.split()
    while words and words[-1].lower() in _TRAILING_DROP_WORDS:
        words = words[:-1]
    if len(words) <= int(max_words):
        return " ".join(words)
    return " ".join(words[: int(max_words)])


def _normalize_aspect_items(aspect: str, values: Sequence[str]) -> List[str]:
    max_words = int(ASPECT_WORD_LIMITS.get(aspect, 4))
    max_items = int(ASPECT_MAX_ITEMS.get(aspect, 4))
    out: List[str] = []
    for raw in values:
        phrase = normalize_ws(raw)
        if not phrase:
            continue
        phrase = re.sub(r"^[\-\u2022\*\d\.\)\(]+\s*", "", phrase)
        if aspect == "method":
            phrase = _strip_capability_prefix(phrase)
            words = phrase.split()
            if len(words) > max_words and words and words[0].lower().endswith("ing"):
                phrase = " ".join([words[0], *words[-(max_words - 1) :]])
            else:
                phrase = _limit_words(phrase, max_words=max_words)
        else:
            phrase = _limit_words(phrase, max_words=max_words)
        if not phrase:
            continue
        if aspect == "method":
            w = phrase.split()
            if len(w) < 2 and phrase.lower() not in _METHOD_SINGLETON_WHITELIST:
                continue
        out.append(phrase)
    return _dedupe_phrases(out)[:max_items]


def _token_set_loose(phrase: str) -> set[str]:
    toks = []
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", normalize_ws(phrase).lower()):
        if t in _NEAR_DUP_STOPWORDS:
            continue
        toks.append(t)
    return set(toks)


def _is_near_duplicate(a: str, b: str) -> bool:
    aa = a.casefold()
    bb = b.casefold()
    if aa == bb:
        return True
    ta = _token_set_loose(a)
    tb = _token_set_loose(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    uni = len(ta | tb)
    if uni == 0:
        return False
    j = inter / uni
    if j >= 0.80:
        return True
    if ta.issubset(tb) or tb.issubset(ta):
        return True
    return False


def _extract_method_fallbacks(text: str) -> List[str]:
    parts = re.split(r",|;", normalize_ws(text), flags=re.IGNORECASE)
    out: List[str] = []
    for part in parts:
        phrase = _strip_capability_prefix(part)
        if not phrase:
            continue
        if not _METHOD_CUE_RE.search(phrase.lower()):
            continue
        words = phrase.split()
        if len(words) < 1:
            continue
        out.append(_limit_words(phrase, max_words=int(ASPECT_WORD_LIMITS["method"])))
    return _normalize_aspect_items("method", out)


def _extract_domain_fallbacks(text: str) -> List[str]:
    base = _strip_capability_prefix(text)
    if not base:
        return []
    preferred = base
    m = re.search(r"\b(?:for|in|on)\b\s+(.+)$", base, flags=re.IGNORECASE)
    if m:
        preferred = normalize_ws(m.group(1))
    preferred = re.split(r",|;|\bwith\b|\busing\b|\bvia\b|\bthrough\b", preferred, maxsplit=1, flags=re.IGNORECASE)[0]
    preferred = _strip_capability_prefix(preferred)
    preferred = re.sub(
        r"^(manag(?:e|es|ed|ing)|identif(?:y|ies|ied|ying)|evaluat(?:e|es|ed|ing)|"
        r"measur(?:e|es|ed|ing)|implement(?:ation|ing|ed|s)|design(?:ing|ed|s)|"
        r"develop(?:ing|ed|s)|creat(?:e|es|ed|ing)|distribut(?:e|es|ed|ing))\s+",
        "",
        preferred,
        flags=re.IGNORECASE,
    )
    cleaned = _normalize_aspect_items("domain", [preferred])
    if cleaned:
        return cleaned
    return _normalize_aspect_items("domain", [_limit_words(base, max_words=int(ASPECT_WORD_LIMITS["domain"]))])


def _extract_target_fallbacks(text: str) -> List[str]:
    base = _strip_capability_prefix(text)
    if not base:
        return []
    candidates: List[str] = []
    for m in re.finditer(
        r"\b(?:for|involving|among|serving|targeting|toward|towards|within)\b\s+([^,;]+)",
        base,
        flags=re.IGNORECASE,
    ):
        seg = normalize_ws(m.group(1))
        if not seg:
            continue
        parts = re.split(r"\band\b|,|;", seg, flags=re.IGNORECASE)
        candidates.extend(normalize_ws(p) for p in parts if normalize_ws(p))
    targetish: List[str] = []
    for c in candidates:
        if _TARGET_LIKE_RE.search(c.lower()):
            targetish.append(c)
    if targetish:
        return _normalize_aspect_items("target", targetish)
    return _normalize_aspect_items("target", candidates)


def _ensure_dense_decomposition(text: str, decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {aspect: list(decomp.get(aspect, [])) for aspect in ASPECTS}
    if not out.get("domain"):
        out["domain"] = _extract_domain_fallbacks(text)
    if not out.get("method"):
        out["method"] = _extract_method_fallbacks(text)
    if not out.get("target"):
        out["target"] = _extract_target_fallbacks(text)
    if not out["domain"] and out["target"]:
        out["domain"] = _normalize_aspect_items("domain", [out["target"][0]])
    if not out["target"] and out["domain"]:
        out["target"] = _normalize_aspect_items("target", [out["domain"][0]])
    if not out["method"]:
        seed = out["domain"][0] if out["domain"] else (out["target"][0] if out["target"] else "")
        if not seed:
            seed = _limit_words(_strip_capability_prefix(text), max_words=int(ASPECT_WORD_LIMITS["method"]))
        out["method"] = _normalize_aspect_items("method", [seed])
        if not out["method"]:
            out["method"] = _normalize_aspect_items("method", [f"{seed} method"])
    return {aspect: _normalize_aspect_items(aspect, out.get(aspect, [])) for aspect in ASPECTS}


def _clean_decomposition(text: str, decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {aspect: _normalize_aspect_items(aspect, decomp.get(aspect, [])) for aspect in ASPECTS}
    original_domain = list(cleaned.get("domain", []))
    cleaned["method"] = [p for p in cleaned["method"] if _METHOD_CUE_RE.search(p.lower())]
    if not cleaned["method"]:
        cleaned["method"] = _extract_method_fallbacks(text)

    method_keys = {p.casefold() for p in cleaned["method"]}
    cleaned["target"] = [p for p in cleaned["target"] if p.casefold() not in method_keys]
    cleaned["domain"] = [p for p in cleaned["domain"] if p.casefold() not in method_keys]
    cleaned["domain"] = [p for p in cleaned["domain"] if not _METHOD_CUE_RE.search(p.lower())]

    target_vals = list(cleaned["target"])
    cleaned["domain"] = [d for d in cleaned["domain"] if not any(_is_near_duplicate(d, t) for t in target_vals)]
    if target_vals:
        cleaned["domain"] = [d for d in cleaned["domain"] if not _TARGET_LIKE_RE.search(d.lower())]
    cleaned["domain"] = [d for d in cleaned["domain"] if not _DELIVERABLE_LIKE_RE.search(d.lower())]

    if not cleaned["domain"]:
        for d in original_domain:
            dl = d.lower()
            if _METHOD_CUE_RE.search(dl):
                continue
            if _DELIVERABLE_LIKE_RE.search(dl):
                continue
            cleaned["domain"] = [d]
            break

    if len(cleaned["target"]) > 1 and cleaned["domain"]:
        pruned_target = [t for t in cleaned["target"] if not any(_is_near_duplicate(t, d) for d in cleaned["domain"])]
        if pruned_target:
            cleaned["target"] = pruned_target
    cleaned = _ensure_dense_decomposition(text, cleaned)
    for aspect in ASPECTS:
        cleaned[aspect] = _normalize_aspect_items(aspect, cleaned[aspect])
    return cleaned


def decompose_specs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    items: Sequence[SpecItem],
    existing: Dict[str, Dict[str, Any]],
    output_path: Any,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = [item for item in items if item.item_id not in existing]
    print(f"decompose_existing={len(existing)} decompose_pending={len(pending)}")
    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[SpecItem] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[SpecItem]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"Decompose {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar
        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts: List[str] = []
            task_items: List[Tuple[str, str, SpecItem]] = []
            for aspect in ASPECTS:
                for item in chunk:
                    prompts.append(
                        build_prompt(
                            tokenizer,
                            model_id=model_id,
                            system_prompt=DECOMPOSE_SYSTEM_PROMPTS_BY_ASPECT[aspect],
                            user_prompt=DECOMPOSE_USER_PROMPT_TEMPLATE.format(aspect=aspect, text=item.text),
                        )
                    )
                    task_items.append((item.item_id, aspect, item))
            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            rows: List[Dict[str, Any]] = []
            grouped: Dict[str, Dict[str, Any]] = {}
            for task, response in zip(task_items, responses):
                item_id, aspect, item = task
                parsed = extract_json_object(response)
                items_out, ok = _parse_decomposition_items(parsed, aspect)
                bucket = grouped.setdefault(
                    item_id,
                    {"item": item, "decomposition": {a: [] for a in ASPECTS}, "ok": {}, "raw_responses": {}},
                )
                bucket["decomposition"][aspect] = items_out
                bucket["ok"][aspect] = bool(ok)
                bucket["raw_responses"][aspect] = response

            for item_id, bucket in grouped.items():
                item = bucket["item"]
                decomp = _clean_decomposition(item.text, bucket["decomposition"])
                ok_map = bucket["ok"]
                parsed_aspects = [aspect for aspect in ASPECTS if bool(ok_map.get(aspect))]
                nonempty_aspects = [aspect for aspect in ASPECTS if len(decomp.get(aspect, [])) > 0]
                parse_ok = len(parsed_aspects) >= max(1, len(ASPECTS) - 1) and len(nonempty_aspects) >= 1
                row = {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "text": item.text,
                    "meta": item.meta,
                    "decomposition": decomp,
                    "parse_ok": bool(parse_ok),
                    "decomposition_parse": {
                        "parsed_aspects_count": int(len(parsed_aspects)),
                        "nonempty_aspects_count": int(len(nonempty_aspects)),
                        "parsed_aspects": parsed_aspects,
                        "nonempty_aspects": nonempty_aspects,
                    },
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_responses": bucket["raw_responses"],
                }
                if len(parsed_aspects) == 0:
                    next_pending.append(item)
                else:
                    existing[item.item_id] = row
                    rows.append(row)
            written_this_attempt += len(rows)
            append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)), refresh=False)
        if bar is not None:
            bar.close()
        pending = next_pending
        if pending:
            print(f"decompose_retry_pending={len(pending)} attempt={attempt + 1}")

    if pending:
        rows = []
        for item in pending:
            decomp = _ensure_dense_decomposition(item.text, {aspect: [] for aspect in ASPECTS})
            row = {
                "item_id": item.item_id,
                "kind": item.kind,
                "text": item.text,
                "meta": item.meta,
                "decomposition": decomp,
                "parse_ok": False,
                "attempt": int(max(1, int(max_attempts))),
                "model_id": model_id,
                "raw_responses": {},
            }
            existing[item.item_id] = row
            rows.append(row)
        append_jsonl(output_path, rows)
        print(f"decompose_failed_written={len(rows)}")
    return existing
