from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ce2.aspect_common import ASPECTS, SpecItem, append_jsonl
from ce2.llm_runtime import batched, build_prompt, coerce_score, extract_json_object, generate_responses_batch, normalize_ws, score_to_band
from ce2.prompt.aspect_scoring_prompts import SCORE_SYSTEM_PROMPTS_BY_ASPECT, SCORE_USER_PROMPT_TEMPLATE

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


def _parse_single_score(obj: Optional[Dict[str, Any]]) -> Tuple[float, str, bool]:
    if not isinstance(obj, dict):
        return 0.0, "", False
    if "score" not in obj:
        return 0.0, "", False
    return coerce_score(obj.get("score")), normalize_ws(obj.get("reason")), True


def score_pairs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    pairs: Sequence[Tuple[SpecItem, SpecItem, float, str]],
    decompositions: Dict[str, Dict[str, Any]],
    existing_pair_ids: set[str],
    output_path: Any,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = [p for p in pairs if f"{p[0].item_id}::{p[1].item_id}" not in existing_pair_ids]
    print(f"score_existing={len(existing_pair_ids)} score_pending={len(pending)}")
    written_rows: List[Dict[str, Any]] = []
    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[Tuple[SpecItem, SpecItem, float, str]] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[Tuple[SpecItem, SpecItem, float, str]]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"Score {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar
        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts: List[str] = []
            task_items: List[Tuple[str, str, SpecItem, SpecItem, float, str, Dict[str, Any], Dict[str, Any]]] = []
            for aspect in ASPECTS:
                for grant, fac, lexical_score, pair_source in chunk:
                    g_dec_row = decompositions.get(grant.item_id, {})
                    f_dec_row = decompositions.get(fac.item_id, {})
                    g_dec = g_dec_row.get("decomposition") if isinstance(g_dec_row, dict) else {}
                    f_dec = f_dec_row.get("decomposition") if isinstance(f_dec_row, dict) else {}
                    pair_id = f"{grant.item_id}::{fac.item_id}"
                    user_prompt = SCORE_USER_PROMPT_TEMPLATE.format(
                        aspect=aspect,
                        grant_text=grant.text,
                        grant_aspect_items_json=json.dumps(g_dec.get(aspect, []), ensure_ascii=False),
                        grant_decomposition_json=json.dumps(g_dec, ensure_ascii=False),
                        fac_text=fac.text,
                        fac_aspect_items_json=json.dumps(f_dec.get(aspect, []), ensure_ascii=False),
                        fac_decomposition_json=json.dumps(f_dec, ensure_ascii=False),
                    )
                    prompts.append(
                        build_prompt(
                            tokenizer,
                            model_id=model_id,
                            system_prompt=SCORE_SYSTEM_PROMPTS_BY_ASPECT[aspect],
                            user_prompt=user_prompt,
                        )
                    )
                    task_items.append((pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row))

            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )

            rows: List[Dict[str, Any]] = []
            grouped: Dict[str, Dict[str, Any]] = {}
            for item, response in zip(task_items, responses):
                pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row = item
                parsed = extract_json_object(response)
                score, reason, ok = _parse_single_score(parsed)
                bucket = grouped.setdefault(
                    pair_id,
                    {
                        "grant": grant,
                        "fac": fac,
                        "lexical_score": lexical_score,
                        "pair_source": pair_source,
                        "g_dec_row": g_dec_row,
                        "f_dec_row": f_dec_row,
                        "scores": {},
                        "reasons": {},
                        "raw_responses": {},
                        "ok": {},
                    },
                )
                bucket["scores"][aspect] = float(score)
                bucket["reasons"][aspect] = reason
                bucket["raw_responses"][aspect] = response
                bucket["ok"][aspect] = bool(ok)

            for pair_id, bucket in grouped.items():
                grant = bucket["grant"]
                fac = bucket["fac"]
                lexical_score = bucket["lexical_score"]
                pair_source = bucket["pair_source"]
                g_dec_row = bucket["g_dec_row"]
                f_dec_row = bucket["f_dec_row"]
                score_map = bucket["scores"]
                reason_map = bucket["reasons"]
                ok_map = bucket["ok"]
                parse_ok = all(bool(ok_map.get(aspect)) and aspect in score_map for aspect in ASPECTS)
                if not parse_ok:
                    next_pending.append((grant, fac, lexical_score, pair_source))
                    continue
                aspect_scores = {aspect: float(score_map[aspect]) for aspect in ASPECTS}
                overall_score = float(sum(aspect_scores.values()) / max(1, len(ASPECTS)))
                row = {
                    "pair_id": pair_id,
                    "grant": {
                        "item_id": grant.item_id,
                        "text": grant.text,
                        "meta": grant.meta,
                        "decomposition": g_dec_row.get("decomposition", {}),
                    },
                    "faculty": {
                        "item_id": fac.item_id,
                        "text": fac.text,
                        "meta": fac.meta,
                        "decomposition": f_dec_row.get("decomposition", {}),
                    },
                    "scores": {**aspect_scores, "overall": overall_score},
                    "bands": {
                        **{aspect: score_to_band(score) for aspect, score in aspect_scores.items()},
                        "overall": score_to_band(overall_score),
                    },
                    "reasons": {aspect: reason_map.get(aspect, "") for aspect in ASPECTS},
                    "lexical_prefilter_score": float(lexical_score),
                    "pair_source": pair_source,
                    "parse_ok": True,
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_responses": bucket["raw_responses"],
                }
                existing_pair_ids.add(pair_id)
                rows.append(row)
                written_rows.append(row)

            written_this_attempt += len(rows)
            append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)), refresh=False)
        if bar is not None:
            bar.close()
        pending = next_pending
        if pending:
            print(f"score_retry_pending={len(pending)} attempt={attempt + 1}")
    if pending:
        print(f"score_failed={len(pending)}")
    return written_rows
