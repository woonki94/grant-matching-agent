"""Capability probe: what the STS and NLI model families can and cannot handle.

A pair is "handled" by a family if that family's mean score falls within the
pair's expected_score_range. Reports per-pair scores (with a within-range check)
and a handled-rate per subtype for each family, plus a short verdict.

Dataset (you control it): ce6/test/eval_pairs.jsonl
  fields: id, grant_keyword, faculty_keyword, kind, expected_score_range [lo, hi]
  (subtype is read from the id, e.g. "T9-negation" -> "negation")

Writes ONE result file per run -> ce6/test/capability_result.json

Usage:
  python3 ce6/test/bench_capability.py
  python3 ce6/test/bench_capability.py --device cpu
  python3 ce6/test/bench_capability.py path/to/other_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)   # flush prints immediately, even when piped

import numpy as np

print("bench_capability: importing sentence-transformers… (first import can take ~10s)")
from sentence_transformers import CrossEncoder
print("bench_capability: libraries ready\n")

DATA = Path(__file__).parent / "dataset" / "eval_pairs.jsonl"
OUT = Path(__file__).with_name("capability_result.json")

STS = {  # symmetric-similarity family
    "mce-base":   "dleemiller/ModernCE-base-sts",
    "mce-large":  "dleemiller/ModernCE-large-sts",
    "roberta-lg": "cross-encoder/stsb-roberta-large",
    "crossgemma": "dleemiller/CrossGemma-sts-300m",
}
NLI = {  # entailment family (scored as coverage proxy = 1 - P(entail))
    "nli-base":  "dleemiller/ModernCE-base-nli",
    "nli-large": "dleemiller/ModernCE-large-nli",
}


def pick_device(arg):
    if arg:
        return arg
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def sts_score(hf, pairs, device):
    m = CrossEncoder(hf, device=device, trust_remote_code=True)
    inputs = [(p["grant_keyword"], p["faculty_keyword"]) for p in pairs]
    raw = np.asarray(m.predict(inputs, show_progress_bar=True), dtype=float).ravel()
    return 1 / (1 + np.exp(-raw)) if ((raw < 0).any() or (raw > 1).any()) else raw


def nli_score(hf, pairs, device):
    m = CrossEncoder(hf, device=device, trust_remote_code=True)
    inputs = [(p["faculty_keyword"], p["grant_keyword"]) for p in pairs]
    raw = np.asarray(m.predict(inputs, show_progress_bar=True), dtype=float)
    raw = raw - raw.max(1, keepdims=True)
    probs = np.exp(raw)
    probs = probs / probs.sum(1, keepdims=True)
    id2 = getattr(getattr(getattr(m, "model", None), "config", None), "id2label", {}) or {}
    idx = next((int(i) for i, lab in id2.items() if "entail" in str(lab).lower()), 0)
    return 1 - probs[:, idx]


def subtype(pid):
    m = re.match(r"^[A-Za-z]*\d+-(.+)$", pid)
    return m.group(1) if m else pid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default=str(DATA))
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    args = ap.parse_args()

    device = pick_device(args.device)
    pairs = [json.loads(line) for line in open(args.dataset, encoding="utf-8") if line.strip()]
    lo = np.array([p["expected_score_range"][0] for p in pairs])
    hi = np.array([p["expected_score_range"][1] for p in pairs])
    print(f"dataset = {args.dataset}  |  pairs = {len(pairs)}  |  device = {device}")
    print("handled = family mean score within expected range;  NLI = coverage proxy (1 - P(entail))\n")

    scores = {}
    for lab, hf in {**STS, **NLI}.items():
        fam = nli_score if lab in NLI else sts_score
        print(f"Inferring: {lab}  ({hf})")
        scores[lab] = fam(hf, pairs, device)

    sts_mean = np.mean([scores[l] for l in STS], axis=0)
    nli_mean = np.mean([scores[l] for l in NLI], axis=0)
    sts_ok = (sts_mean >= lo) & (sts_mean <= hi)
    nli_ok = (nli_mean >= lo) & (nli_mean <= hi)

    # (1) per-pair
    print("### Per-pair   (family mean score; ✓ = within expected range)\n")
    print("| id | expected | STS | NLI |")
    print("|---|---|---|---|")
    for i, p in enumerate(pairs):
        print(f"| {p['id']} | {lo[i]:.2f}-{hi[i]:.2f} | {sts_mean[i]:.2f} {'✓' if sts_ok[i] else '✗'} | "
              f"{nli_mean[i]:.2f} {'✓' if nli_ok[i] else '✗'} |")

    # (2) handled-rate by subtype
    subs = {}
    for i, p in enumerate(pairs):
        subs.setdefault(subtype(p["id"]), []).append(i)
    print("\n### Handled-rate by subtype\n")
    print("| subtype | n | STS | NLI |")
    print("|---|---|---|---|")
    rows = []
    for sub in sorted(subs):
        idx = subs[sub]
        sr = round(float(np.mean([sts_ok[i] for i in idx])), 2)
        nr = round(float(np.mean([nli_ok[i] for i in idx])), 2)
        rows.append({"subtype": sub, "n": len(idx), "sts": sr, "nli": nr})
        print(f"| {sub} | {len(idx)} | {sr:.2f} | {nr:.2f} |")

    # (3) verdict
    def split(fam):
        return ([r["subtype"] for r in rows if r[fam] >= 0.8],
                [r["subtype"] for r in rows if r[fam] <= 0.2])
    sg, sb = split("sts")
    ng, nb = split("nli")
    print("\n### Verdict   (handles = ≥80% in range, fails = ≤20% in range)")
    print(f"STS handles: {', '.join(sg) or 'none'}")
    print(f"STS fails:   {', '.join(sb) or 'none'}")
    print(f"NLI handles: {', '.join(ng) or 'none'}")
    print(f"NLI fails:   {', '.join(nb) or 'none'}")

    OUT.write_text(json.dumps({
        "device": device, "n": len(pairs),
        "overall": {"sts_handled": round(float(sts_ok.mean()), 3), "nli_handled": round(float(nli_ok.mean()), 3)},
        "by_subtype": rows,
    }, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
