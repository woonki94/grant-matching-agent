"""Efficiency benchmark across models of different architecture and size.

Measures, per model: parameter count, load time, single-pair latency, batched
throughput, and memory. Also reports Spearman vs the teacher score as a quick
quality sanity-check.

Workload: pairs pulled from the ce1 teacher-scored data
(ce/dataset/distill/...listwise.jsonl, one {query, doc, score} per line).

Writes ONE result file per run -> ce6/test/efficiency_result.json

Usage:
  python3 ce6/test/bench_efficiency.py                     # auto device, 200 pairs
  python3 ce6/test/bench_efficiency.py --n 400 --device cpu
  python3 ce6/test/bench_efficiency.py --models modernce-base-150m crossgemma-303m
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder


CE1_DATA = Path(__file__).parent / "dataset" / "eval_pairs_large.jsonl"
OUT = Path(__file__).with_name("efficiency_result.json")

# label -> (architecture, HuggingFace id); spans BERT/RoBERTa/ModernBERT/Gemma-decoder/XLM-R, 14M-568M
MODELS = {
    "tinybert-14m":        ("BERT",        "cross-encoder/stsb-TinyBERT-L4"),
    "ettin-xs-32m":        ("Ettin",       "dleemiller/EttinX-sts-xs"),
    "distilroberta-82m":   ("RoBERTa",     "cross-encoder/stsb-distilroberta-base"),
    "modernce-base-150m":  ("ModernBERT",  "dleemiller/ModernCE-base-sts"),
    "crossgemma-303m":     ("Gemma-dec",   "dleemiller/CrossGemma-sts-300m"),
    "roberta-large-355m":  ("RoBERTa",     "cross-encoder/stsb-roberta-large"),
    "modernce-large-396m": ("ModernBERT",  "dleemiller/ModernCE-large-sts"),
    "bge-reranker-568m":   ("XLM-R",       "BAAI/bge-reranker-v2-m3"),
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


def load_pairs(n):
    rows = []
    for line in open(CE1_DATA, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("query") and d.get("doc") and isinstance(d.get("score"), (int, float)):
            rows.append((d["query"], d["doc"], float(d["score"])))
        if len(rows) >= n:
            break
    return rows


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def sim(model, pairs):
    raw = np.asarray(model.predict([(q, d) for q, d, _ in pairs]), dtype=float).ravel()
    return 1 / (1 + np.exp(-raw)) if ((raw < 0).any() or (raw > 1).any()) else raw


def mps_mem_mb():
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 1e6
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of workload pairs (default 200)")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    device = pick_device(args.device)
    pairs = load_pairs(args.n)
    teacher = [s for _, _, s in pairs]
    print(f"workload = {len(pairs)} pairs from ce1  |  device = {device}\n")

    results = []
    for lab in args.models:
        arch, hf = MODELS[lab]
        try:
            m0 = mps_mem_mb()
            t0 = time.perf_counter()
            model = CrossEncoder(hf, device=device, trust_remote_code=True)
            load_s = time.perf_counter() - t0
            params = sum(p.numel() for p in model.model.parameters())
            sim(model, pairs[:4])                                   # warmup
            singles = []
            for p in pairs[:min(100, len(pairs))]:                 # single-pair latency
                t = time.perf_counter()
                sim(model, [p])
                singles.append((time.perf_counter() - t) * 1000.0)
            singles.sort()
            t = time.perf_counter()
            s = sim(model, pairs)                                   # batched throughput + scores
            batch_s = time.perf_counter() - t
            m1 = mps_mem_mb()
            mem = round(m1 - m0, 0) if (m0 is not None and m1 is not None) else round(params * 4 / 1e6, 0)
            row = {
                "model": lab, "arch": arch, "params_m": round(params / 1e6, 1),
                "load_s": round(load_s, 2), "latency_ms": round(singles[len(singles) // 2], 1),
                "throughput_ps": round(len(pairs) / batch_s, 1), "mem_mb": mem,
                "spearman_teacher": round(spearman(list(s), teacher), 3),
            }
            results.append(row)
            print(f"[ok] {lab:20s} {arch:11s} params={row['params_m']}M  load={row['load_s']}s  "
                  f"lat={row['latency_ms']}ms  thru={row['throughput_ps']}/s  mem={row['mem_mb']}MB  "
                  f"spear={row['spearman_teacher']}")
            del model
            gc.collect()
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {lab}: {type(exc).__name__}: {exc}")

    print("\n| model | arch | params(M) | load(s) | lat(ms) | thru(p/s) | mem(MB) | spear_teacher |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['arch']} | {r['params_m']} | {r['load_s']} | {r['latency_ms']} | "
              f"{r['throughput_ps']} | {r['mem_mb']} | {r['spearman_teacher']} |")

    OUT.write_text(json.dumps({"device": device, "n": len(pairs), "results": results}, indent=2))
    print(f"\nsaved -> {OUT}")
    print("(mem = MPS allocated delta on mps, else weight-size estimate; latency = median single pair)")


if __name__ == "__main__":
    main()
