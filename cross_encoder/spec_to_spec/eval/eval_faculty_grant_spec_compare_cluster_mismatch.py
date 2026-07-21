from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here] + list(here.parents):
        if (candidate / "cross_encoder").exists() and (candidate / "db").exists():
            return candidate
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cross_encoder.spec_to_spec.eval import eval_faculty_grant_spec_compare as base


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Distill-mode spec_to_spec evaluation that keeps only rows where predicted "
            "cluster differs from ground-truth cluster."
        )
    )

    p.add_argument("--finetuned-model", type=str, default=base.FINETUNED_MODEL_DEFAULT)
    p.add_argument("--base-model", type=str, default=base.PURE_BGE_MODEL_ID)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)

    p.add_argument("--distill-test-input", type=str, default=base.DISTILL_TEST_INPUT_DEFAULT)
    p.add_argument("--distill-input", type=str, default=base.DISTILL_INPUT_DEFAULT)
    p.add_argument(
        "--distill-ground-truth",
        type=str,
        default="raw",
        choices=["normalized", "raw"],
        help="Teacher score field used for GT clustering.",
    )
    p.add_argument("--distill-high-threshold", type=float, default=0.67)
    p.add_argument("--distill-mid-threshold", type=float, default=0.34)
    p.add_argument("--distill-sample-high", type=int, default=0)
    p.add_argument("--distill-sample-mid", type=int, default=0)
    p.add_argument("--distill-sample-low", type=int, default=0)
    p.add_argument("--distill-seed", type=int, default=42)
    p.add_argument("--distill-max-pairs-scan", type=int, default=0)
    p.add_argument("--distill-print-pair-tables", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--distill-save-pair-tables", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--distill-print-rows-per-band", type=int, default=30)
    p.add_argument("--distill-save-rows-per-band", type=int, default=0)

    p.add_argument("--distill-order-top-k", type=int, default=5)
    p.add_argument("--distill-pair-eps", type=float, default=0.01)
    p.add_argument("--distill-hard-gap-max", type=float, default=0.15)
    p.add_argument("--distill-medium-gap-max", type=float, default=0.40)

    p.add_argument(
        "--mismatch-target",
        type=str,
        default="finetuned",
        choices=["finetuned", "base", "either", "both"],
        help=(
            "Which model mismatch condition to keep: "
            "finetuned!=gt, base!=gt, either, or both."
        ),
    )
    p.add_argument("--pred-high-threshold", type=float, default=-1.0)
    p.add_argument("--pred-mid-threshold", type=float, default=-1.0)

    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", type=str, default=base.OUTPUT_DIR_DEFAULT)
    return p


def _keep_row(row: Dict[str, Any], *, target: str) -> bool:
    ft_mis = bool(row.get("finetuned_cluster_mismatch"))
    b_mis = bool(row.get("base_cluster_mismatch"))
    t = str(target or "finetuned").strip().lower()
    if t == "base":
        return b_mis
    if t == "either":
        return ft_mis or b_mis
    if t == "both":
        return ft_mis and b_mis
    return ft_mis


def _format_mismatch_pairs_table(
    *,
    rows: List[Dict[str, Any]],
    title: str,
    max_rows: int,
    text_width: int,
    truncate_text: bool,
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(f"=== {title} ===")
    lines.append(
        f"{'PAIR_KEY':<28} {'GT':>7} {'GT_CLS':>7} {'FINETUNED':>10} {'FT_CLS':>7} {'BASE':>10} {'BASE_CLS':>8} {'FT_MIS':>7} {'B_MIS':>7} {'QUERY | DOC':<{text_width}}"
    )
    lines.append("-" * (104 + text_width))
    limited = list(rows[: max(0, int(max_rows))]) if max_rows > 0 else list(rows)
    for row in limited:
        pair_key = (
            f"{base._clean_text(row.get('grant_id'))}::spec#"
            f"{base._safe_int(row.get('spec_idx'), default=0, minimum=0, maximum=1_000_000_000)}"
        )
        merged_text = f"Q: {base._clean_text(row.get('query_text'))} || D: {base._clean_text(row.get('doc_text'))}"
        rendered_text = base._shorten(merged_text, text_width) if truncate_text else merged_text
        lines.append(
            f"{pair_key:<28} "
            f"{float(row.get('teacher_score_used') or 0.0):>7.4f} "
            f"{base._clean_text(row.get('gt_cluster')):>7} "
            f"{float(row.get('finetuned_score') or 0.0):>10.4f} "
            f"{base._clean_text(row.get('finetuned_cluster')):>7} "
            f"{float(row.get('base_score') or 0.0):>10.4f} "
            f"{base._clean_text(row.get('base_cluster')):>8} "
            f"{str(bool(row.get('finetuned_cluster_mismatch'))):>7} "
            f"{str(bool(row.get('base_cluster_mismatch'))):>7} "
            f"{rendered_text:<{text_width}}"
        )
    return "\n".join(lines)


def _run() -> int:
    args = _build_parser().parse_args()

    batch_size = base._safe_int(args.batch_size, default=32, minimum=1, maximum=4096)
    max_length = base._safe_int(args.max_length, default=512, minimum=64, maximum=4096)
    sample_high = base._safe_int(args.distill_sample_high, default=0, minimum=0, maximum=1_000_000)
    sample_mid = base._safe_int(args.distill_sample_mid, default=0, minimum=0, maximum=1_000_000)
    sample_low = base._safe_int(args.distill_sample_low, default=0, minimum=0, maximum=1_000_000)
    high_threshold = base._safe_float(args.distill_high_threshold, default=0.67, minimum=0.0, maximum=1.0)
    mid_threshold = base._safe_float(args.distill_mid_threshold, default=0.34, minimum=0.0, maximum=1.0)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold
    max_pairs_scan = base._safe_int(args.distill_max_pairs_scan, default=0, minimum=0, maximum=50_000_000)
    seed = base._safe_int(args.distill_seed, default=42, minimum=0, maximum=2_147_483_647)

    pred_high_default = float(high_threshold)
    pred_mid_default = float(mid_threshold)
    pred_high = float(args.pred_high_threshold if float(args.pred_high_threshold) >= 0.0 else pred_high_default)
    pred_mid = float(args.pred_mid_threshold if float(args.pred_mid_threshold) >= 0.0 else pred_mid_default)
    pred_high = max(0.0, min(1.0, pred_high))
    pred_mid = max(0.0, min(1.0, pred_mid))
    if pred_mid > pred_high:
        pred_mid = pred_high

    order_top_k = base._safe_int(args.distill_order_top_k, default=5, minimum=1, maximum=100)
    pair_eps = base._safe_float(args.distill_pair_eps, default=0.01, minimum=0.0, maximum=1.0)
    hard_gap_max = base._safe_float(args.distill_hard_gap_max, default=0.15, minimum=0.0, maximum=1.0)
    medium_gap_max = base._safe_float(args.distill_medium_gap_max, default=0.40, minimum=0.0, maximum=1.0)
    if medium_gap_max < hard_gap_max:
        medium_gap_max = hard_gap_max

    ground_truth_mode = base._clean_text(args.distill_ground_truth).lower() or "normalized"
    if ground_truth_mode not in {"normalized", "raw"}:
        ground_truth_mode = "normalized"

    mismatch_target = base._clean_text(args.mismatch_target).lower() or "finetuned"

    finetuned_ref = base._resolve_model_ref(base._clean_text(args.finetuned_model) or base.FINETUNED_MODEL_DEFAULT)
    base_ref = base._resolve_model_ref(base._clean_text(args.base_model) or base.PURE_BGE_MODEL_ID)

    distill_input = base._resolve_path(args.distill_test_input)
    if not distill_input.exists():
        distill_input = base._resolve_path(args.distill_input)
    if not distill_input.exists():
        fallback = base._resolve_path(base.DISTILL_INPUT_FALLBACK)
        if fallback.exists():
            distill_input = fallback
        else:
            raise RuntimeError(
                f"Distill input file not found: {distill_input}. "
                f"Also fallback not found: {fallback}"
            )

    sampled_rows, sample_meta = base._reservoir_sample_distill_pairs(
        distill_input=distill_input,
        sample_high=sample_high,
        sample_mid=sample_mid,
        sample_low=sample_low,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        ground_truth_mode=ground_truth_mode,
        seed=seed,
        max_pairs_scan=max_pairs_scan,
    )
    if not sampled_rows:
        raise RuntimeError("No pairs were sampled from distill file. Check thresholds and input file.")

    device = base._pick_device()
    tok_finetuned = AutoTokenizer.from_pretrained(finetuned_ref)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(finetuned_ref, num_labels=1).to(device).eval()
    tok_base = AutoTokenizer.from_pretrained(base_ref)
    model_base = AutoModelForSequenceClassification.from_pretrained(base_ref, num_labels=1).to(device).eval()

    finetuned_scores = base._score_query_doc_rows(
        model=model_finetuned,
        tokenizer=tok_finetuned,
        rows=sampled_rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    base_scores = base._score_query_doc_rows(
        model=model_base,
        tokenizer=tok_base,
        rows=sampled_rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    all_rows: List[Dict[str, Any]] = []
    for i, src in enumerate(sampled_rows):
        gt = float(src.get("teacher_score_used") or 0.0)
        ft = float(finetuned_scores[i])
        b = float(base_scores[i])

        gt_cluster = base._score_band_from_teacher(
            gt,
            high_threshold=float(high_threshold),
            mid_threshold=float(mid_threshold),
        )
        ft_cluster = base._score_band_from_teacher(
            ft,
            high_threshold=float(pred_high),
            mid_threshold=float(pred_mid),
        )
        b_cluster = base._score_band_from_teacher(
            b,
            high_threshold=float(pred_high),
            mid_threshold=float(pred_mid),
        )

        row = dict(src)
        row["score_band"] = gt_cluster
        row["gt_cluster"] = gt_cluster
        row["finetuned_score"] = ft
        row["base_score"] = b
        row["finetuned_cluster"] = ft_cluster
        row["base_cluster"] = b_cluster
        row["finetuned_cluster_mismatch"] = bool(ft_cluster != gt_cluster)
        row["base_cluster_mismatch"] = bool(b_cluster != gt_cluster)
        row["finetuned_margin"] = float(ft - gt)
        row["base_margin"] = float(b - gt)
        row["finetuned_abs_margin"] = abs(float(ft - gt))
        row["base_abs_margin"] = abs(float(b - gt))
        row["abs_margin_gain"] = float(row["base_abs_margin"] - row["finetuned_abs_margin"])
        all_rows.append(row)

    filtered_rows = [r for r in all_rows if _keep_row(r, target=mismatch_target)]

    stats = base._compute_margin_stats(filtered_rows)
    order_metrics_finetuned = base._compute_order_metrics_for_model(
        rows=filtered_rows,
        model_score_key="finetuned_score",
        top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        mrr_rel_threshold=high_threshold,
        recall_rel_threshold=mid_threshold,
    )
    order_metrics_base = base._compute_order_metrics_for_model(
        rows=filtered_rows,
        model_score_key="base_score",
        top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        mrr_rel_threshold=high_threshold,
        recall_rel_threshold=mid_threshold,
    )
    raw_sanity_finetuned = base._compute_raw_score_sanity(
        rows=filtered_rows,
        model_score_key="finetuned_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    raw_sanity_base = base._compute_raw_score_sanity(
        rows=filtered_rows,
        model_score_key="base_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )

    by_band: Dict[str, List[Dict[str, Any]]] = {"high": [], "mid": [], "low": []}
    for row in filtered_rows:
        band = base._clean_text(row.get("score_band")).lower()
        if band in by_band:
            by_band[band].append(row)
    for band in by_band:
        by_band[band] = sorted(by_band[band], key=lambda x: float(x.get("teacher_score_used") or 0.0), reverse=True)

    mismatch_counts = {
        "finetuned": int(sum(1 for r in all_rows if bool(r.get("finetuned_cluster_mismatch")))),
        "base": int(sum(1 for r in all_rows if bool(r.get("base_cluster_mismatch")))),
        "either": int(
            sum(
                1
                for r in all_rows
                if bool(r.get("finetuned_cluster_mismatch")) or bool(r.get("base_cluster_mismatch"))
            )
        ),
        "both": int(
            sum(
                1
                for r in all_rows
                if bool(r.get("finetuned_cluster_mismatch")) and bool(r.get("base_cluster_mismatch"))
            )
        ),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base._resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"distill_cluster_mismatch_{mismatch_target}_{ts}.json"
    output_txt = output_dir / f"distill_cluster_mismatch_{mismatch_target}_{ts}.txt"

    meta_header = (
        f"mode=distill_cluster_mismatch\n"
        f"distill_input={distill_input}\n"
        f"ground_truth_mode={ground_truth_mode}\n"
        f"mismatch_target={mismatch_target}\n"
        f"high_threshold={high_threshold} mid_threshold={mid_threshold}\n"
        f"pred_high_threshold={pred_high} pred_mid_threshold={pred_mid}\n"
        f"selected_before_filter={len(all_rows)} selected_after_filter={len(filtered_rows)}\n"
        f"mismatch_counts={json.dumps(mismatch_counts, ensure_ascii=False)}\n"
        f"selected_high={len(by_band['high'])} selected_mid={len(by_band['mid'])} selected_low={len(by_band['low'])}\n"
        f"available_high={int(sample_meta.get('available_by_band', {}).get('high', 0))} "
        f"available_mid={int(sample_meta.get('available_by_band', {}).get('mid', 0))} "
        f"available_low={int(sample_meta.get('available_by_band', {}).get('low', 0))}\n"
        f"order_top_k={order_top_k} pair_eps={pair_eps} hard_gap_max={hard_gap_max} medium_gap_max={medium_gap_max}\n"
        f"finetuned_model={finetuned_ref}\n"
        f"base_model={base_ref}\n"
        f"device={device}\n"
    )

    blocks_print: List[str] = [
        meta_header,
        base._format_order_summary_table(finetuned=order_metrics_finetuned, base=order_metrics_base),
        base._format_margin_summary_table(stats),
        base._format_raw_sanity_table(finetuned=raw_sanity_finetuned, base=raw_sanity_base),
    ]
    blocks_save: List[str] = [
        meta_header,
        base._format_order_summary_table(finetuned=order_metrics_finetuned, base=order_metrics_base),
        base._format_margin_summary_table(stats),
        base._format_raw_sanity_table(finetuned=raw_sanity_finetuned, base=raw_sanity_base),
    ]

    print_pair_tables = bool(args.distill_print_pair_tables)
    save_pair_tables = bool(args.distill_save_pair_tables)
    print_per_band = base._safe_int(args.distill_print_rows_per_band, default=30, minimum=0, maximum=1_000_000)
    save_rows_per_band = base._safe_int(args.distill_save_rows_per_band, default=0, minimum=0, maximum=5_000_000)

    if print_pair_tables:
        blocks_print.append(
            _format_mismatch_pairs_table(
                rows=by_band["high"],
                title="High Score Pairs (Cluster Mismatch Only)",
                max_rows=print_per_band,
                text_width=140,
                truncate_text=True,
            )
        )
        blocks_print.append(
            _format_mismatch_pairs_table(
                rows=by_band["mid"],
                title="Mid Score Pairs (Cluster Mismatch Only)",
                max_rows=print_per_band,
                text_width=140,
                truncate_text=True,
            )
        )
        blocks_print.append(
            _format_mismatch_pairs_table(
                rows=by_band["low"],
                title="Low Score Pairs (Cluster Mismatch Only)",
                max_rows=print_per_band,
                text_width=140,
                truncate_text=True,
            )
        )

    if save_pair_tables:
        blocks_save.append(
            _format_mismatch_pairs_table(
                rows=by_band["high"],
                title="High Score Pairs (Cluster Mismatch Only)",
                max_rows=save_rows_per_band,
                text_width=140,
                truncate_text=False,
            )
        )
        blocks_save.append(
            _format_mismatch_pairs_table(
                rows=by_band["mid"],
                title="Mid Score Pairs (Cluster Mismatch Only)",
                max_rows=save_rows_per_band,
                text_width=140,
                truncate_text=False,
            )
        )
        blocks_save.append(
            _format_mismatch_pairs_table(
                rows=by_band["low"],
                title="Low Score Pairs (Cluster Mismatch Only)",
                max_rows=save_rows_per_band,
                text_width=140,
                truncate_text=False,
            )
        )

    report_text_print = "\n".join(blocks_print).strip() + "\n"
    report_text_save = "\n".join(blocks_save).strip() + "\n"

    if bool(args.print):
        print(report_text_print)

    if bool(args.save):
        payload = {
            "meta": {
                "created_at_local": datetime.now().isoformat(),
                "mode": "distill_cluster_mismatch",
                "distill_input": str(distill_input),
                "ground_truth_mode": ground_truth_mode,
                "sample_meta": sample_meta,
                "sample_high": int(sample_high),
                "sample_mid": int(sample_mid),
                "sample_low": int(sample_low),
                "high_threshold": float(high_threshold),
                "mid_threshold": float(mid_threshold),
                "pred_high_threshold": float(pred_high),
                "pred_mid_threshold": float(pred_mid),
                "mismatch_target": mismatch_target,
                "mismatch_counts": mismatch_counts,
                "selected_before_filter": int(len(all_rows)),
                "selected_after_filter": int(len(filtered_rows)),
                "distill_print_pair_tables": bool(print_pair_tables),
                "distill_save_pair_tables": bool(save_pair_tables),
                "distill_print_rows_per_band": int(print_per_band),
                "distill_save_rows_per_band": int(save_rows_per_band),
                "order_top_k": int(order_top_k),
                "pair_eps": float(pair_eps),
                "hard_gap_max": float(hard_gap_max),
                "medium_gap_max": float(medium_gap_max),
                "finetuned_model": finetuned_ref,
                "base_model": base_ref,
                "device": str(device),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "output_txt": str(output_txt),
            },
            "order_metrics": {
                "finetuned": order_metrics_finetuned,
                "base": order_metrics_base,
            },
            "margin_stats": stats,
            "raw_score_sanity": {
                "finetuned": raw_sanity_finetuned,
                "base": raw_sanity_base,
            },
            "rows": filtered_rows,
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(report_text_save, encoding="utf-8")
        print(f"saved_json={output_json}")
        print(f"saved_txt={output_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
