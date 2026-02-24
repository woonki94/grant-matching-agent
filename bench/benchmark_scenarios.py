#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "benchmark_scenarios.config.json"
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for token in _parse_csv_list(raw):
        try:
            out.append(int(token))
        except Exception as exc:
            raise ValueError(f"Invalid integer token in list: '{token}'") from exc
    if not out:
        raise ValueError("At least one integer value is required.")
    return sorted(set(out))


def _parse_optional_broad_category(raw: Optional[str]) -> Optional[str | List[str]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    parts = _parse_csv_list(txt)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ranked = sorted(float(v) for v in values)
    idx = max(0, min(len(ranked) - 1, math.ceil((p / 100.0) * len(ranked)) - 1))
    return float(ranked[idx])


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


@dataclass
class ScenarioSpec:
    name: str
    scenario_type: str
    payload: Dict[str, Any]
    team_size_requested: Optional[int] = None


def _default_config() -> Dict[str, Any]:
    return {
        "single_email": "your_single_faculty@osu.edu",
        "group_emails": [
            "faculty1@osu.edu",
            "faculty2@osu.edu",
        ],
        "single_query_text": "machine learning",
        "single_broad_category": "applied_research",
        "single_top_k": 10,
        "group_team_sizes": "2,3,4,5",
        "group_top_k": 5,
        "group_query_text": "",
        "group_broad_category": "",
        "warmup": 1,
        "repeats": 3,
        "output_dir": str(PROJECT_ROOT / "outputs" / "benchmarks"),
        "skip_email_precheck": False,
        "dry_run_scenarios": False,
    }


def _write_config_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_default_config(), f, ensure_ascii=False, indent=2)


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _write_config_template(path)
        raise FileNotFoundError(
            f"Config file not found. A template was created at: {path}\n"
            "Edit it once, then run this script again with no parameters."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object.")
    return data


def _pick(cli_value: Any, cfg_value: Any, default_value: Any = None) -> Any:
    if cli_value is not None:
        if isinstance(cli_value, str):
            if cli_value.strip():
                return cli_value
        else:
            return cli_value
    if cfg_value is not None:
        if isinstance(cfg_value, str):
            if cfg_value.strip():
                return cfg_value
        else:
            return cfg_value
    return default_value


def _is_placeholder_email(raw: str) -> bool:
    x = str(raw or "").strip().lower()
    if not x:
        return True
    if x in {"your_single_faculty@osu.edu", "faculty1@osu.edu", "faculty2@osu.edu"}:
        return True
    if x.startswith("your_") or x.startswith("faculty"):
        return True
    return False


def _try_auto_fill_emails(single_email: str, group_emails_csv: str) -> tuple[str, str]:
    single = str(single_email or "").strip().lower()
    group_items = [e.strip().lower() for e in _parse_csv_list(str(group_emails_csv or ""))]
    needs_single = _is_placeholder_email(single)
    needs_group = len(group_items) < 2 or all(_is_placeholder_email(x) for x in group_items)

    if not needs_single and not needs_group:
        return single, ",".join(group_items)

    try:
        from db.db_conn import SessionLocal
        from db.models.faculty import Faculty

        with SessionLocal() as sess:
            rows = (
                sess.query(Faculty.email)
                .filter(Faculty.email.isnot(None))
                .order_by(Faculty.faculty_id.asc())
                .limit(5)
                .all()
            )
        candidates = _ordered_unique(
            [str(r[0]).strip().lower() for r in rows if r and str(r[0]).strip()]
        )
    except Exception:
        candidates = []

    if needs_single and candidates:
        single = candidates[0]
    if needs_group and len(candidates) >= 2:
        group_items = candidates[:2]

    return single, ",".join(group_items)


def _build_scenarios(settings: Dict[str, Any]) -> List[ScenarioSpec]:
    single_email = str(settings["single_email"]).strip().lower()
    group_emails = _ordered_unique([e.strip().lower() for e in _parse_csv_list(str(settings["group_emails"]))])
    if len(group_emails) < 2:
        raise ValueError("group_emails must contain at least 2 emails.")

    single_broad = _parse_optional_broad_category(str(settings["single_broad_category"]))
    group_broad = _parse_optional_broad_category(str(settings["group_broad_category"]))

    scenarios: List[ScenarioSpec] = [
        ScenarioSpec(
            name="single_fac",
            scenario_type="single",
            payload={
                "user_input": f"Find matching grants for {single_email}",
                "email": single_email,
                "requested_top_k_grants": int(settings["single_top_k"]),
            },
        ),
        ScenarioSpec(
            name="single_fac_with_query",
            scenario_type="single",
            payload={
                "user_input": f"Find grants related to {settings['single_query_text']} for {single_email}",
                "email": single_email,
                "topic_query": str(settings["single_query_text"]),
                "requested_top_k_grants": int(settings["single_top_k"]),
            },
        ),
        ScenarioSpec(
            name="single_fac_with_query_and_broad_category",
            scenario_type="single",
            payload={
                "user_input": (
                    f"Find {settings['single_broad_category']} grants related to "
                    f"{settings['single_query_text']} for {single_email}"
                ),
                "email": single_email,
                "topic_query": str(settings["single_query_text"]),
                "desired_broad_category": single_broad,
                "requested_top_k_grants": int(settings["single_top_k"]),
            },
        ),
    ]

    for team_size in _parse_int_csv(str(settings["group_team_sizes"])):
        payload: Dict[str, Any] = {
            "user_input": (
                f"Find group matches for {', '.join(group_emails)} "
                f"with desired team size {team_size}"
            ),
            "emails": list(group_emails),
            "requested_team_size": int(team_size),
            "requested_top_k_grants": int(settings["group_top_k"]),
        }
        if settings["group_query_text"]:
            payload["topic_query"] = str(settings["group_query_text"])
        if group_broad is not None:
            payload["desired_broad_category"] = group_broad

        scenarios.append(
            ScenarioSpec(
                name=f"group_team_size_{team_size}",
                scenario_type="group",
                payload=payload,
                team_size_requested=int(team_size),
            )
        )

    return scenarios


def _run_one_request(
    orchestrator: Any,
    grant_request: Any,
    *,
    thread_id: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    prev = t0
    node_rows: List[Dict[str, Any]] = []
    final_output: Dict[str, Any] = {}
    node_index = 0

    for evt in orchestrator.stream(grant_request, thread_id=thread_id):
        now = time.perf_counter()
        et = evt.get("type")
        if et == "step":
            node_rows.append(
                {
                    "node_index": node_index,
                    "node": str(evt.get("node") or ""),
                    "duration_seconds": now - prev,
                }
            )
            prev = now
            node_index += 1
        elif et == "final":
            final_output = evt.get("output") or {}
            node_rows.append(
                {
                    "node_index": node_index,
                    "node": "__finalize__",
                    "duration_seconds": now - prev,
                }
            )
            prev = now

    total_seconds = time.perf_counter() - t0

    result = final_output.get("result") or {}
    matches = result.get("matches")
    matches_count = len(matches) if isinstance(matches, list) else 0

    return {
        "total_seconds": total_seconds,
        "node_rows": node_rows,
        "next_action": _to_str(result.get("next_action")),
        "source": _to_str(result.get("source")),
        "matches_count": int(matches_count),
        "resolved_team_size": result.get("team_size"),
        "error": _to_str(result.get("error") or result.get("recommendation_error")),
        "final_output": final_output,
    }


def _ensure_faculty_emails_exist(
    *,
    all_emails: List[str],
) -> Dict[str, Any]:
    from services.agent_v2.agents import FacultyContextAgent

    fac_agent = FacultyContextAgent()
    resolved = fac_agent.resolve_faculties(emails=all_emails)
    return resolved


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(raw_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_runs:
        grouped.setdefault(str(row["scenario_name"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for scenario_name, rows in grouped.items():
        totals = [float(r["total_seconds"]) for r in rows]
        match_counts = [int(r["matches_count"]) for r in rows]
        out.append(
            {
                "scenario_name": scenario_name,
                "scenario_type": rows[0]["scenario_type"],
                "team_size_requested": rows[0]["team_size_requested"],
                "repeats": len(rows),
                "mean_seconds": statistics.fmean(totals) if totals else 0.0,
                "median_seconds": statistics.median(totals) if totals else 0.0,
                "p95_seconds": _percentile(totals, 95),
                "min_seconds": min(totals) if totals else 0.0,
                "max_seconds": max(totals) if totals else 0.0,
                "std_seconds": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
                "mean_matches_count": statistics.fmean(match_counts) if match_counts else 0.0,
            }
        )
    out.sort(key=lambda x: str(x["scenario_name"]))
    return out


def _maybe_plot(summary_rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generated: List[Path] = []

    # Plot 1: mean latency by scenario
    labels = [str(r["scenario_name"]) for r in summary_rows]
    means = [float(r["mean_seconds"]) for r in summary_rows]
    stds = [float(r["std_seconds"]) for r in summary_rows]

    fig_w = max(10.0, len(labels) * 1.2)
    plt.figure(figsize=(fig_w, 5.5))
    x = list(range(len(labels)))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Mean latency (seconds)")
    plt.title("Scenario Latency Benchmark")
    plt.tight_layout()
    p1 = out_dir / "scenario_latency_bar.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    generated.append(p1)

    # Plot 2: group scaling line by requested team size
    group_rows = [r for r in summary_rows if str(r.get("scenario_type")) == "group"]
    if group_rows:
        group_rows.sort(key=lambda r: int(r.get("team_size_requested") or 0))
        xs = [int(r.get("team_size_requested") or 0) for r in group_rows]
        ys = [float(r["mean_seconds"]) for r in group_rows]
        yerr = [float(r["std_seconds"]) for r in group_rows]
        plt.figure(figsize=(8.5, 5.0))
        plt.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4)
        plt.xlabel("Requested team size")
        plt.ylabel("Mean latency (seconds)")
        plt.title("Group Matching Latency Scaling")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p2 = out_dir / "group_scaling_line.png"
        plt.savefig(p2, dpi=160)
        plt.close()
        generated.append(p2)

    return generated


def _print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'scenario':42} {'type':8} {'team_k':7} {'mean(s)':>10} "
        f"{'p95(s)':>10} {'std(s)':>10} {'avg_matches':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{str(r['scenario_name']):42} "
            f"{str(r['scenario_type']):8} "
            f"{str(r['team_size_requested'] or '-'):7} "
            f"{float(r['mean_seconds']):10.3f} "
            f"{float(r['p95_seconds']):10.3f} "
            f"{float(r['std_seconds']):10.3f} "
            f"{float(r['mean_matches_count']):12.2f}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end latency for one-to-one and group matching scenarios, "
            "then export CSV + plots. No args is supported via config file."
        )
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON config path (default: bench/benchmark_scenarios.config.json).",
    )
    p.add_argument(
        "--write-config-template",
        action="store_true",
        help="Write a config template and exit.",
    )
    p.add_argument("--single-email", default=None, help="Faculty email for one-to-one scenarios.")
    p.add_argument(
        "--group-emails",
        default=None,
        help="Comma-separated faculty emails for group scenarios (at least 2).",
    )
    p.add_argument("--single-query-text", default=None, help="Query text for single scenario 2/3.")
    p.add_argument(
        "--single-broad-category",
        default=None,
        help="Broad category for single scenario 3 (supports comma list).",
    )
    p.add_argument("--single-top-k", type=int, default=None, help="Top-k grants for single scenarios.")
    p.add_argument(
        "--group-team-sizes",
        default=None,
        help="Comma-separated requested team sizes for group scaling.",
    )
    p.add_argument("--group-top-k", type=int, default=None, help="Top-k grants for group scenarios.")
    p.add_argument("--group-query-text", default=None, help="Optional query text for group scenarios.")
    p.add_argument(
        "--group-broad-category",
        default=None,
        help="Optional broad category for group scenarios (supports comma list).",
    )
    p.add_argument("--warmup", type=int, default=None, help="Warmup runs per scenario (not recorded).")
    p.add_argument("--repeats", type=int, default=None, help="Measured runs per scenario.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory where benchmark artifacts are written.",
    )
    p.add_argument(
        "--skip-email-precheck",
        action="store_true",
        help="Skip faculty-in-DB validation before running benchmarks.",
    )
    p.add_argument(
        "--dry-run-scenarios",
        action="store_true",
        help="Print generated scenarios and exit without calling the pipeline.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    config_path = Path(args.config).resolve()
    if args.write_config_template:
        _write_config_template(config_path)
        print(f"Wrote config template to: {config_path}")
        return

    cfg = _load_config(config_path)

    group_emails_cfg = cfg.get("group_emails")
    if isinstance(group_emails_cfg, list):
        group_emails_cfg = ",".join(str(x).strip() for x in group_emails_cfg if str(x).strip())

    settings = {
        "single_email": _pick(args.single_email, cfg.get("single_email"), ""),
        "group_emails": _pick(args.group_emails, group_emails_cfg, ""),
        "single_query_text": _pick(args.single_query_text, cfg.get("single_query_text"), "machine learning"),
        "single_broad_category": _pick(
            args.single_broad_category, cfg.get("single_broad_category"), "applied_research"
        ),
        "single_top_k": int(_pick(args.single_top_k, cfg.get("single_top_k"), 10)),
        "group_team_sizes": _pick(args.group_team_sizes, cfg.get("group_team_sizes"), "2,3,4,5"),
        "group_top_k": int(_pick(args.group_top_k, cfg.get("group_top_k"), 5)),
        "group_query_text": _pick(args.group_query_text, cfg.get("group_query_text"), ""),
        "group_broad_category": _pick(args.group_broad_category, cfg.get("group_broad_category"), ""),
        "warmup": int(_pick(args.warmup, cfg.get("warmup"), 1)),
        "repeats": int(_pick(args.repeats, cfg.get("repeats"), 3)),
        "output_dir": _pick(args.output_dir, cfg.get("output_dir"), str(PROJECT_ROOT / "outputs" / "benchmarks")),
        "skip_email_precheck": bool(cfg.get("skip_email_precheck", False)) or bool(args.skip_email_precheck),
        "dry_run_scenarios": bool(cfg.get("dry_run_scenarios", False)) or bool(args.dry_run_scenarios),
        "config_path": str(config_path),
    }

    auto_single, auto_group = _try_auto_fill_emails(
        str(settings["single_email"]),
        str(settings["group_emails"]),
    )
    settings["single_email"] = auto_single
    settings["group_emails"] = auto_group

    if not str(settings["single_email"]).strip():
        raise ValueError("single_email is required. Set it in config or pass --single-email.")
    if not str(settings["group_emails"]).strip():
        raise ValueError("group_emails is required. Set it in config or pass --group-emails.")

    if settings["warmup"] < 0:
        raise ValueError("--warmup must be >= 0")
    if settings["repeats"] < 1:
        raise ValueError("--repeats must be >= 1")

    scenarios = _build_scenarios(settings)
    if settings["dry_run_scenarios"]:
        print(json.dumps([{"name": s.name, "type": s.scenario_type, "payload": s.payload} for s in scenarios], indent=2))
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(str(settings["output_dir"])).resolve() / f"latency_benchmark_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_emails = _ordered_unique(
        [
            str(settings["single_email"]).strip().lower(),
            *[e.strip().lower() for e in _parse_csv_list(str(settings["group_emails"]))],
        ]
    )

    if not settings["skip_email_precheck"]:
        resolved = _ensure_faculty_emails_exist(all_emails=all_emails)
        if not resolved.get("all_in_db"):
            missing = resolved.get("missing_emails") or []
            raise RuntimeError(
                "Faculty precheck failed. Missing emails in DB: "
                + ", ".join(str(x) for x in missing)
            )

    from services.agent_v2 import GrantMatchOrchestrator, GrantMatchRequest

    orchestrator = GrantMatchOrchestrator()

    raw_runs: List[Dict[str, Any]] = []
    node_breakdown_rows: List[Dict[str, Any]] = []

    for spec in scenarios:
        print(f"\n[scenario] {spec.name}")
        for i in range(int(settings["warmup"])):
            warm_req = GrantMatchRequest(**spec.payload)
            _run_one_request(
                orchestrator,
                warm_req,
                thread_id=f"bench-{spec.name}-warmup-{i}-{int(time.time() * 1000)}",
            )
            print(f"  warmup {i + 1}/{settings['warmup']} done")

        for rep in range(1, int(settings["repeats"]) + 1):
            req = GrantMatchRequest(**spec.payload)
            out = _run_one_request(
                orchestrator,
                req,
                thread_id=f"bench-{spec.name}-rep-{rep}-{int(time.time() * 1000)}",
            )
            raw_row = {
                "scenario_name": spec.name,
                "scenario_type": spec.scenario_type,
                "team_size_requested": spec.team_size_requested,
                "iteration": rep,
                "total_seconds": float(out["total_seconds"]),
                "steps_count": len(out["node_rows"]),
                "next_action": out["next_action"],
                "source": out["source"],
                "matches_count": int(out["matches_count"]),
                "resolved_team_size": out["resolved_team_size"],
                "error": out["error"],
            }
            raw_runs.append(raw_row)
            for nr in out["node_rows"]:
                node_breakdown_rows.append(
                    {
                        "scenario_name": spec.name,
                        "scenario_type": spec.scenario_type,
                        "team_size_requested": spec.team_size_requested,
                        "iteration": rep,
                        "node_index": nr["node_index"],
                        "node": nr["node"],
                        "duration_seconds": nr["duration_seconds"],
                    }
                )
            print(
                f"  run {rep}/{settings['repeats']}: {out['total_seconds']:.3f}s "
                f"(next_action={out['next_action']}, matches={out['matches_count']})"
            )

    summary_rows = _summarize(raw_runs)

    _write_csv(
        out_dir / "benchmark_raw_runs.csv",
        raw_runs,
        [
            "scenario_name",
            "scenario_type",
            "team_size_requested",
            "iteration",
            "total_seconds",
            "steps_count",
            "next_action",
            "source",
            "matches_count",
            "resolved_team_size",
            "error",
        ],
    )
    _write_csv(
        out_dir / "benchmark_node_breakdown.csv",
        node_breakdown_rows,
        [
            "scenario_name",
            "scenario_type",
            "team_size_requested",
            "iteration",
            "node_index",
            "node",
            "duration_seconds",
        ],
    )
    _write_csv(
        out_dir / "benchmark_summary.csv",
        summary_rows,
        [
            "scenario_name",
            "scenario_type",
            "team_size_requested",
            "repeats",
            "mean_seconds",
            "median_seconds",
            "p95_seconds",
            "min_seconds",
            "max_seconds",
            "std_seconds",
            "mean_matches_count",
        ],
    )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "settings": settings,
        "scenario_names": [s.name for s in scenarios],
    }
    with (out_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    plot_paths = _maybe_plot(summary_rows, out_dir)

    print("\nBenchmark summary")
    _print_summary(summary_rows)
    print(f"\nArtifacts written to: {out_dir}")
    if plot_paths:
        print("Plots:")
        for p in plot_paths:
            print(f"  - {p}")
    else:
        print("Plots not generated (matplotlib unavailable in current environment).")


if __name__ == "__main__":
    main()
