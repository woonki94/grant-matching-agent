from __future__ import annotations

import json
from pathlib import Path

from train_cross_encoder.train_cross_encoder_paper_style import ALL_LOSSES, _build_parser, _load_pointwise_examples


def test_pointwise_mse_objective_is_registered():
    assert "pointwise_mse" in set(ALL_LOSSES)
    parser = _build_parser()
    objective_arg = next(a for a in parser._actions if a.dest == "objective")
    assert "pointwise_mse" in set(objective_arg.choices or [])


def test_load_pointwise_examples_reads_relevance_rows(tmp_path: Path):
    rows = [
        {"query": "q1", "doc": "d1", "relevance": 0.9, "confidence": 0.8, "query_spec_id": "q1"},
        {"query": "q2", "doc": "d2", "relevance_score": 0.2, "weight": 0.5, "query_spec_id": "q2"},
        {
            "query": "q3",
            "positive": "dp",
            "negative": "dn",
            "positive_teacher_score": 1.0,
            "negative_teacher_score": 0.1,
            "confidence": 0.7,
            "query_spec_id": "q3",
        },
    ]
    path = tmp_path / "pointwise.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    examples, stats = _load_pointwise_examples(
        dataset_jsonl=path,
        max_rows=0,
        min_weight=0.1,
        seed=42,
    )

    assert len(examples) == 4
    assert stats["pointwise_rows_used"] == 4
    assert all(0.0 <= float(x.relevance) <= 1.0 for x in examples)
