from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from train_cross_encoder.agentic_training.adapters import build_stage_datasets
from train_cross_encoder.agentic_training.models import (
    PairwisePreferenceSignal,
    PositiveCorrectionSignal,
    RelevanceLabelSignal,
)
from train_cross_encoder.agentic_training.workflow import AgenticTrainingConfig, AgenticTrainingWorkflow


class _StubVerifier:
    def verify_false_positive(self, task):
        return RelevanceLabelSignal(
            query=task.query,
            query_spec_id=task.query_spec_id,
            doc=task.doc,
            relevance=0.85,
            confidence=0.92,
            task_id=task.task_id,
            metadata={"agent": "false_positive_agent"},
        )

    def verify_false_negative(self, task):
        return PositiveCorrectionSignal(
            query=task.query,
            query_spec_id=task.query_spec_id,
            doc=task.doc,
            is_positive=True,
            confidence=0.91,
            task_id=task.task_id,
            metadata={"agent": "false_negative_agent"},
        )

    def verify_rank_check(self, task):
        return PairwisePreferenceSignal(
            query=task.query,
            query_spec_id=task.query_spec_id,
            doc_a=task.doc_a,
            doc_b=task.doc_b,
            preferred="A",
            confidence=0.93,
            task_id=task.task_id,
            metadata={"agent": "rank_check_agent"},
        )


class _TrainerSpy:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs or {}))
        out = Path(kwargs["output_dir"]).resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "dummy_model.txt").write_text("ok", encoding="utf-8")
        return {"output_dir": str(out), "objective": kwargs.get("objective")}


class _SequenceEvaluator:
    def __init__(self, seq: List[Dict[str, float]]) -> None:
        self.seq = list(seq)
        self.calls = 0

    def evaluate(self, **kwargs):
        _ = kwargs
        self.calls += 1
        if self.seq:
            return dict(self.seq.pop(0))
        return {"pairwise_accuracy": 0.0, "ndcg_at_10": 0.0, "pointwise_mse": 1.0}


def _synthetic_ranked_queries() -> List[Dict[str, Any]]:
    return [
        {
            "query_spec_id": "q1",
            "query": "robot locomotion reinforcement learning",
            "candidates": [
                {"i": 1, "text": "quadruped locomotion rl", "candidate_type": "topk", "cosine_sim": 0.92},
                {"i": 2, "text": "bipedal walking sim to real", "candidate_type": "hard_negative", "cosine_sim": 0.88},
                {"i": 3, "text": "financial compliance reporting", "candidate_type": "random_negative", "cosine_sim": 0.17},
            ],
            "ranking": [1, 2, 3],
        },
        {
            "query_spec_id": "q2",
            "query": "grant budget financial reporting compliance",
            "candidates": [
                {"i": 1, "text": "federal grant reporting compliance", "candidate_type": "topk", "cosine_sim": 0.95},
                {"i": 2, "text": "audit-ready budget governance", "candidate_type": "hard_negative", "cosine_sim": 0.83},
                {"i": 3, "text": "humanoid robot gait control", "candidate_type": "random_negative", "cosine_sim": 0.13},
            ],
            "ranking": [1, 2, 3],
        },
    ]


def _build_workflow(
    *,
    tmp_path: Path,
    trainer,
    evaluator,
    max_rounds: int = 5,
    quality_target: float = 0.78,
    patience: int = 2,
):
    calls = {"retrieval": 0}

    def _provider():
        calls["retrieval"] += 1
        return _synthetic_ranked_queries(), {"source": "stub"}

    cfg = AgenticTrainingConfig(
        output_dir=tmp_path,
        llm_model="",
        max_rounds=max_rounds,
        quality_target=quality_target,
        patience=patience,
        min_improvement=0.01,
        confidence_threshold=0.6,
        max_tasks_per_round=200,
        relabel_query_budget=2,
        holdout_ratio=0.1,
        train_epochs=0.1,
        min_stage_rows=1,
    )
    wf = AgenticTrainingWorkflow(
        config=cfg,
        candidate_provider=_provider,
        verifier=_StubVerifier(),
        trainer_fn=trainer,
        evaluator=evaluator,
    )
    return wf, calls


def test_router_dispatches_three_subagents(tmp_path: Path):
    trainer = _TrainerSpy()
    evaluator = _SequenceEvaluator([{"pairwise_accuracy": 1.0, "ndcg_at_10": 1.0, "pointwise_mse": 0.0}])
    wf, _ = _build_workflow(tmp_path=tmp_path, trainer=trainer, evaluator=evaluator)

    fp = wf._node_router(
        {
            "pending_tasks": [
                {
                    "task_id": "t1",
                    "task_type": "false_positive_check",
                    "round_index": 1,
                    "query": "q",
                    "query_spec_id": "q1",
                    "doc": "d",
                }
            ]
        }
    )
    assert fp["route_decision"] == "false_positive_agent"

    fn = wf._node_router(
        {
            "pending_tasks": [
                {
                    "task_id": "t2",
                    "task_type": "false_negative_check",
                    "round_index": 1,
                    "query": "q",
                    "query_spec_id": "q1",
                    "doc": "d",
                }
            ]
        }
    )
    assert fn["route_decision"] == "false_negative_agent"

    rc = wf._node_router(
        {
            "pending_tasks": [
                {
                    "task_id": "t3",
                    "task_type": "rank_check",
                    "round_index": 1,
                    "query": "q",
                    "query_spec_id": "q1",
                    "doc_a": "a",
                    "doc_b": "b",
                }
            ]
        }
    )
    assert rc["route_decision"] == "rank_check_agent"


def test_filter_signals_applies_confidence_and_dedup(tmp_path: Path):
    trainer = _TrainerSpy()
    evaluator = _SequenceEvaluator([{"pairwise_accuracy": 1.0, "ndcg_at_10": 1.0, "pointwise_mse": 0.0}])
    wf, _ = _build_workflow(tmp_path=tmp_path, trainer=trainer, evaluator=evaluator)

    state = {
        "round_index": 1,
        "round_raw_signals": [
            {
                "type": "relevance_label",
                "query": "q",
                "query_spec_id": "q1",
                "doc": "doc1",
                "relevance": 0.9,
                "confidence": 0.95,
            },
            {
                "type": "relevance_label",
                "query": "q",
                "query_spec_id": "q1",
                "doc": "doc1",
                "relevance": 0.8,
                "confidence": 0.96,
            },
            {
                "type": "relevance_label",
                "query": "q",
                "query_spec_id": "q1",
                "doc": "doc2",
                "relevance": 0.1,
                "confidence": 0.2,
            },
        ],
        "all_signals": [],
        "seen_signal_keys": [],
    }
    out = wf._node_filter_signals(state)
    assert len(out["round_filtered_signals"]) == 1
    assert len(out["all_signals"]) == 1


def test_dataset_conversion_outputs_three_dataset_types(tmp_path: Path):
    signals = [
        {
            "type": "relevance_label",
            "query": "q",
            "query_spec_id": "q1",
            "doc": "good doc",
            "relevance": 0.92,
            "confidence": 0.88,
        },
        {
            "type": "positive_correction",
            "query": "q",
            "query_spec_id": "q1",
            "doc": "hard positive",
            "is_positive": True,
            "confidence": 0.9,
        },
        {
            "type": "pairwise_preference",
            "query": "q",
            "query_spec_id": "q1",
            "doc_a": "doc A",
            "doc_b": "doc B",
            "preferred": "B",
            "confidence": 0.93,
        },
    ]
    bundle = build_stage_datasets(
        signals=signals,
        query_doc_pool={"q1": ["hard positive", "random negative", "other"]},
        output_dir=tmp_path,
        val_ratio=0.1,
        seed=42,
    )
    sizes = bundle["sizes"]
    assert sizes["pointwise_all"] == 1
    assert sizes["pairwise_all"] == 1
    assert sizes["contrastive_all"] == 1


def test_single_round_stops_when_quality_target_reached(tmp_path: Path):
    trainer = _TrainerSpy()
    evaluator = _SequenceEvaluator([{"pairwise_accuracy": 0.9, "ndcg_at_10": 0.9, "pointwise_mse": 0.1}])
    wf, calls = _build_workflow(
        tmp_path=tmp_path,
        trainer=trainer,
        evaluator=evaluator,
        max_rounds=5,
        quality_target=0.78,
        patience=2,
    )

    result = wf.run_sync()
    assert result.rounds_completed == 1
    assert result.stop_reason == "quality_target_reached"
    assert calls["retrieval"] == 1


def test_multi_round_stops_by_patience_and_retrieval_cached(tmp_path: Path):
    trainer = _TrainerSpy()
    evaluator = _SequenceEvaluator(
        [
            {"pairwise_accuracy": 0.55, "ndcg_at_10": 0.50, "pointwise_mse": 0.3},
            {"pairwise_accuracy": 0.55, "ndcg_at_10": 0.50, "pointwise_mse": 0.3},
            {"pairwise_accuracy": 0.55, "ndcg_at_10": 0.50, "pointwise_mse": 0.3},
        ]
    )
    wf, calls = _build_workflow(
        tmp_path=tmp_path,
        trainer=trainer,
        evaluator=evaluator,
        max_rounds=5,
        quality_target=0.99,
        patience=2,
    )

    result = wf.run_sync()
    assert result.rounds_completed == 3
    assert result.stop_reason == "patience_exhausted"
    assert calls["retrieval"] == 1


def test_stage_chaining_uses_previous_stage_output_as_next_input(tmp_path: Path):
    trainer = _TrainerSpy()
    evaluator = _SequenceEvaluator([{"pairwise_accuracy": 0.95, "ndcg_at_10": 0.95, "pointwise_mse": 0.05}])
    wf, _ = _build_workflow(
        tmp_path=tmp_path,
        trainer=trainer,
        evaluator=evaluator,
        max_rounds=1,
        quality_target=0.70,
        patience=2,
    )

    _ = wf.run_sync()
    assert len(trainer.calls) >= 3

    stage1 = trainer.calls[0]
    stage2 = trainer.calls[1]
    stage3 = trainer.calls[2]

    assert str(stage2.get("model_name")) == str(stage1.get("output_dir"))
    assert str(stage3.get("model_name")) == str(stage2.get("output_dir"))

    summary_path = tmp_path / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("best_model_path")
