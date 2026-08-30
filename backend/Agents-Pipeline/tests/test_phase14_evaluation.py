import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation.evaluator import (
    factual_accuracy_proxy,
    groundedness_score,
    precision_at_k,
    recall_at_k,
    keyword_evidence_score,
    aggregate_results,
    CaseResult,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "d"], 3) == 1 / 3


def test_keyword_evidence_score():
    assert keyword_evidence_score(["Tomato disease and fungal spots"], ["tomato", "fungal"]) == 1.0


def test_factual_accuracy_proxy():
    assert factual_accuracy_proxy("Use drainage and avoid overwatering.", ["drainage", "overwatering"]) == 1.0


def test_groundedness_score():
    assert groundedness_score("See [Source 1].", [{"doc_id": "doc-a"}], True) == 1.0


def test_aggregate_results():
    result = CaseResult(
        case_id="x",
        latency_ms=100,
        error=None,
        selected_agents=["disease_agent"],
        selected_tools=["vector_database"],
        expected_agents=["disease_agent"],
        expected_tools=["vector_database"],
        task_success=True,
        agent_selection_correct=True,
        tool_selection_correct=True,
        groundedness=1.0,
        factual_accuracy_proxy=1.0,
        retrieval={"recall@5": 1.0, "precision@5": 0.2},
        satisfaction=5,
    )
    report = aggregate_results([result])
    assert report["summary"]["task_success_rate"] == 1.0
    assert report["summary"]["error_rate"] == 0.0
    assert report["summary"]["satisfaction_1_to_5"] == 5.0
