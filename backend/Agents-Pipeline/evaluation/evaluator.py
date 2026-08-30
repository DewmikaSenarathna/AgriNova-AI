"""
evaluation/evaluator.py
=======================
PHASE 14 — System Evaluation.

This module evaluates the running AgriNova AI system instead of only
checking whether one request "works".

Metrics:
  Retrieval:
    - Recall@K / Precision@K when gold relevant document IDs are supplied
    - keyword-recall/precision proxy when a case has required evidence terms

  Generation:
    - groundedness (evidence/citation support)
    - factual-accuracy proxy (required-fact coverage against a reference)

  Agent workflow:
    - task success rate
    - expected-agent selection accuracy

  Performance / reliability:
    - end-to-end latency (average, median, p95, max)
    - error rate

  Tools:
    - correct tool selection rate

  User experience:
    - average 1–5 satisfaction score from an optional ratings JSON file

IMPORTANT:
Automatic factual-accuracy checks are intentionally called a
"proxy". True factual accuracy requires human or domain-expert
verification (or a separately validated evaluator). This module never
pretends that keyword matching is proof of truth.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from agent_types import AgentResult
from rag_bridge import Retriever, VectorStoreEmpty, VectorStoreUnavailable


@dataclass
class EvaluationCase:
    id: str
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    expected_agents: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)

    # Gold retrieval labels. Use stable document IDs from your ChromaDB.
    relevant_doc_ids: List[str] = field(default_factory=list)

    # Optional evidence terms used for a weaker, automatic retrieval proxy.
    # These are NOT a replacement for human-created gold labels.
    required_evidence_terms: List[str] = field(default_factory=list)

    # Optional generation facts. Each fact is a phrase that should be
    # supported by the answer/reference answer.
    required_facts: List[str] = field(default_factory=list)
    reference_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "expected_agents": self.expected_agents,
            "expected_tools": self.expected_tools,
            "relevant_doc_ids": self.relevant_doc_ids,
            "required_evidence_terms": self.required_evidence_terms,
            "required_facts": self.required_facts,
            "reference_answer": self.reference_answer,
        }


@dataclass
class CaseResult:
    case_id: str
    latency_ms: float
    error: Optional[str]
    selected_agents: List[str]
    selected_tools: List[str]
    expected_agents: List[str]
    expected_tools: List[str]
    task_success: bool
    agent_selection_correct: Optional[bool]
    tool_selection_correct: Optional[bool]
    groundedness: float
    factual_accuracy_proxy: Optional[float]
    retrieval: Dict[str, Any]
    satisfaction: Optional[float]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "selected_agents": self.selected_agents,
            "selected_tools": self.selected_tools,
            "expected_agents": self.expected_agents,
            "expected_tools": self.expected_tools,
            "task_success": self.task_success,
            "agent_selection_correct": self.agent_selection_correct,
            "tool_selection_correct": self.tool_selection_correct,
            "groundedness": round(self.groundedness, 3),
            "factual_accuracy_proxy": (
                round(self.factual_accuracy_proxy, 3)
                if self.factual_accuracy_proxy is not None else None
            ),
            "retrieval": self.retrieval,
            "satisfaction": self.satisfaction,
        }


def load_cases(path: Path) -> List[EvaluationCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Evaluation dataset must contain a JSON list: {path}")
    return [EvaluationCase(**item) for item in data]


def load_ratings(path: Optional[Path]) -> Dict[str, float]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Ratings file must contain an object: {path}")
    ratings = {}
    for case_id, value in data.items():
        score = float(value)
        if not 1 <= score <= 5:
            raise ValueError(f"Satisfaction score for {case_id} must be between 1 and 5.")
        ratings[case_id] = score
    return ratings


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> Optional[float]:
    """Standard Recall@K over explicit gold document IDs."""
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None
    top = list(retrieved_doc_ids[:k])
    return len(set(top) & relevant) / len(relevant)


def precision_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> Optional[float]:
    """Standard Precision@K over explicit gold document IDs."""
    relevant = set(relevant_doc_ids)
    if not relevant or k <= 0:
        return None
    top = list(retrieved_doc_ids[:k])
    return len(set(top) & relevant) / k


def keyword_evidence_score(
    retrieved_texts: Sequence[str], required_terms: Sequence[str]
) -> Optional[float]:
    """
    Automatic retrieval proxy: percentage of required evidence terms that
    occur in the retrieved text. Useful before a human labels gold docs.
    """
    terms = [t.strip().lower() for t in required_terms if t.strip()]
    if not terms:
        return None
    blob = "\n".join(retrieved_texts).lower()
    return sum(term in blob for term in terms) / len(terms)


def groundedness_score(answer: str, sources: Sequence[Dict[str, Any]], grounded: bool) -> float:
    """
    Conservative heuristic:
      + 0.5 if the system marked the answer grounded
      + 0.5 if the answer cites at least one source that actually exists
    """
    score = 0.0
    if grounded:
        score += 0.5
    citation_count = 0
    text = answer or ""
    for i in range(1, len(sources) + 1):
        if f"[Source {i}]" in text:
            citation_count += 1
    if sources and citation_count > 0:
        score += 0.5
    elif not sources and not grounded:
        # Honest no-evidence responses are not falsely rewarded as grounded.
        score += 0.0
    return score


def factual_accuracy_proxy(answer: str, required_facts: Sequence[str]) -> Optional[float]:
    """
    Required-fact coverage, not a proof of factual accuracy.
    Each required fact is matched case-insensitively as a phrase.
    """
    facts = [f.strip().lower() for f in required_facts if f.strip()]
    if not facts:
        return None
    text = (answer or "").lower()
    return sum(fact in text for fact in facts) / len(facts)


def _infer_tools(agent_results: Sequence[AgentResult]) -> Set[str]:
    """
    Infer tool provenance from the existing Phase 9 result structure.
    This avoids changing every agent solely to expose evaluation metadata.
    """
    tools: Set[str] = set()
    for result in agent_results:
        name = result.agent_name
        sources = result.sources or []
        joined = json.dumps(sources, ensure_ascii=False).lower()

        if name == "weather_agent" or "open-meteo" in joined:
            tools.add("weather_api")
        if name == "market_agent" and ("market" in joined or result.data.get("crop")):
            tools.add("market_price_api")
        if name == "government_agent":
            if "government pdf search" in joined or "pdf" in joined:
                tools.add("government_pdf_search")
            if "vector" in joined or any("chunk_id" in s for s in sources):
                tools.add("vector_database")
        if name in {"disease_agent", "pest_agent", "fertilizer_agent", "soil_agent", "general"}:
            if sources or result.grounded:
                tools.add("vector_database")
        if name == "image_agent":
            tools.add("image_model")
    return tools


def _agent_selection_correct(actual: Sequence[str], expected: Sequence[str]) -> Optional[bool]:
    if not expected:
        return None
    return set(actual) == set(expected)


def _tool_selection_correct(actual: Set[str], expected: Sequence[str]) -> Optional[bool]:
    if not expected:
        return None
    return actual == set(expected)


def _task_success(
    *,
    error: Optional[str],
    actual_agents: Sequence[str],
    expected_agents: Sequence[str],
    groundedness: float,
    factual_proxy: Optional[float],
) -> bool:
    if error:
        return False
    if expected_agents and set(actual_agents) != set(expected_agents):
        return False
    if groundedness < 0.5:
        return False
    if factual_proxy is not None and factual_proxy < 0.5:
        return False
    return True


class SystemEvaluator:
    def __init__(self, orchestrator, retriever: Optional[Retriever] = None):
        self.orchestrator = orchestrator
        self.retriever = retriever

    def evaluate_case(self, case: EvaluationCase, satisfaction: Optional[float] = None) -> CaseResult:
        started = time.perf_counter()
        error = None
        answer = None

        try:
            answer = self.orchestrator.handle(case.question, context=case.context)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000

        if error or answer is None:
            return CaseResult(
                case_id=case.id,
                latency_ms=latency_ms,
                error=error,
                selected_agents=[],
                selected_tools=[],
                expected_agents=case.expected_agents,
                expected_tools=case.expected_tools,
                task_success=False,
                agent_selection_correct=False if case.expected_agents else None,
                tool_selection_correct=False if case.expected_tools else None,
                groundedness=0.0,
                factual_accuracy_proxy=None,
                retrieval={},
                satisfaction=satisfaction,
            )

        selected_agents = list(answer.plan.agents_to_run)
        selected_tools = sorted(_infer_tools(answer.agent_results))
        final = answer.final_report
        grounded = groundedness_score(final.details or final.summary, final.sources, final.grounded)
        factual = factual_accuracy_proxy(
            final.details or final.summary, case.required_facts
        )

        retrieval = self._evaluate_retrieval(case, answer)

        success = _task_success(
            error=None,
            actual_agents=selected_agents,
            expected_agents=case.expected_agents,
            groundedness=grounded,
            factual_proxy=factual,
        )

        return CaseResult(
            case_id=case.id,
            latency_ms=latency_ms,
            error=None,
            selected_agents=selected_agents,
            selected_tools=selected_tools,
            expected_agents=case.expected_agents,
            expected_tools=case.expected_tools,
            task_success=success,
            agent_selection_correct=_agent_selection_correct(selected_agents, case.expected_agents),
            tool_selection_correct=_tool_selection_correct(set(selected_tools), case.expected_tools),
            groundedness=grounded,
            factual_accuracy_proxy=factual,
            retrieval=retrieval,
            satisfaction=satisfaction,
        )

    def _evaluate_retrieval(self, case: EvaluationCase, answer) -> Dict[str, Any]:
        # Use the evidence actually surfaced by specialist agents/report.
        chunks = []
        for result in answer.agent_results:
            chunks.extend(result.sources or [])
        chunks.extend(answer.final_report.sources or [])

        retrieved_doc_ids = [
            str(s.get("doc_id"))
            for s in chunks
            if s.get("doc_id")
        ]
        retrieved_texts = [
            str(s.get("text", ""))
            for s in chunks
            if s.get("text")
        ]

        metrics: Dict[str, Any] = {}
        if case.relevant_doc_ids:
            for k in (1, 3, 5):
                metrics[f"recall@{k}"] = recall_at_k(
                    retrieved_doc_ids, case.relevant_doc_ids, k
                )
                metrics[f"precision@{k}"] = precision_at_k(
                    retrieved_doc_ids, case.relevant_doc_ids, k
                )

        proxy = keyword_evidence_score(retrieved_texts, case.required_evidence_terms)
        if proxy is not None:
            metrics["keyword_evidence_coverage"] = proxy

        metrics["retrieved_source_count"] = len(chunks)
        return metrics

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        ratings = ratings or {}
        results = [
            self.evaluate_case(case, satisfaction=ratings.get(case.id))
            for case in cases
        ]
        return aggregate_results(results)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in values if x is not None]
    return statistics.mean(xs) if xs else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def aggregate_results(results: Sequence[CaseResult]) -> Dict[str, Any]:
    total = len(results)
    errors = sum(bool(r.error) for r in results)
    successful = sum(r.task_success for r in results)
    agent_scores = [r.agent_selection_correct for r in results if r.agent_selection_correct is not None]
    tool_scores = [r.tool_selection_correct for r in results if r.tool_selection_correct is not None]
    grounded = [r.groundedness for r in results]
    factual = [r.factual_accuracy_proxy for r in results if r.factual_accuracy_proxy is not None]
    satisfaction = [r.satisfaction for r in results if r.satisfaction is not None]
    latencies = [r.latency_ms for r in results]

    retrieval_metrics = {}
    for key in ("recall@1", "recall@3", "recall@5", "precision@1", "precision@3", "precision@5", "keyword_evidence_coverage"):
        value = _mean(r.retrieval.get(key) for r in results)
        if value is not None:
            retrieval_metrics[key] = round(value, 3)

    return {
        "summary": {
            "cases": total,
            "task_success_rate": round(successful / total, 3) if total else 0.0,
            "error_rate": round(errors / total, 3) if total else 0.0,
            "agent_selection_accuracy": round(statistics.mean(agent_scores), 3) if agent_scores else None,
            "tool_selection_accuracy": round(statistics.mean(tool_scores), 3) if tool_scores else None,
            "groundedness": round(statistics.mean(grounded), 3) if grounded else 0.0,
            "factual_accuracy_proxy": round(statistics.mean(factual), 3) if factual else None,
            "satisfaction_1_to_5": round(statistics.mean(satisfaction), 2) if satisfaction else None,
            "latency_ms": {
                "average": round(statistics.mean(latencies), 2) if latencies else None,
                "median": round(statistics.median(latencies), 2) if latencies else None,
                "p95": round(percentile(latencies, 0.95), 2) if latencies else None,
                "max": round(max(latencies), 2) if latencies else None,
            },
            "retrieval": retrieval_metrics,
            "notes": [
                "Recall@K and Precision@K require manually curated relevant_doc_ids.",
                "keyword_evidence_coverage is only a retrieval proxy.",
                "factual_accuracy_proxy is required-fact phrase coverage, not proof of truth.",
                "Task success requires expected-agent match when labels exist and groundedness >= 0.5.",
            ],
        },
        "cases": [r.to_dict() for r in results],
    }


def save_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
