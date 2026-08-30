"""
run_evaluation.py
=================
PHASE 14 — command-line evaluation runner.

Examples:

    # Run all cases
    python evaluation/run_evaluation.py

    # Run selected cases
    python evaluation/run_evaluation.py --case weather_01 --case disease_01

    # Add manually reviewed UX scores from ratings.json
    python evaluation/run_evaluation.py --ratings evaluation/ratings.json

The runner intentionally uses the real AgentOrchestrator. Therefore
it measures actual end-to-end latency, planner behaviour, retrieval,
tools, generation, errors and final reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make `backend/Agents-Pipeline` importable when this file is run directly.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent_config
from agent_orchestrator import AgentOrchestrator
from evaluation.evaluator import (
    load_cases,
    load_ratings,
    save_report,
    SystemEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AgriNova AI Phase 14.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=HERE / "evaluation" / "evaluation_dataset.json",
        help="Path to the evaluation dataset JSON.",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=HERE / "evaluation" / "ratings.json",
        help="Optional JSON file containing case_id -> satisfaction score (1-5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "output" / "evaluation" / "latest_report.json",
        help="Where to save the evaluation report.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only this case ID. Repeat the flag for multiple cases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = load_cases(args.dataset)

    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.id in selected]
        missing = selected - {case.id for case in cases}
        if missing:
            raise SystemExit(f"Unknown evaluation case(s): {', '.join(sorted(missing))}")

    ratings = load_ratings(args.ratings) if args.ratings.exists() else {}

    print("=" * 72)
    print("AgriNova AI — Phase 14 System Evaluation")
    print("=" * 72)
    print(f"Cases: {len(cases)}")
    print(f"Planner mode: {agent_config.PLANNER_MODE}")
    print(f"Collaboration mode: {agent_config.COLLABORATION_MODE}")
    print()

    orchestrator = AgentOrchestrator()
    evaluator = SystemEvaluator(orchestrator)
    report = evaluator.evaluate(cases, ratings=ratings)

    summary = report["summary"]
    print(f"Task success rate       : {summary['task_success_rate']:.1%}")
    print(f"Error rate              : {summary['error_rate']:.1%}")
    print(f"Groundedness            : {summary['groundedness']:.1%}")

    if summary["factual_accuracy_proxy"] is not None:
        print(f"Factual accuracy proxy  : {summary['factual_accuracy_proxy']:.1%}")
    if summary["agent_selection_accuracy"] is not None:
        print(f"Agent selection accuracy: {summary['agent_selection_accuracy']:.1%}")
    if summary["tool_selection_accuracy"] is not None:
        print(f"Tool selection accuracy : {summary['tool_selection_accuracy']:.1%}")
    if summary["satisfaction_1_to_5"] is not None:
        print(f"Satisfaction (1-5)      : {summary['satisfaction_1_to_5']:.2f}")

    latency = summary["latency_ms"]
    print(
        f"Latency ms              : avg={latency['average']}, "
        f"median={latency['median']}, p95={latency['p95']}, max={latency['max']}"
    )

    if summary["retrieval"]:
        print("Retrieval metrics:")
        for key, value in summary["retrieval"].items():
            print(f"  {key:28s}: {value:.1%}")

    print()
    print("Case results:")
    for case in report["cases"]:
        status = "PASS" if case["task_success"] else "FAIL"
        print(
            f"  [{status}] {case['case_id']} | "
            f"{case['latency_ms']:.0f} ms | "
            f"agents={','.join(case['selected_agents']) or '-'} | "
            f"tools={','.join(case['selected_tools']) or '-'}"
        )
        if case["error"]:
            print(f"         error: {case['error']}")

    save_report(report, args.output)
    print()
    print(f"Full JSON report: {args.output}")


if __name__ == "__main__":
    main()
