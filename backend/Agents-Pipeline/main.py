"""
main.py
"""

import logging
import sys

import agent_config
from agent_orchestrator import AgentOrchestrator, OrchestratedAnswer


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, agent_config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_answer(result: OrchestratedAnswer):
    print(f"\n{'='*70}")
    print(f"PLANNER'S REASONING  (via {result.plan.method})")
    print("=" * 70)
    if result.plan.steps:
        for i, step in enumerate(result.plan.steps, start=1):
            arrow = "" if i == len(result.plan.steps) else "\n   ↓"
            print(f"{i}. Need {step.need}  →  {step.agent}")
            print(f"   ({step.reason}){arrow}")
    else:
        print(f"Agents run: {', '.join(result.plan.agents_to_run)}")
        if result.plan.reasoning:
            print(f"Reasoning: {result.plan.reasoning}")

    for r in result.agent_results:
        print(f"\n{'-'*70}\n{r.agent_name.upper()}  (grounded={r.grounded})\n{'-'*70}")
        print(r.details or r.summary)
        if r.error:
            print(f"[error: {r.error}]")

    print(f"\n{'='*70}\nFINAL REPORT\n{'='*70}")
    print(result.final_report.details or result.final_report.summary)
    if result.final_report.sources:
        print(f"\n{'-'*70}\nCOMBINED SOURCES\n{'-'*70}")
        for i, s in enumerate(result.final_report.sources, start=1):
            print(f"[{i}] {s}")
    print()


def run_once(question: str):
    orchestrator = AgentOrchestrator()
    result = orchestrator.handle(question)
    print_answer(result)


def run_interactive():
    print(f"\n{'='*70}\nAgriNova AI — Agents Pipeline (Phase 7)\n{'='*70}")
    print(f"Planner mode: {agent_config.PLANNER_MODE}")
    print("Type a farming question, or 'exit' to quit.\n")

    orchestrator = AgentOrchestrator()
    while True:
        try:
            question = input("Farmer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        result = orchestrator.handle(question)
        print_answer(result)


if __name__ == "__main__":
    configure_logging()
    cli_question = " ".join(sys.argv[1:]).strip()
    if cli_question:
        run_once(cli_question)
    else:
        run_interactive()
