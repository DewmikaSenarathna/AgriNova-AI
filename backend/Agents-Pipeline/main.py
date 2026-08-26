"""
main.py
=======
PHASE 7/10 — Agents Pipeline (interactive CLI)

Run this to ask AgriNova AI questions through the full multi-agent
pipeline (Planner -> specialists, collaborating in sequence -> Report
Agent) from the terminal:

    python main.py
    python main.py "my tomato leaves have brown spots, what should I do?"

Phase 9 — attach a photo from the CLI with --image:

    python main.py "what's wrong with my tomato plant?" --image leaf.jpg

Prerequisites (in order):
    1. Document-Processing-Pipeline has processed at least one PDF.
    2. Chunking-Embedding-Pipeline has chunked + embedded it.
    3. An LLM backend is reachable (see ../RAG-Pipeline/.env.example —
       Agents-Pipeline reuses the same LLM_PROVIDER settings).
    4. (Optional, for --image) A vision-capable model is configured —
       see ../RAG-Pipeline/.env.example's OLLAMA_VISION_MODEL /
       GROQ_VISION_MODEL / OPENAI_COMPATIBLE_VISION_MODEL.
"""

import base64
import logging
import sys
from pathlib import Path
from typing import Optional

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
    print(f"PLANNER'S REASONING  (via {result.plan.method}, collaboration={result.collaboration_mode})")
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


def _load_image_base64(image_path: str) -> Optional[str]:
    path = Path(image_path)
    if not path.exists():
        print(f"[warning] --image path not found: {image_path} (continuing without it)")
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_cli_args(argv):
    """Very small parser: everything except a trailing `--image <path>`
    pair is joined back into the question, so quoting the question isn't
    required for the common case."""
    image_path = None
    args = list(argv)
    if "--image" in args:
        idx = args.index("--image")
        if idx + 1 < len(args):
            image_path = args[idx + 1]
            del args[idx:idx + 2]
        else:
            del args[idx:idx + 1]
    return " ".join(args).strip(), image_path


def run_once(question: str, image_path: Optional[str] = None):
    orchestrator = AgentOrchestrator()
    context = {}
    if image_path:
        image_base64 = _load_image_base64(image_path)
        if image_base64:
            context["image_base64"] = image_base64
    result = orchestrator.handle(question, context=context)
    print_answer(result)


def run_interactive():
    print(f"\n{'='*70}\nAgriNova AI — Agents Pipeline (Phase 10 — Multi-Agent Collaboration)\n{'='*70}")
    print(f"Planner mode: {agent_config.PLANNER_MODE} | Collaboration mode: {agent_config.COLLABORATION_MODE}")
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
