"""
main.py
=======
PHASE 7/10/11/13 — Agents Pipeline (interactive CLI)

Run this to ask AgriNova AI questions through the full multi-agent
pipeline (Planner -> specialists, collaborating in sequence -> Report
Agent) from the terminal:

    python main.py
    python main.py "my tomato leaves have brown spots, what should I do?"

Phase 9 — attach a photo from the CLI with --image:

    python main.py "what's wrong with my tomato plant?" --image leaf.jpg

Phase 11 — carry conversation memory across runs with --session:

    python main.py "my tomato crop in Kurunegala has yellowing leaves" --session farmer-42
    python main.py "should I irrigate today?" --session farmer-42
    # ^ the second call already knows the crop and location from the first

In interactive mode, a session ID is generated automatically (or pass
one with --session to resume an earlier farmer's conversation from a
previous run — memory is persisted to disk, see conversation_memory.py)
so every question asked in one CLI run naturally shares memory, and
`--reset-memory` clears it before starting.

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
import uuid
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
    if result.session_id and result.recalled_memory:
        recalled_bits = ", ".join(f"{k}={v}" for k, v in result.recalled_memory.items())
        print(f"[Phase 11 memory] Recalled for session '{result.session_id}': {recalled_bits}")
    elif result.session_id:
        print(f"[Phase 11 memory] Session '{result.session_id}' — nothing recalled yet (first turn).")
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

    # PHASE 13 — Explainable AI: Recommendation -> Reason -> Supporting
    # documents -> Confidence -> References, parsed/computed from the
    # final report above (see explainability.py). Printed separately so
    # the structure a farmer would see in the frontend is visible here
    # too, not just the raw report text.
    if result.explanation:
        exp = result.explanation
        print(f"\n{'='*70}\nEXPLANATION  (Phase 13 — Explainable AI)\n{'='*70}")
        print(f"RECOMMENDATION\n  {exp.recommendation}")
        if exp.reason:
            print(f"\nREASON\n  {exp.reason}")
        if exp.next_steps:
            print(f"\nRECOMMENDED NEXT STEPS\n  {exp.next_steps}")
        confidence = exp.confidence
        print(f"\nCONFIDENCE: {confidence['level']} ({confidence['score']:.0%})")
        for factor in confidence["factors"]:
            print(f"  - {factor}")
        if exp.references:
            print("\nREFERENCES")
            for ref in exp.references:
                sim = f" ({ref['similarity']:.0%} match)" if ref.get("similarity") is not None else ""
                print(f"  [{ref['n']}] {ref['label']}{sim}")
    elif result.final_report.sources:
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
    """Very small parser: pulls a trailing `--image <path>`,
    `--session <id>` and/or a bare `--reset-memory` flag out of argv;
    everything else is joined back into the question, so quoting the
    question isn't required for the common case."""
    image_path = None
    session_id = None
    reset_memory = False
    args = list(argv)

    if "--image" in args:
        idx = args.index("--image")
        if idx + 1 < len(args):
            image_path = args[idx + 1]
            del args[idx:idx + 2]
        else:
            del args[idx:idx + 1]

    if "--session" in args:
        idx = args.index("--session")
        if idx + 1 < len(args):
            session_id = args[idx + 1]
            del args[idx:idx + 2]
        else:
            del args[idx:idx + 1]

    if "--reset-memory" in args:
        args.remove("--reset-memory")
        reset_memory = True

    return " ".join(args).strip(), image_path, session_id, reset_memory


def run_once(
    question: str,
    image_path: Optional[str] = None,
    session_id: Optional[str] = None,
    reset_memory: bool = False,
):
    orchestrator = AgentOrchestrator()
    if reset_memory and session_id:
        orchestrator.memory_store.delete(session_id)
        print(f"[Phase 11 memory] Cleared memory for session '{session_id}'.")

    context = {}
    if image_path:
        image_base64 = _load_image_base64(image_path)
        if image_base64:
            context["image_base64"] = image_base64
    result = orchestrator.handle(question, context=context, session_id=session_id)
    print_answer(result)


def run_interactive(session_id: Optional[str] = None, reset_memory: bool = False):
    print(f"\n{'='*70}\nAgriNova AI — Agents Pipeline "
          f"(Phase 10 — Multi-Agent Collaboration, Phase 11 — Conversation Memory)\n{'='*70}")
    print(f"Planner mode: {agent_config.PLANNER_MODE} | Collaboration mode: {agent_config.COLLABORATION_MODE}")

    orchestrator = AgentOrchestrator()

    # PHASE 11 — every question asked in this run shares one session_id
    # so memory naturally carries across turns (the "Day 1 / Day 2"
    # scenario, just within one CLI session instead of two). Pass
    # --session to instead RESUME a farmer's conversation from an
    # earlier run (memory is persisted to disk between runs).
    if not session_id:
        session_id = f"cli-{uuid.uuid4().hex[:8]}"
        print(f"Session: {session_id} (new — resume it later with --session {session_id})")
    else:
        existing = orchestrator.memory_store.get(session_id)
        if reset_memory:
            orchestrator.memory_store.delete(session_id)
            print(f"Session: {session_id} (memory cleared)")
        elif not existing.is_empty():
            print(f"Session: {session_id} (resumed — remembers: {existing.known_context()})")
        else:
            print(f"Session: {session_id} (no memory yet)")

    print("Type a farming question, or 'exit' to quit.\n")

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

        result = orchestrator.handle(question, session_id=session_id)
        print_answer(result)


if __name__ == "__main__":
    configure_logging()
    cli_question, cli_image_path, cli_session_id, cli_reset_memory = _parse_cli_args(sys.argv[1:])
    if cli_question:
        run_once(
            cli_question,
            image_path=cli_image_path,
            session_id=cli_session_id,
            reset_memory=cli_reset_memory,
        )
    else:
        run_interactive(session_id=cli_session_id, reset_memory=cli_reset_memory)
