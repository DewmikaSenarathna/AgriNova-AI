"""
main.py
"""

import logging
import sys

import config
from rag_pipeline import RAGPipeline


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_answer(result):
    print(f"\n{'='*70}")
    print("ANSWER" + ("" if result.grounded else "  (not grounded in the knowledge base)"))
    print("=" * 70)
    print(result.answer)

    if result.sources:
        print(f"\n{'-'*70}\nSOURCES USED\n{'-'*70}")
        for i, s in enumerate(result.sources, start=1):
            print(
                f"[Source {i}] {s['doc_id']}  ›  {s['heading']}  "
                f"(similarity {s['similarity']:.2f})"
            )
    print()


def run_once(question: str, top_k: int = None):
    pipeline = RAGPipeline()
    result = pipeline.answer(question, top_k=top_k)
    print_answer(result)


def run_interactive():
    print(f"\n{'='*70}\nAgriNova AI — RAG Pipeline (Phase 6)\n{'='*70}")
    print(f"LLM provider: {config.LLM_PROVIDER}   |   Top-K: {config.RETRIEVAL_TOP_K}")
    print("Type a farming question, or 'exit' to quit.\n")

    pipeline = RAGPipeline()
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

        result = pipeline.answer(question)
        print_answer(result)


if __name__ == "__main__":
    configure_logging()
    cli_question = " ".join(sys.argv[1:]).strip()
    if cli_question:
        run_once(cli_question)
    else:
        run_interactive()
