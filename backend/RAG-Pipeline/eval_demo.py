"""
eval_demo.py
"""

import sys

import config
from llm_client import LLMClient, LLMError
from rag_pipeline import RAGPipeline

NO_RAG_SYSTEM_PROMPT = (
    "You are a helpful agricultural assistant. Answer the farmer's question as best you can "
    "from your own general knowledge."
)


def without_rag(question: str) -> str:
    llm = LLMClient()
    try:
        return llm.generate(NO_RAG_SYSTEM_PROMPT, question)
    except LLMError as e:
        return f"(LLM unavailable: {e})"


def with_rag(question: str):
    pipeline = RAGPipeline()
    return pipeline.answer(question)


def main():
    question = " ".join(sys.argv[1:]).strip() or "how do I treat aphids on tomato plants"

    print(f"\n{'='*70}\nQUESTION: {question}\n{'='*70}")

    print(f"\n--- WITHOUT RAG (LLM guesses, no evidence) ---")
    print(without_rag(question))

    print(f"\n--- WITH RAG (grounded in retrieved documents) ---")
    result = with_rag(question)
    print(result.answer)
    if result.sources:
        print(f"\nSources used:")
        for i, s in enumerate(result.sources, start=1):
            print(f"  [Source {i}] {s['doc_id']} › {s['heading']} (similarity {s['similarity']:.2f})")
    else:
        print("\n(No sources were relevant enough to use — this fell back to an honest 'I don't know'.)")
    print()


if __name__ == "__main__":
    main()
