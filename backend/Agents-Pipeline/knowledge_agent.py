"""
knowledge_agent.py
"""

import logging
from typing import List, Optional

from base_agent import BaseAgent
from agent_types import AgentRequest, AgentResult
from rag_bridge import (
    Retriever,
    LLMClient,
    LLMError,
    RetrievedChunk,
    VectorStoreEmpty,
    VectorStoreUnavailable,
    build_context_block,
    rag_config,
)

logger = logging.getLogger(__name__)


class KnowledgeAgent(BaseAgent):
    """
    Subclasses must set:
      name, description   -> from BaseAgent
      domain_label         -> short human label, e.g. "crop disease"
      query_hints           -> list[str] prepended to the farmer's raw
                               question before embedding, to bias
                               retrieval toward this agent's domain
                               (e.g. ["crop disease", "symptoms", "treatment"])
      system_prompt          -> domain-specific instructions for the LLM
    """

    domain_label: str = "agriculture"
    query_hints: List[str] = []
    system_prompt: str = (
        "You are AgriNova AI, a trustworthy agricultural assistant. Answer using ONLY the "
        "numbered SOURCE excerpts provided, citing them as [Source N]. If the sources don't "
        "fully answer the question, say so plainly and suggest the farmer consult a local "
        "agricultural extension officer. Never invent dosages, chemical names, or figures "
        "that are not present in the sources. Keep the answer short, plain-language and "
        "actionable."
    )

    def __init__(self, retriever: Optional[Retriever] = None, llm: Optional[LLMClient] = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    # -- Step A — Domain-biased query expansion 
    def _expand_query(self, question: str) -> str:
        if not self.query_hints:
            return question
        return f"{' '.join(self.query_hints)}: {question}"

    # -- Step B — Retrieve + generate, grounded in the shared knowledge base -
    def run(self, request: AgentRequest) -> AgentResult:
        question = (request.query or "").strip()
        if not question:
            return AgentResult(
                agent_name=self.name,
                summary=f"No question was provided to the {self.domain_label} agent.",
                grounded=False,
            )

        expanded_query = self._expand_query(question)

        try:
            chunks: List[RetrievedChunk] = self.retriever.retrieve(expanded_query)
        except (VectorStoreUnavailable, VectorStoreEmpty) as e:
            return AgentResult(
                agent_name=self.name,
                summary=(
                    f"The knowledge base isn't ready yet, so the {self.domain_label} agent "
                    f"can't look up trusted sources right now."
                ),
                details=str(e),
                grounded=False,
            )

        if not chunks:
            return AgentResult(
                agent_name=self.name,
                summary=(
                    f"No sources specific to {self.domain_label} were found in the "
                    f"knowledge base for this question."
                ),
                details=(
                    f"I don't have grounded information on this in the knowledge base yet. "
                    f"Please confirm with a local agricultural extension officer before acting."
                ),
                grounded=False,
            )

        context_block, used_chunks = build_context_block(chunks)
        user_prompt = (
            f"SOURCES:\n{context_block}\n\n---\n\n"
            f"FARMER'S QUESTION ({self.domain_label}): {question}\n\n"
            f"Answer using only the SOURCES above, citing them as [Source N]."
        )

        try:
            answer_text = self.llm.generate(self.system_prompt, user_prompt)
        except LLMError as e:
            return AgentResult(
                agent_name=self.name,
                summary=(
                    f"Found relevant {self.domain_label} sources but couldn't reach the "
                    f"language model to turn them into an answer."
                ),
                details=str(e),
                grounded=False,
                sources=[c.to_dict() for c in used_chunks],
            )

        return AgentResult(
            agent_name=self.name,
            summary=answer_text.strip().split("\n")[0][:200],
            details=answer_text.strip(),
            grounded=True,
            sources=[c.to_dict() for c in used_chunks],
        )
