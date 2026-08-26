"""
knowledge_agent.py
===================
Shared base class for the five agents whose "one job" is: search the
SAME trusted knowledge base Phase 6 built, but through a domain lens —

    Disease Agent      -> crop disease diagnosis & treatment
    Fertilizer Agent   -> fertilizer choice, dosage, application timing
    Pest Agent         -> pest identification & management
    Soil Agent         -> soil health, pH, preparation
    Government Agent   -> government schemes, subsidies, guidelines

Rather than five near-identical copies of "embed -> search -> filter ->
prompt -> generate", each of those agents is a ~15-line subclass that
only supplies what makes it different: a domain label, a short list of
query-expansion hints, and a domain-specific system prompt. This class
does the actual retrieval + grounded generation, reusing Phase 6's
Retriever and LLMClient via rag_bridge.py rather than duplicating them.

This mirrors rag_pipeline.py's Step 1-4 flow (embed -> similarity
search -> top-k -> grounded prompt -> generate), just parameterized by
domain and reused across five agents instead of written once for a
single general pipeline.

PHASE 9 note: the actual "embed -> similarity search" call now goes
through `tools.vector_db_tool.VectorDBTool` (the "Vector Database" tool
from the Phase 9 architecture diagram) instead of calling `Retriever`
directly — same shared Retriever underneath, just reached through the
same tool interface every other external capability in this pipeline
now uses.

PHASE 10 note: when `agent_orchestrator.py` runs its agents in
sequential-collaboration mode, `request.context["prior_findings"]`
carries every earlier agent's result in this chain. `run()` renders it
(see `agent_types.format_prior_findings`) into the prompt ahead of the
retrieved SOURCES, so e.g. the Fertilizer Agent genuinely sees what the
Soil Agent and Weather Agent already found for this same question,
rather than answering blind to its teammates.

PHASE 11 note: when the caller passed a `session_id`,
`request.context["memory_summary"]` (built by
`conversation_memory.FarmerMemory.to_prompt_block`) carries what's
already known about this farmer from EARLIER turns — crop, location,
previous disease/fertilizer findings, recent weather. `run()` renders
this ahead of `prior_findings` so the LLM can say "since your tomato
crop had early blight recently..." instead of asking the farmer to
repeat themselves, or answering as if this were a first-ever question.
"""

import logging
from typing import List, Optional

from base_agent import BaseAgent
from agent_types import AgentRequest, AgentResult, format_prior_findings
from rag_bridge import Retriever, LLMClient, LLMError, rag_config
from tools.vector_db_tool import VectorDBTool

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

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLMClient] = None,
        vector_tool: Optional[VectorDBTool] = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()
        # Phase 9 — every knowledge agent reaches the vector database
        # through the shared VectorDBTool instead of calling Retriever
        # directly. `self.retriever` is kept as an attribute (some
        # callers, e.g. api.py's /health check, still read
        # `agent.retriever.store.count()`) but retrieval itself now
        # flows through the tool.
        self.vector_tool = vector_tool or VectorDBTool(retriever=self.retriever)

    # -- Step A — Domain-biased query expansion ------------------------------
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
        # Phase 9 — if an Image Agent already ran (e.g. the farmer
        # attached a crop photo), fold its plain-language description
        # into the retrieval query so a photo of, say, yellowing leaves
        # can surface disease-agent sources even if the farmer's own
        # text didn't mention "yellowing".
        image_description = (request.context or {}).get("image_description")
        if image_description:
            expanded_query = f"{expanded_query}. Visible in photo: {image_description}"

        tool_result = self.vector_tool.execute(query=expanded_query)

        if not tool_result.ok:
            return AgentResult(
                agent_name=self.name,
                summary=(
                    f"No sources specific to {self.domain_label} were found in the "
                    f"knowledge base for this question."
                ),
                details=(
                    f"I don't have grounded information on this in the knowledge base yet "
                    f"({tool_result.error}). Please confirm with a local agricultural "
                    f"extension officer before acting."
                ),
                grounded=False,
            )

        context_block = tool_result.text
        used_chunks = tool_result.data.get("chunks", [])

        # PHASE 10 — if earlier agents in this collaboration chain already
        # ran (agent_config.COLLABORATION_MODE == "sequential"), fold
        # their findings in ahead of the SOURCES block so this agent
        # reasons WITH its teammates instead of in isolation. Empty in
        # parallel mode or for the first agent in a chain.
        prior_findings_block = format_prior_findings(request.context.get("prior_findings", []))
        # PHASE 11 — "" for a brand-new session, or if the caller never
        # passed a session_id at all (see agent_orchestrator.handle).
        memory_block = request.context.get("memory_summary") or ""

        user_prompt = (
            f"{memory_block}"
            f"{prior_findings_block}"
            f"SOURCES:\n{context_block}\n\n---\n\n"
            f"FARMER'S QUESTION ({self.domain_label}): {question}\n\n"
        )
        if image_description:
            user_prompt += (
                f"THE FARMER ALSO ATTACHED A PHOTO. The Image Agent (a vision model, NOT a "
                f"knowledge-base source) described it as: {image_description}\n\n"
            )
        user_prompt += (
            "Answer using only the SOURCES above, citing them as [Source N]. "
            "If earlier specialist findings were given above, factor them into your "
            "answer where relevant instead of ignoring them. If facts already known about "
            "this farmer were given above, use them instead of asking the farmer to repeat "
            "themselves."
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
                sources=used_chunks,
            )

        return AgentResult(
            agent_name=self.name,
            summary=answer_text.strip().split("\n")[0][:200],
            details=answer_text.strip(),
            grounded=True,
            sources=used_chunks,
        )
