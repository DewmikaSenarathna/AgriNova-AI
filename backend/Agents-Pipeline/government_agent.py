"""
government_agent.py
=====================
Single responsibility: government agriculture schemes, subsidies and
official guidelines.

PHASE 9 gives this agent TWO tools instead of one:

    1. VectorDBTool (inherited from KnowledgeAgent) -> semantic search
       over the chunked/embedded knowledge base, same as every other
       KnowledgeAgent subclass.
    2. GovernmentPDFSearchTool -> literal full-text search over the
       original processed PDF text for documents that look official
       (see agent_config.GOVERNMENT_DOCUMENT_TYPE_LABELS), independent
       of the vector database / embedding model.

Merging both means a farmer asking about a specific scheme gets both
"semantically similar" AND "exact keyword" evidence, and still gets an
answer from whichever tool is healthy if the other one is down or has
nothing indexed yet.
"""

import logging

from agent_types import AgentRequest, AgentResult
from knowledge_agent import KnowledgeAgent
from rag_bridge import LLMError
from tools.government_pdf_search_tool import GovernmentPDFSearchTool

logger = logging.getLogger(__name__)


class GovernmentAgent(KnowledgeAgent):
    name = "government_agent"
    description = "Surfaces government agriculture schemes, subsidies and official guidelines."
    domain_label = "government agriculture policy"
    query_hints = ["government scheme", "subsidy", "agriculture ministry guideline", "eligibility"]
    system_prompt = (
        "You are AgriNova AI's Government Agent, an assistant for official agriculture "
        "schemes, subsidies and guidelines.\n\n"
        "Using ONLY the numbered SOURCE excerpts and any PDF SEARCH RESULTS provided:\n"
        "1. Explain the relevant scheme / subsidy / guideline, citing sources as [Source N] "
        "and PDF matches by their file name.\n"
        "2. If eligibility criteria, deadlines, or application steps are in the sources, list "
        "them clearly. If they are NOT in the sources, say that explicitly rather than "
        "guessing — official requirements change and a wrong guess here can cost a farmer "
        "their application.\n"
        "3. Recommend confirming current details with the local agriculture office before "
        "acting, since official programs can be updated after this knowledge base was built.\n"
        "4. Keep the answer short, plain-language and actionable."
    )

    def __init__(self, *args, pdf_search_tool: GovernmentPDFSearchTool = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdf_search_tool = pdf_search_tool or GovernmentPDFSearchTool()

    def run(self, request: AgentRequest) -> AgentResult:
        question = (request.query or "").strip()
        if not question:
            return AgentResult(
                agent_name=self.name,
                summary=f"No question was provided to the {self.domain_label} agent.",
                grounded=False,
            )

        # Tool 1 — literal full-text search over official PDFs.
        pdf_result = self.pdf_search_tool.execute(query=question)

        # Tool 2 — semantic search over the shared vector database
        # (VectorDBTool, via KnowledgeAgent's own retrieval step).
        expanded_query = self._expand_query(question)
        vector_result = self.vector_tool.execute(query=expanded_query)

        if not pdf_result.ok and not vector_result.ok:
            reasons = "; ".join(filter(None, [pdf_result.error, vector_result.error]))
            return AgentResult(
                agent_name=self.name,
                summary="No official government sources were found for this question.",
                details=(
                    f"Neither the Government PDF Search nor the vector database had matching "
                    f"official documents ({reasons}). Please confirm current schemes/subsidies "
                    f"directly with the local agriculture office."
                ),
                grounded=False,
            )

        evidence_blocks = []
        sources = []
        if vector_result.ok:
            evidence_blocks.append(f"KNOWLEDGE-BASE SOURCES (semantic search):\n{vector_result.text}")
            sources.extend(vector_result.data.get("chunks", []))
        if pdf_result.ok:
            evidence_blocks.append(
                f"GOVERNMENT PDF SEARCH RESULTS (exact keyword matches in official documents):\n"
                f"{pdf_result.text}"
            )
            for hit in pdf_result.data.get("hits", []):
                sources.append({"source": "Government PDF Search", **hit})

        user_prompt = (
            f"{chr(10).join(evidence_blocks)}\n\n---\n\n"
            f"FARMER'S QUESTION ({self.domain_label}): {question}\n\n"
            f"Answer using only the evidence above, citing knowledge-base sources as "
            f"[Source N] and PDF Search results by file name."
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
                sources=sources,
            )

        return AgentResult(
            agent_name=self.name,
            summary=answer_text.strip().split("\n")[0][:200],
            details=answer_text.strip(),
            grounded=True,
            sources=sources,
        )
