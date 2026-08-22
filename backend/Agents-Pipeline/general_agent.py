"""
general_agent.py
"""

from typing import Optional

from base_agent import BaseAgent
from agent_types import AgentRequest, AgentResult
from rag_bridge import RAGPipeline


class GeneralAgent(BaseAgent):
    name = "general_agent"
    description = "Answers general farming questions using the whole knowledge base (Phase 6 RAG)."

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RAGPipeline()

    def run(self, request: AgentRequest) -> AgentResult:
        rag_answer = self.rag_pipeline.answer(request.query)
        return AgentResult(
            agent_name=self.name,
            summary=rag_answer.answer.strip().split("\n")[0][:200],
            details=rag_answer.answer,
            grounded=rag_answer.grounded,
            sources=rag_answer.sources,
        )
