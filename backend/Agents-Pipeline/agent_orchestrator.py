"""
agent_orchestrator.py
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent_types import AgentRequest, AgentResult, PlanDecision
from planner_agent import PlannerAgent
from base_agent import BaseAgent
from disease_agent import DiseaseAgent
from weather_agent import WeatherAgent
from market_agent import MarketAgent
from government_agent import GovernmentAgent
from soil_agent import SoilAgent
from fertilizer_agent import FertilizerAgent
from pest_agent import PestAgent
from report_agent import ReportAgent
from general_agent import GeneralAgent
from rag_bridge import make_shared_retriever, make_shared_llm, RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class OrchestratedAnswer:
    """Everything the frontend needs to render the full agentic answer."""
    question: str
    plan: PlanDecision
    agent_results: List[AgentResult]
    final_report: AgentResult

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "plan": self.plan.to_dict(),
            "agent_results": [r.to_dict() for r in self.agent_results],
            "final_report": self.final_report.to_dict(),
        }


class AgentOrchestrator:
    """
    Builds every agent ONCE (sharing one embedding model + ChromaDB
    connection + LLM client across the five knowledge agents, exactly
    like `RAGPipeline` does for Phase 6), then, per question:
      1. asks the Planner which agents apply,
      2. runs each of them (independently — one failure doesn't cancel
         the others, see BaseAgent.execute),
      3. hands every result to the Report Agent for final synthesis.
    """

    def __init__(self):
        shared_retriever = make_shared_retriever()
        shared_llm = make_shared_llm()

        self.planner = PlannerAgent(llm=shared_llm)

        self.agent_registry: Dict[str, BaseAgent] = {
            "disease_agent": DiseaseAgent(retriever=shared_retriever, llm=shared_llm),
            "pest_agent": PestAgent(retriever=shared_retriever, llm=shared_llm),
            "fertilizer_agent": FertilizerAgent(retriever=shared_retriever, llm=shared_llm),
            "soil_agent": SoilAgent(retriever=shared_retriever, llm=shared_llm),
            "government_agent": GovernmentAgent(retriever=shared_retriever, llm=shared_llm),
            "market_agent": MarketAgent(retriever=shared_retriever, llm=shared_llm),
            "weather_agent": WeatherAgent(llm=shared_llm),
            "general": GeneralAgent(rag_pipeline=RAGPipeline(retriever=shared_retriever, llm=shared_llm)),
        }
        self.report_agent = ReportAgent(llm=shared_llm)

    def handle(self, question: str, context: Optional[Dict] = None) -> OrchestratedAnswer:
        question = (question or "").strip()
        context = context or {}

        if not question:
            empty_report = AgentResult(
                agent_name="report_agent",
                summary="Please ask a question so I can help.",
                grounded=False,
            )
            return OrchestratedAnswer(
                question=question,
                plan=PlanDecision(agents_to_run=[], reasoning="Empty question.", method="none"),
                agent_results=[],
                final_report=empty_report,
            )

        # Step 1 — Planner Agent decides WHO handles this question, and WHY
        # (see planner_agent.py — this is now a reasoning chain, not just a
        # flat agent list).
        plan = self.planner.plan(question)
        logger.info(f"Planner chain ({plan.method}): {plan.reasoning}")
        logger.info(f"Planner selected agents to run: {plan.agents_to_run}")

        # Step 2 — Run each selected specialist independently.
        agent_results: List[AgentResult] = []
        for agent_name in plan.agents_to_run:
            agent = self.agent_registry.get(agent_name)
            if agent is None:
                logger.warning(f"Planner selected unknown agent '{agent_name}', skipping.")
                continue
            request = AgentRequest(query=question, context=context)
            agent_results.append(agent.execute(request))

        if not agent_results:
            # Should only happen if the planner returned only unknown
            # names — fall back to the General Agent so the farmer
            # still gets an answer.
            agent_results.append(
                self.agent_registry["general"].execute(AgentRequest(query=question, context=context))
            )

        # Step 3 — Report Agent synthesizes everything into one answer.
        # The Planner's reasoning chain is passed through too, so the
        # Report Agent can frame the answer the way the plan intended
        # (e.g. "the Planner flagged this as a fertilizer-timing decision
        # that also depends on the weather") rather than re-guessing why
        # each specialist was consulted.
        report_request = AgentRequest(
            query=question,
            context={**context, "agent_results": agent_results, "plan": plan},
        )
        final_report = self.report_agent.execute(report_request)

        return OrchestratedAnswer(
            question=question,
            plan=plan,
            agent_results=agent_results,
            final_report=final_report,
        )
