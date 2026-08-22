"""
agent_types.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentRequest:
    """
    What the Planner Agent hands to a specialized agent.

    query   -> the farmer's original (or lightly rephrased) question
    context -> optional structured hints the Planner already knows,
               e.g. {"crop": "tomato", "location": "Kurunegala",
               "latitude": 7.48, "longitude": 80.36}. Every agent is
               free to ignore keys it doesn't understand.
    """
    query: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """
    What every specialized agent hands back to the Planner / Report Agent.

    agent_name -> which agent produced this (matches BaseAgent.name)
    summary    -> one short sentence, safe to show a farmer directly
                  even if the full report is never read
    details    -> the fuller explanation / recommendation text
    grounded   -> True if `details` is backed by retrieved knowledge-base
                  sources or a real external API call, False if it's a
                  best-effort / generic fallback (mirrors RAGAnswer.grounded
                  from Phase 6, so the whole system is consistently honest
                  about what's evidence-backed vs. not)
    sources    -> citable evidence, if any (knowledge-base chunks, or a
                  description of the external API that was called)
    data       -> structured, machine-usable output (e.g. weather
                  numbers, market prices) that the frontend or the
                  Report Agent can render without re-parsing prose
    error      -> set (and everything else left safe/empty) when the
                  agent could not do its job at all, e.g. a required
                  external service was unreachable
    """
    agent_name: str
    summary: str
    details: str = ""
    grounded: bool = False
    sources: List[Dict] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "summary": self.summary,
            "details": self.details,
            "grounded": self.grounded,
            "sources": self.sources,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class PlanDecision:
    """The Planner Agent's routing decision for one farmer question."""
    agents_to_run: List[str]
    reasoning: str
    method: str  # "keyword" or "llm"

    def to_dict(self) -> dict:
        return {
            "agents_to_run": self.agents_to_run,
            "reasoning": self.reasoning,
            "method": self.method,
        }
