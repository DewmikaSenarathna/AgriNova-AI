"""
base_agent.py
=============
The contract every specialized agent (Disease, Weather, Market,
Government, Soil, Fertilizer, Pest, Report) implements — "single
responsibility" per the Phase 7 brief means each subclass only needs
to implement `run()`; routing, aggregation and error-boundaries are
handled once, here and in `agent_orchestrator.py`.
"""

import logging
from abc import ABC, abstractmethod

from agent_types import AgentRequest, AgentResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Every agent:
      - has exactly ONE job (name + description say what it is)
      - takes an AgentRequest, returns an AgentResult
      - never raises out of `execute()` — a broken external API or a
        down LLM should degrade to an honest AgentResult.error, not
        take down the whole multi-agent request
    """

    name: str = "base_agent"
    description: str = "Base agent — not used directly."

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResult:
        """Subclasses implement their one job here."""
        raise NotImplementedError

    def execute(self, request: AgentRequest) -> AgentResult:
        """
        Public entry point the orchestrator calls. Wraps `run()` so that
        ANY unexpected exception inside a single agent turns into a
        clearly-labelled AgentResult instead of crashing the farmer's
        entire request just because, say, the Weather Agent's API call
        failed in an unanticipated way.
        """
        try:
            return self.run(request)
        except Exception as e:  # last-resort safety net, by design
            logger.exception(f"{self.name} failed unexpectedly.")
            return AgentResult(
                agent_name=self.name,
                summary=f"{self.name.replace('_', ' ').title()} could not complete this request.",
                details="",
                grounded=False,
                sources=[],
                data={},
                error=str(e),
            )
