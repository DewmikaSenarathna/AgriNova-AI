"""
weather_agent.py
==================
Single responsibility: current + short-range weather, and what it means
for farming decisions (spraying, irrigation, harvest timing).

Unlike the five KnowledgeAgent subclasses, this agent's evidence comes
from a live external API call, not the knowledge base — Phase 9 puts
that call behind `tools.weather_tool.WeatherTool` (the "Weather API"
tool from the system architecture diagram), so this file's own job
shrinks down to: call the tool, then turn numbers into farmer-facing
guidance.

    farmer's question (may mention a place)
            |
            v
    Step 1: WeatherTool resolves a location -> (lat, lon)
            |
            v
    Step 2: WeatherTool fetches a short-range forecast [Open-Meteo]
            |
            v
    Step 3: turn raw numbers into a farmer-facing summary
            (LLM if available, deterministic text fallback if not)
"""

import logging
from typing import Optional

from base_agent import BaseAgent
from agent_types import AgentRequest, AgentResult, format_prior_findings
from rag_bridge import LLMClient, LLMError
from tools.weather_tool import WeatherTool

logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    name = "weather_agent"
    description = "Fetches a short-range weather forecast and explains what it means for farming."

    def __init__(self, llm: Optional[LLMClient] = None, tool: Optional[WeatherTool] = None):
        # The LLM is used only to phrase the summary; if it's unavailable
        # this agent still returns the raw forecast, unlike the
        # KnowledgeAgent subclasses where the LLM is essential.
        self.llm = llm or LLMClient()
        self.tool = tool or WeatherTool()

    def run(self, request: AgentRequest) -> AgentResult:
        question = request.query or ""
        result = self.tool.execute(query=question, context=request.context or {})

        if not result.ok:
            location_label = (request.context or {}).get("location", "your location")
            return AgentResult(
                agent_name=self.name,
                summary=f"Couldn't reach the weather service for {location_label}.",
                details=result.error or "",
                grounded=False,
                error=result.error,
            )

        # PHASE 10 — earlier agents' findings in this collaboration chain
        # (e.g. Disease Agent's diagnosis), if any; "" in parallel mode
        # or when this is the first agent to run.
        prior_findings_block = format_prior_findings(request.context.get("prior_findings", []))
        details = self._summarize_for_farming(question, result.text, prior_findings_block) or result.text

        current = result.data["current"]
        location_label = result.data["location"]
        return AgentResult(
            agent_name=self.name,
            summary=f"{location_label}: currently {current['condition']}, {current['temperature_c']}°C.",
            details=details,
            grounded=True,
            sources=[result.source],
            data=result.data,
        )

    def _summarize_for_farming(
        self, question: str, forecast_text: str, prior_findings_block: str = ""
    ) -> Optional[str]:
        """Best-effort: ask the LLM to translate raw numbers into farming
        guidance (spraying/irrigation/harvest timing). Falls back to the
        raw forecast text (returned by run()) if the LLM isn't reachable —
        the agent should never fail just because this phrasing step did.

        PHASE 10: `prior_findings_block` (see agent_types.format_prior_findings)
        carries what earlier agents in a sequential collaboration chain
        already found — e.g. a Disease Agent diagnosis — so the weather
        guidance can be specific ("humid conditions favor the fungal
        spread the Disease Agent flagged") instead of generic."""
        system_prompt = (
            "You are AgriNova AI's Weather Agent. You are given a real, current weather "
            "forecast and a farmer's question. Explain in plain language what the forecast "
            "means for farming decisions relevant to their question (e.g. whether it's a good "
            "window to spray, irrigate, plant, or harvest). Base your reasoning only on the "
            "forecast numbers given — do not invent temperatures or rainfall figures. If "
            "earlier specialist findings are provided, factor them into your guidance where "
            "relevant. Keep it short and actionable."
        )
        user_prompt = (
            f"{prior_findings_block}"
            f"FORECAST DATA:\n{forecast_text}\n\nFARMER'S QUESTION: {question}"
        )
        try:
            return self.llm.generate(system_prompt, user_prompt)
        except LLMError as e:
            logger.warning(f"Weather Agent: LLM summarization unavailable, using raw forecast: {e}")
            return None
