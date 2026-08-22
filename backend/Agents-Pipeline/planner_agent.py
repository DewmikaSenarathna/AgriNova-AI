"""
planner_agent.py
"""

import json
import logging
import re
from typing import Dict, List, Optional

import agent_config
from agent_types import PlanDecision
from rag_bridge import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Every agent name here must match a BaseAgent.name used in
# agent_orchestrator.py's AGENT_REGISTRY.
_KEYWORD_MAP: Dict[str, List[str]] = {
    "disease_agent": [
        "disease", "infect", "fungus", "fungal", "blight", "wilt", "rot", "spot",
        "yellowing leaves", "mildew", "virus", "lesion",
    ],
    "pest_agent": [
        "pest", "pests", "insect", "insects", "bug", "bugs", "aphid", "caterpillar",
        "worm", "worms", "infestation", "mite", "mites", "locust", "termite",
    ],
    "fertilizer_agent": [
        "fertilizer", "fertiliser", "manure", "compost", "npk", "urea", "nutrient",
        "nutrient deficiency", "top dressing", "nitrogen", "phosphorus", "potassium",
    ],
    "soil_agent": [
        "soil", "ph level", "soil ph", "soil test", "land preparation", "loam",
        "clay soil", "sandy soil", "erosion", "soil health",
    ],
    "weather_agent": [
        "weather", "rain", "rainfall", "forecast", "temperature", "drought",
        "humidity", "wind", "storm", "irrigation timing", "monsoon", "season",
    ],
    "market_agent": [
        "price", "prices", "market", "sell", "selling", "buyer", "profit",
        "cost of", "market rate", "wholesale", "demand",
    ],
    "government_agent": [
        "government", "subsidy", "subsidies", "scheme", "grant", "loan",
        "policy", "ministry", "extension officer", "certificate", "license",
    ],
}


class PlannerAgent:
    """Not a BaseAgent subclass on purpose: the Planner produces a
    *routing decision* (PlanDecision), not an AgentResult — it never
    itself answers the farmer's question."""

    def __init__(self, llm: LLMClient = None):
        self.llm = llm  # only constructed lazily if PLANNER_MODE == "llm"

    def plan(self, question: str) -> PlanDecision:
        mode = agent_config.PLANNER_MODE
        if mode == "llm":
            decision = self._plan_with_llm(question)
            if decision is not None:
                return decision
            logger.warning("Planner Agent: LLM planning failed, falling back to keyword routing.")

        return self._plan_with_keywords(question)

    # -- Default: keyword / regex routing 
    def _plan_with_keywords(self, question: str) -> PlanDecision:
        question_lower = (question or "").lower()
        matches: List[str] = []
        matched_terms: Dict[str, List[str]] = {}

        for agent_name, keywords in _KEYWORD_MAP.items():
            # Word-boundary at the START of the keyword only (not the end),
            # then allow trailing word characters, so a keyword like "rain"
            # matches "rains"/"rainfall"/"raining" (plurals/stems the
            # keyword list can't enumerate) WITHOUT matching a keyword that
            # merely happens to appear mid-word (e.g. "rain" inside
            # "training" — no word boundary immediately before the "r").
            hits = [
                kw for kw in keywords
                if re.search(r"\b" + re.escape(kw) + r"\w*", question_lower)
            ]
            if hits:
                matches.append(agent_name)
                matched_terms[agent_name] = hits

        if not matches:
            return PlanDecision(
                agents_to_run=[agent_config.PLANNER_FALLBACK_AGENT],
                reasoning="No domain keywords matched; falling back to the General Agent.",
                method="keyword",
            )

        matches = matches[: agent_config.PLANNER_MAX_AGENTS_PER_REQUEST]
        reasoning = "; ".join(f"{a} <- {matched_terms[a]}" for a in matches)
        return PlanDecision(agents_to_run=matches, reasoning=reasoning, method="keyword")

    # -- Optional: LLM-based routing 
    def _plan_with_llm(self, question: str) -> Optional[PlanDecision]:
        if self.llm is None:
            self.llm = LLMClient()

        agent_names = list(_KEYWORD_MAP.keys()) + [agent_config.PLANNER_FALLBACK_AGENT]
        system_prompt = (
            "You are the Planner Agent for an agricultural assistant. Given a farmer's "
            "question, choose which specialist agent(s) should handle it. Respond with ONLY "
            "a JSON object, no other text, in exactly this shape:\n"
            '{"agents": ["agent_name", ...], "reasoning": "short reason"}\n\n'
            f"Valid agent names: {', '.join(agent_names)}.\n"
            "Choose 1-3 agents. If nothing else fits, choose only "
            f'"{agent_config.PLANNER_FALLBACK_AGENT}".'
        )
        try:
            raw = self.llm.generate(system_prompt, f"FARMER'S QUESTION: {question}")
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            agents = [a for a in parsed.get("agents", []) if a in agent_names]
            if not agents:
                agents = [agent_config.PLANNER_FALLBACK_AGENT]
            agents = agents[: agent_config.PLANNER_MAX_AGENTS_PER_REQUEST]
            return PlanDecision(
                agents_to_run=agents,
                reasoning=parsed.get("reasoning", ""),
                method="llm",
            )
        except (LLMError, json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Planner Agent LLM routing failed: {e}")
            return None
