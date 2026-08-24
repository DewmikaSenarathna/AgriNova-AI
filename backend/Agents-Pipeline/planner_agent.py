"""
planner_agent.py
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import agent_config
from agent_types import PlanDecision, PlanStep
from rag_bridge import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Every agent name here must match a BaseAgent.name registered in
# agent_orchestrator.py's agent_registry.
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

# The single generic need used for a domain that was matched directly
# by keyword but doesn't have a richer chain below.
_GENERIC_NEED: Dict[str, str] = {
    "disease_agent": "disease diagnosis and treatment guidance",
    "pest_agent": "pest identification and management guidance",
    "fertilizer_agent": "fertilizer type, dosage and timing guidance",
    "soil_agent": "soil health / preparation guidance",
    "weather_agent": "the current weather / forecast",
    "market_agent": "current market price guidance",
    "government_agent": "government scheme / subsidy information",
}

# THE MANAGER'S PLAYBOOK — when a farmer's question is clearly ABOUT
# one of these domains, answering it well needs more than that one
# domain alone. Each entry is the full ordered reasoning chain for
# that primary domain: (need_phrase, agent_name, reason). This is
# exactly the "planner thinks" chain from the Phase 8 example,
# generalized to the other domains where the same kind of dependency
# applies (spraying/treatment timing also depends on weather).
_REASONING_CHAINS: Dict[str, List[Tuple[str, str, str]]] = {
    "fertilizer_agent": [
        (
            "the weather outlook",
            "weather_agent",
            "Rain shortly after application can wash fertilizer away before the "
            "crop absorbs it, so timing depends on the forecast.",
        ),
        (
            "the fertilizer type, dosage and timing",
            "fertilizer_agent",
            "This is the farmer's core question — which fertilizer, how much, and when.",
        ),
        (
            "the crop's growth stage",
            "soil_agent",
            "Fertilizer needs (and how the soil holds nutrients) change with the "
            "crop's growth stage, so this is needed before finalizing a dosage.",
        ),
        (
            "rainfall in the coming days",
            "weather_agent",
            "Confirms specifically whether conditions stay dry long enough after "
            "application for the fertilizer to take effect.",
        ),
    ],
    "pest_agent": [
        (
            "pest identification",
            "pest_agent",
            "The farmer's core question — what pest this is and how to manage it.",
        ),
        (
            "weather conditions for spraying",
            "weather_agent",
            "Wind and rain affect whether spraying now is effective and safe to apply.",
        ),
    ],
    "disease_agent": [
        (
            "disease diagnosis",
            "disease_agent",
            "The farmer's core question — what disease this matches and how to treat it.",
        ),
        (
            "humidity and rainfall conditions",
            "weather_agent",
            "Many fungal and bacterial crop diseases spread faster in humid, wet "
            "conditions, which affects how urgently to act.",
        ),
    ],
}

_RECOMMENDATION_STEP = PlanStep(
    need="a final recommendation",
    agent="report_agent",
    reason="Combines every specialist finding above into one clear answer for the farmer.",
)


class PlannerAgent:
    """Not a BaseAgent subclass on purpose: the Planner produces a
    *routing decision* (PlanDecision) — a reasoning chain of PlanSteps
    plus the flattened agent list — not an AgentResult. It never
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

    # -- Default: keyword routing + dependency chains 
    def _plan_with_keywords(self, question: str) -> PlanDecision:
        question_lower = (question or "").lower()
        primary_matches: List[str] = []
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
                primary_matches.append(agent_name)
                matched_terms[agent_name] = hits

        if not primary_matches:
            fallback_step = PlanStep(
                need="general farming guidance",
                agent=agent_config.PLANNER_FALLBACK_AGENT,
                reason="No domain keywords matched this question.",
            )
            steps = [fallback_step, _RECOMMENDATION_STEP]
            return PlanDecision(
                agents_to_run=[agent_config.PLANNER_FALLBACK_AGENT],
                reasoning=self._reasoning_from_steps(steps),
                method="keyword",
                steps=steps,
            )

        # Step A — Build the reasoning chain: a matched domain with its
        # own playbook contributes its FULL chain (including implied
        # needs the farmer didn't mention); a matched domain without one
        # contributes a single generic step.
        steps: List[PlanStep] = []
        seen_step_keys = set()  # (need, agent) pairs, to avoid literal duplicates

        for agent_name in primary_matches:
            chain = _REASONING_CHAINS.get(agent_name)
            if chain:
                for need, chained_agent, reason in chain:
                    key = (need, chained_agent)
                    if key in seen_step_keys:
                        continue
                    seen_step_keys.add(key)
                    steps.append(PlanStep(need=need, agent=chained_agent, reason=reason))
            else:
                need = _GENERIC_NEED.get(agent_name, agent_name.replace("_", " "))
                key = (need, agent_name)
                if key not in seen_step_keys:
                    seen_step_keys.add(key)
                    steps.append(PlanStep(
                        need=need,
                        agent=agent_name,
                        reason=f"Matched keyword(s) {matched_terms[agent_name]} in the question.",
                    ))

        steps.append(_RECOMMENDATION_STEP)

        # Step B — Flatten to a unique, ordered agent list for the
        # orchestrator to actually execute (report_agent is handled
        # separately by the orchestrator, so it's excluded here).
        agents_to_run: List[str] = []
        for step in steps:
            if step.agent != "report_agent" and step.agent not in agents_to_run:
                agents_to_run.append(step.agent)
        agents_to_run = agents_to_run[: agent_config.PLANNER_MAX_AGENTS_PER_REQUEST]

        return PlanDecision(
            agents_to_run=agents_to_run,
            reasoning=self._reasoning_from_steps(steps),
            method="keyword",
            steps=steps,
        )

    @staticmethod
    def _reasoning_from_steps(steps: List[PlanStep]) -> str:
        """Human-readable one-liner mirroring the Phase 8 example's
        "Need weather -> Need fertilizer schedule -> ..." trace."""
        return " → ".join(f"Need {s.need}" for s in steps)

    # -- Optional: LLM-based planning 
    def _plan_with_llm(self, question: str) -> Optional[PlanDecision]:
        if self.llm is None:
            self.llm = LLMClient()

        agent_names = list(_KEYWORD_MAP.keys()) + [agent_config.PLANNER_FALLBACK_AGENT]
        system_prompt = (
            "You are the Planner Agent for an agricultural assistant — the pipeline's "
            "manager. Given a farmer's question, think it through step by step like an "
            "agronomist would, then respond with ONLY a JSON object (no other text) in "
            "exactly this shape:\n"
            '{"steps": [{"need": "short phrase, e.g. \'the weather outlook\'", '
            '"agent": "agent_name", "reason": "one sentence"}], "reasoning": "short overall summary"}\n\n'
            f"Valid agent names: {', '.join(agent_names)}.\n"
            "Include a step for every piece of information genuinely needed to answer "
            "well — including ones the farmer didn't explicitly mention (e.g. a "
            "fertilizer-timing question also needs the weather outlook, because rain can "
            "wash fertilizer away). Use 1-5 steps. If nothing specific fits, use a single "
            f'step with agent "{agent_config.PLANNER_FALLBACK_AGENT}". '
            'Do NOT include a step for "report_agent" — that is added automatically.'
        )
        try:
            raw = self.llm.generate(system_prompt, f"FARMER'S QUESTION: {question}")
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(raw)

            steps: List[PlanStep] = []
            for item in parsed.get("steps", []):
                agent = item.get("agent")
                need = item.get("need")
                if agent not in agent_names or not need:
                    continue
                steps.append(PlanStep(
                    need=str(need),
                    agent=str(agent),
                    reason=str(item.get("reason", "")),
                ))

            if not steps:
                steps = [PlanStep(
                    need="general farming guidance",
                    agent=agent_config.PLANNER_FALLBACK_AGENT,
                    reason="LLM planner returned no usable steps.",
                )]
            steps.append(_RECOMMENDATION_STEP)

            agents_to_run: List[str] = []
            for step in steps:
                if step.agent != "report_agent" and step.agent not in agents_to_run:
                    agents_to_run.append(step.agent)
            agents_to_run = agents_to_run[: agent_config.PLANNER_MAX_AGENTS_PER_REQUEST]

            reasoning = parsed.get("reasoning") or self._reasoning_from_steps(steps)
            return PlanDecision(agents_to_run=agents_to_run, reasoning=reasoning, method="llm", steps=steps)
        except (LLMError, json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Planner Agent LLM routing failed: {e}")
            return None
