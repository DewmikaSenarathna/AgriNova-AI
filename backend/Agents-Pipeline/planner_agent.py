"""
planner_agent.py
==================
PHASE 8 — The Planner Agent: the pipeline's "manager".

A farmer's question rarely maps to exactly one specialist in isolation.
Asked "Should I apply fertilizer tomorrow?", a good agronomist doesn't
just answer "use urea" — they think it through:

    Need weather            (will rain wash the fertilizer away?)
        |
        v
    Need fertilizer schedule (what type/dose/timing applies here?)
        |
        v
    Need crop stage          (young plants need different handling)
        |
        v
    Need rainfall             (specifically: is it dry long enough?)
        |
        v
    Need recommendation        (combine all of the above)

That chain — not just a flat "call these agents" list — is what the
Planner now produces. Each link is a `PlanStep` (see agent_types.py):
a need, the agent that supplies it, and why it's needed. The LAST step
is always "Need recommendation" -> `report_agent`, made explicit so the
reasoning chain reads the same way a human manager would explain it,
even though `agent_orchestrator.py` runs the Report Agent separately
(see there for why).

    Farmer's question
            |
            v
      Planner Agent  ---------------------------------------------+
            |                                                      |
            v                                                      |
    agent_config.PLANNER_MODE == "keyword" (default)                |
        -> deterministic keyword matching + a small table of         |
           "this need implies that need too" dependency rules         |
           (_REASONING_CHAINS below)                                   |
                                                                     |
    agent_config.PLANNER_MODE == "llm"                               |
        -> one structured-JSON call asking the LLM for the whole      |
           reasoning chain directly                                   |
                                                                     |
            |                                                      |
            v                                                      |
    PlanDecision(steps=[...], agents_to_run=[...], ...) <-----------+
            |
            v
    agent_orchestrator.py runs each unique agent named in the chain,
    in that order, then the Report Agent (see report_agent.py)

The Planner NEVER runs a specialist agent itself — it only decides
which ones should run, and in what reasoning order. That separation is
what keeps this class small and lets each specialist stay
independently testable.

PHASE 10 note: the chain ORDER produced here now matters even more
than it did in Phase 8. By default (`agent_config.COLLABORATION_MODE
== "sequential"`), `agent_orchestrator.py` runs `agents_to_run` one at
a time in exactly this order, handing each agent every earlier agent's
findings — so the chain this file builds isn't just a human-readable
explanation anymore, it's the actual collaboration order. See the
`disease_agent` / `weather_agent` entries in `_REASONING_CHAINS` below
for the canonical Phase 10 example: "My tomato plants are turning
yellow. Should I water them today?" -> Disease Agent -> Weather Agent
-> Soil Agent -> Fertilizer Agent -> Planner (via Report Agent) ->
Final Answer.
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
        "yellowing leaves", "turning yellow", "yellow", "mildew", "virus", "lesion",
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
        "humidity", "wind", "storm", "irrigation", "monsoon", "season",
        "water", "watering", "irrigate",
    ],
    "market_agent": [
        "price", "prices", "market", "sell", "selling", "buyer", "profit",
        "cost of", "market rate", "wholesale", "demand",
    ],
    "government_agent": [
        "government", "subsidy", "subsidies", "scheme", "grant", "loan",
        "policy", "ministry", "extension officer", "certificate", "license",
    ],
    # Phase 9 — matches when a farmer mentions attaching/sending a photo.
    # The orchestrator also auto-adds this agent whenever
    # context["image_base64"] is present, regardless of keyword match
    # (see agent_orchestrator.py), so a photo attached with no caption
    # still gets analyzed.
    "image_agent": [
        "photo", "picture", "image", "pic", "snapshot", "attached",
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
    "image_agent": "a description of what the attached photo shows",
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
    # PHASE 10 example chain — "My tomato plants are turning yellow.
    # Should I water them today?": Disease Agent -> Weather Agent ->
    # Soil Agent -> Fertilizer Agent -> (Planner synthesizes) Final
    # Answer. Yellowing has several possible root causes an agronomist
    # would check in this order: disease first, then whether recent/
    # upcoming weather explains it (or whether watering today even
    # makes sense), then soil moisture/drainage (over- or under-watered
    # soil also yellows leaves), then whether it's actually a nutrient
    # deficiency rather than disease at all. Each step's agent receives
    # every earlier step's findings when COLLABORATION_MODE ==
    # "sequential" (see agent_orchestrator.py / agent_types.py), so this
    # chain is genuinely collaborative, not just four independent
    # answers to the same question.
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
            "conditions, and recent/upcoming rain also affects whether watering today "
            "is even needed.",
        ),
        (
            "soil moisture and drainage",
            "soil_agent",
            "Waterlogged or poorly-drained soil can itself cause yellowing that mimics "
            "disease, and determines whether watering today would help or make things worse.",
        ),
        (
            "possible nutrient deficiency",
            "fertilizer_agent",
            "Yellowing leaves are a classic sign of nutrient deficiency (e.g. nitrogen), "
            "not just disease, so ruling that in or out matters before recommending treatment.",
        ),
    ],
    # A watering/irrigation question that DOESN'T mention disease still
    # needs the weather outlook and the soil's current moisture, in
    # that order, before "should I water today" can be answered well.
    "weather_agent": [
        (
            "the weather outlook",
            "weather_agent",
            "Recent and upcoming rain directly determines whether watering today is needed.",
        ),
        (
            "soil moisture and drainage",
            "soil_agent",
            "Whether the soil actually needs water — and how well it drains — matters as "
            "much as the forecast when deciding whether to irrigate today.",
        ),
    ],
    # Phase 9 — used when "image_agent" is itself the matched primary
    # domain (farmer's wording mentioned a photo/picture). When a photo
    # is attached but not mentioned in the wording, the orchestrator
    # injects this same step directly (see
    # agent_orchestrator.py._ensure_image_step) since keyword matching
    # alone can't detect "there IS an attachment".
    "image_agent": [
        (
            "what the attached photo shows",
            "image_agent",
            "A vision-model description of visible symptoms helps ground the "
            "diagnosis in what the farmer's photo actually shows, not just their words.",
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

    # -- Default: keyword routing + dependency chains ---------------------------
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

    # -- Optional: LLM-based planning --------------------------------------------
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
