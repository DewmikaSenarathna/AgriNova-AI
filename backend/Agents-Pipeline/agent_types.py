"""
agent_types.py
==============
The shared "envelope" every agent speaks, so the Planner Agent and the
Report Agent can treat all eight agents identically without knowing
anything about how each one works internally.

    AgentRequest   ->  BaseAgent.run()  ->  AgentResult

Every agent in this pipeline (disease, weather, market, government,
soil, fertilizer, pest, report) takes an AgentRequest and returns an
AgentResult — that single contract is what makes them pluggable.

Phase 8 adds `PlanStep`: the Planner Agent no longer just names which
agents to run, it exposes the reasoning CHAIN behind that choice (see
planner_agent.py's module docstring for the full picture).
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


def format_prior_findings(prior_findings: List[Dict[str, Any]]) -> str:
    """
    PHASE 10 — turns the accumulated findings from EARLIER agents in a
    sequential collaboration chain into a short prompt block, so a
    later agent (e.g. Fertilizer Agent) can genuinely build on what an
    earlier one already found (e.g. the Soil Agent's soil-moisture
    finding, or the Disease Agent's diagnosis) instead of answering the
    farmer's question in isolation, blind to what its teammates just
    said.

    `prior_findings` is a list of small dicts — see
    `agent_orchestrator.py`'s `_run_sequential_collaboration` for how
    it's built — each with `agent_name`, `summary`, `details`,
    `grounded`. Returns "" when nothing has run yet (the first agent in
    a chain, or `agent_config.COLLABORATION_MODE == "parallel"`), so
    every agent's prompt stays unchanged from Phase 7/8/9 in that case.
    """
    if not prior_findings:
        return ""

    lines = [
        "FINDINGS FROM EARLIER SPECIALISTS IN THIS COLLABORATION (from the "
        "same farmer question, already investigated by teammates before you — "
        "build on these, don't just repeat them verbatim, and say so plainly "
        "if something here conflicts with what you find):"
    ]
    for finding in prior_findings:
        label = str(finding.get("agent_name", "agent")).replace("_", " ").title()
        grounded_tag = "grounded" if finding.get("grounded") else "not grounded"
        text = finding.get("details") or finding.get("summary") or "(no findings)"
        lines.append(f"- [{label}, {grounded_tag}] {text}")
    return "\n".join(lines) + "\n\n"


@dataclass
class PlanStep:
    """
    One link in the Planner Agent's reasoning chain — "the manager
    thinking out loud". A farmer asking "Should I apply fertilizer
    tomorrow?" only mentions fertilizer, but answering it well needs a
    small chain of information:

        Need weather -> Need fertilizer schedule -> Need crop stage
        -> Need rainfall -> Need recommendation

    Each PlanStep captures ONE of those needs: what's needed, which
    agent supplies it, and why. `agents_to_run` is the flattened list
    of agents this chain calls for; `steps` is the chain itself, kept
    around so the Planner's reasoning stays inspectable instead of
    collapsing into an opaque list of agent names.
    """
    need: str
    agent: str
    reason: str

    def to_dict(self) -> dict:
        return {"need": self.need, "agent": self.agent, "reason": self.reason}


@dataclass
class PlanDecision:
    """The Planner Agent's routing decision for one farmer question."""
    agents_to_run: List[str]
    reasoning: str
    method: str  # "keyword" or "llm"
    steps: List[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agents_to_run": self.agents_to_run,
            "reasoning": self.reasoning,
            "method": self.method,
            "steps": [s.to_dict() for s in self.steps],
        }
