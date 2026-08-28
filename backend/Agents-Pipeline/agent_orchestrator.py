"""
agent_orchestrator.py
=======================
PHASE 7/10/11/13 — The Agents Pipeline (full orchestration)

PHASE 10 — MULTI-AGENT COLLABORATION (the default mode):

    Farmer asks
        |
        v
    Planner Agent                    (planner_agent.py — decides WHO, and in what ORDER)
        |
        v
    Disease Agent  ---->  Weather Agent  ---->  Soil Agent  ---->  Fertilizer Agent
    (each agent runs AFTER the ones before it, and is handed every
     earlier agent's findings as `request.context["prior_findings"]` —
     see `_run_sequential_collaboration()` below and
     `agent_types.format_prior_findings`. This is what makes it a real
     collaboration: the Fertilizer Agent's advice can reference what
     the Soil Agent and Weather Agent already found for THIS question,
     not just the farmer's original wording.)
        |
        v
    Planner  (via Report Agent — see report_agent.py, and the note
              below on why the Planner's own "final answer" step is
              implemented there rather than back inside PlannerAgent)
        |
        v
    Final Answer — one consolidated, source-cited recommendation

The exact chain that runs (which agents, in which order) still comes
from the Planner's reasoning chain (`planner_agent.py`'s
`_REASONING_CHAINS` / LLM planning) — this file's job is purely
EXECUTION: running that chain either sequentially-with-shared-context
(Phase 10, default) or independently (Phase 7's original fan-out,
still available via `agent_config.COLLABORATION_MODE = "parallel"`),
then handing every result to the Report Agent for final synthesis.

Why the Report Agent, not a second `PlannerAgent.plan()` call, plays
the diagram's second "Planner" box: `PlannerAgent` (see
planner_agent.py) is deliberately NOT a `BaseAgent` — it produces a
*routing decision*, never an `AgentResult`, and never talks to the
farmer directly. The Report Agent is what actually performs the
manager's closing "combine everything the team found into one answer"
step, so it fills that role here. Its docstring covers this too.

This is the Phase 7 counterpart to `RAG-Pipeline/rag_pipeline.py`:
the single module the rest of the app should import. `main.py` (CLI)
and `api.py` (FastAPI) both just call
`AgentOrchestrator().handle(question)`.

PHASE 11 — CONVERSATION MEMORY (opt-in via `session_id`):

    Farmer (Day 1)                    Farmer (Day 2, same session_id)
        |                                       |
        v                                       v
    handle(q1, session_id="farmer-42")     handle(q2, session_id="farmer-42")
        |                                       |
        v                                       v
    1. load FarmerMemory("farmer-42")      1. load FarmerMemory("farmer-42")
       (empty — first turn)                    -> already knows crop, location,
                                                    last disease/fertilizer found,
                                                    recent weather, from Day 1
        |                                       |
        v                                       v
    2. merge memory's known facts into     2. same — so e.g. the Weather Agent
       `context` (crop/location/lat/lon/       already has a location to check
       field), render a memory prompt          without the farmer repeating it
       block for agents to read
        |                                       |
        v                                       v
    3. run the plan as normal (Planner ->  3. same
       specialists -> Report Agent)
        |                                       |
        v                                       v
    4. record_turn(): extract whatever     4. same — Day 2's new findings get
       NEW facts this turn revealed and       folded in on top of Day 1's, so
       persist them for NEXT time              Day 3 knows even more
        |                                       |
        v                                       v
    OrchestratedAnswer(..., session_id,    OrchestratedAnswer(..., session_id,
                       recalled_memory={})                     recalled_memory={...})

`session_id` is entirely optional — omitting it (the Phase 7-10
default) skips memory load/save altogether, so existing callers that
never pass one behave exactly as before. See conversation_memory.py
for the storage/extraction details.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import agent_config
from agent_types import AgentRequest, AgentResult, PlanDecision, PlanStep
from conversation_memory import ConversationMemoryStore, FarmerMemory
from explainability import Explanation, build_explanation
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
from image_agent import ImageAgent
from rag_bridge import make_shared_retriever, make_shared_llm, RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class OrchestratedAnswer:
    """Everything the frontend needs to render the full agentic answer."""
    question: str
    plan: PlanDecision
    agent_results: List[AgentResult]
    final_report: AgentResult
    # PHASE 10 — "sequential" (agents collaborated, each seeing earlier
    # agents' findings) or "parallel" (Phase 7 fan-out). Surfaced so the
    # frontend/CLI can show which mode actually produced this answer.
    collaboration_mode: str = "sequential"
    # PHASE 11 — which conversation this answer belongs to (None if the
    # caller didn't opt into memory for this request), and exactly what
    # facts were recalled from EARLIER turns and used for this one — so
    # the frontend/CLI can show the farmer what the AI already knew
    # instead of that being invisible.
    session_id: Optional[str] = None
    recalled_memory: Dict = field(default_factory=dict)
    # PHASE 13 — the Recommendation / Reason / Supporting documents /
    # Confidence / References structure built from `final_report` — see
    # explainability.py. None only for the empty-question short-circuit
    # below, where there's no report to explain.
    explanation: Optional[Explanation] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "plan": self.plan.to_dict(),
            "agent_results": [r.to_dict() for r in self.agent_results],
            "final_report": self.final_report.to_dict(),
            "collaboration_mode": self.collaboration_mode,
            "session_id": self.session_id,
            "recalled_memory": self.recalled_memory,
            "explanation": self.explanation.to_dict() if self.explanation else None,
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
            "image_agent": ImageAgent(llm=shared_llm),
            "general": GeneralAgent(rag_pipeline=RAGPipeline(retriever=shared_retriever, llm=shared_llm)),
        }
        self.report_agent = ReportAgent(llm=shared_llm)

        # PHASE 11 — one store shared across every request; cheap to
        # construct (just resolves a folder path), unlike the LLM/vector
        # DB connections above, so it isn't worth injecting separately.
        self.memory_store = ConversationMemoryStore()

    def handle(
        self,
        question: str,
        context: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> OrchestratedAnswer:
        question = (question or "").strip()
        explicit_context = dict(context or {})

        # PHASE 11 — Step 0: recall what we already know about this
        # farmer's conversation (no-op, and no memory read/write at all,
        # if the caller didn't pass a session_id — see module docstring).
        memory: Optional[FarmerMemory] = None
        recalled: Dict = {}
        if session_id and agent_config.MEMORY_ENABLED:
            memory = self.memory_store.get(session_id)
            recalled = memory.known_context()

        # Explicit context for THIS question always wins over older
        # memory (e.g. a farmer switching crops mid-conversation, or an
        # explicit location, should not get stuck on stale facts) — see
        # FarmerMemory.known_context()'s docstring.
        context = {**recalled, **explicit_context}
        if memory is not None:
            context["memory_summary"] = memory.to_prompt_block()

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
                collaboration_mode=agent_config.COLLABORATION_MODE,
                session_id=session_id,
                recalled_memory=recalled,
            )

        # Step 1 — Planner Agent decides WHO handles this question, and in
        # what ORDER (see planner_agent.py — this is now a reasoning
        # chain, not just a flat agent list). In Phase 10's default
        # sequential mode, this order is also the collaboration order.
        plan = self.planner.plan(question)
        plan = self._ensure_image_step(plan, context)
        logger.info(f"Planner chain ({plan.method}): {plan.reasoning}")
        logger.info(f"Planner selected agents to run: {plan.agents_to_run}")

        # Step 2 — Run each selected specialist.
        # Phase 9: if the Image Agent is among them, run it FIRST and
        # fold its plain-language photo description into every other
        # agent's context (see knowledge_agent.py's use of
        # context["image_description"]) — a photo of yellowing, spotted
        # leaves should be able to help Disease Agent even if the
        # farmer's own words never mentioned "yellowing" or "spots".
        ordered_agent_names = self._image_agent_first(plan.agents_to_run)

        if agent_config.COLLABORATION_MODE == "parallel":
            agent_results = self._run_parallel(question, ordered_agent_names, context)
        else:
            agent_results = self._run_sequential_collaboration(question, ordered_agent_names, context)

        if not agent_results:
            # Should only happen if the planner returned only unknown
            # names — fall back to the General Agent so the farmer
            # still gets an answer.
            agent_results.append(
                self.agent_registry["general"].execute(AgentRequest(query=question, context=context))
            )

        # Step 3 — Report Agent synthesizes everything into one answer —
        # this is the diagram's second "Planner" box (see this module's
        # docstring for why the Report Agent, not PlannerAgent itself,
        # implements it). The Planner's reasoning chain is passed through
        # too, so the final answer reflects *why* each specialist was
        # consulted and in what order, rather than re-guessing it.
        report_request = AgentRequest(
            query=question,
            context={**context, "agent_results": agent_results, "plan": plan},
        )
        final_report = self.report_agent.execute(report_request)

        # PHASE 13 — turn `final_report` into the Recommendation / Reason
        # / Supporting documents / Confidence / References structure a
        # farmer (or auditor) can actually check, instead of a single
        # block of prose. See explainability.py's module docstring for
        # why this is a parsing + formula step, not another LLM call.
        explanation = build_explanation(agent_results, final_report)

        # PHASE 11 — Step 4: fold whatever NEW facts this turn revealed
        # (explicit context hints, crop mentioned in the question text,
        # and grounded specialist findings) into this session's memory,
        # so the FARMER doesn't have to repeat any of it next time.
        if memory is not None:
            memory = self.memory_store.record_turn(
                session_id=session_id,
                question=question,
                context_hints=explicit_context,
                agent_results=agent_results,
                final_report=final_report,
            )

        return OrchestratedAnswer(
            question=question,
            plan=plan,
            agent_results=agent_results,
            final_report=final_report,
            collaboration_mode=agent_config.COLLABORATION_MODE,
            session_id=session_id,
            recalled_memory=recalled,
            explanation=explanation,
        )

    def _run_sequential_collaboration(
        self, question: str, ordered_agent_names: List[str], context: Dict
    ) -> List[AgentResult]:
        """
        PHASE 10 — default execution mode. Runs each planned agent ONE AT
        A TIME, in the Planner's chain order, and hands every agent after
        the first the accumulated findings of every agent that ran before
        it (`request.context["prior_findings"]`, consumed by
        `knowledge_agent.py` / `weather_agent.py` via
        `agent_types.format_prior_findings`).

        This is what turns "run several agents" into actual
        *collaboration*: for the canonical example (yellowing tomato
        leaves, should I water today?) the Weather Agent's forecast is
        available to the Soil Agent, and both the Weather Agent's and
        Soil Agent's findings are available to the Fertilizer Agent —
        each specialist reasons WITH its teammates' work, not blind to it.

        The Image Agent is handled the same way it always was (Phase 9):
        its photo description is folded into `context["image_description"]`
        rather than into `prior_findings`, since knowledge agents already
        read that key directly and it isn't "a specialist's finding" in
        the same sense (it's raw visual evidence, not a conclusion).

        One agent failing never stops the chain — `BaseAgent.execute()`
        already turns any exception into an honest `AgentResult.error`,
        so a broken Weather API still lets Soil/Fertilizer run (just
        without a weather finding to build on).
        """
        agent_results: List[AgentResult] = []
        enriched_context: Dict = dict(context)
        prior_findings: List[Dict] = []

        for agent_name in ordered_agent_names:
            agent = self.agent_registry.get(agent_name)
            if agent is None:
                logger.warning(f"Planner selected unknown agent '{agent_name}', skipping.")
                continue

            request_context = {**enriched_context, "prior_findings": list(prior_findings)}
            result = agent.execute(AgentRequest(query=question, context=request_context))
            agent_results.append(result)

            if agent_name == "image_agent":
                if result.data.get("description"):
                    enriched_context = {
                        **enriched_context,
                        "image_description": result.data["description"],
                    }
                # The Image Agent's output flows via image_description
                # (above), not prior_findings — see docstring.
                continue

            prior_findings.append({
                "agent_name": result.agent_name,
                "summary": result.summary,
                "details": result.details,
                "grounded": result.grounded,
            })

        return agent_results

    def _run_parallel(
        self, question: str, ordered_agent_names: List[str], context: Dict
    ) -> List[AgentResult]:
        """
        PHASE 7 (original) execution mode, kept for
        `agent_config.COLLABORATION_MODE == "parallel"`: every selected
        agent runs independently off the same base context — no agent
        sees another's findings, except that the Image Agent's photo
        description (if any) is still folded in for every agent (Phase 9
        behaviour, unchanged). Faster (agents don't need each other's
        output first) at the cost of no real cross-agent reasoning.
        """
        agent_results: List[AgentResult] = []
        enriched_context: Dict = dict(context)
        for agent_name in ordered_agent_names:
            agent = self.agent_registry.get(agent_name)
            if agent is None:
                logger.warning(f"Planner selected unknown agent '{agent_name}', skipping.")
                continue
            request = AgentRequest(query=question, context=enriched_context)
            result = agent.execute(request)
            agent_results.append(result)

            if agent_name == "image_agent" and result.data.get("description"):
                enriched_context = {**enriched_context, "image_description": result.data["description"]}

        return agent_results

    @staticmethod
    def _ensure_image_step(plan: PlanDecision, context: Dict) -> PlanDecision:
        """Phase 9 — if a photo was attached (context['image_base64']) but
        the Planner's keyword/LLM routing didn't select image_agent (the
        farmer's wording never has to say "here's a photo" for one to be
        attached), inject it into the plan directly. Keyword matching
        alone can't detect an attachment, so this fills that gap rather
        than asking the Planner to somehow infer it from text."""
        if not context.get("image_base64") or "image_agent" in plan.agents_to_run:
            return plan

        image_step = PlanStep(
            need="what the attached photo shows",
            agent="image_agent",
            reason="A photo was attached to this question.",
        )
        agents_to_run = ["image_agent", *plan.agents_to_run]
        agents_to_run = agents_to_run[: agent_config.PLANNER_MAX_AGENTS_PER_REQUEST]
        return PlanDecision(
            agents_to_run=agents_to_run,
            reasoning=f"Need what the attached photo shows → {plan.reasoning}",
            method=plan.method,
            steps=[image_step, *plan.steps],
        )

    @staticmethod
    def _image_agent_first(agent_names: List[str]) -> List[str]:
        """Runs image_agent before every other selected agent so its
        description can be folded into their context (see handle())."""
        if "image_agent" not in agent_names:
            return agent_names
        rest = [name for name in agent_names if name != "image_agent"]
        return ["image_agent", *rest]
