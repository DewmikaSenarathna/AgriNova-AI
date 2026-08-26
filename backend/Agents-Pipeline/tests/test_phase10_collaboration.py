"""
test_phase10_collaboration.py
==============================
PHASE 10 — Unit tests for the multi-agent collaboration execution path
in `agent_orchestrator.py`.

These tests deliberately do NOT construct a real `AgentOrchestrator()`
(that requires a reachable LLM + a populated ChromaDB, see
`rag_bridge.py`) — they build a bare instance via `__new__` and hand it
a small `agent_registry` of fake, in-memory `BaseAgent`s instead. That
keeps these tests fast, offline, and focused on exactly the thing
Phase 10 added: does `_run_sequential_collaboration()` actually hand
each agent every EARLIER agent's findings, in the right order, and
keep the Image Agent's description flowing through
`context["image_description"]` rather than `prior_findings`.

Run with:
    cd backend/Agents-Pipeline
    python -m unittest tests.test_phase10_collaboration -v
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_orchestrator import AgentOrchestrator  # noqa: E402
from agent_types import AgentRequest, AgentResult  # noqa: E402
from base_agent import BaseAgent  # noqa: E402


class RecordingAgent(BaseAgent):
    """A fake specialist that just remembers the context it was called
    with, and returns a fixed, easily-identifiable AgentResult."""

    def __init__(self, name: str, summary: str, grounded: bool = True):
        self.name = name
        self.description = f"Fake {name} for testing."
        self._summary = summary
        self._grounded = grounded
        self.received_contexts: List[Dict] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.received_contexts.append(request.context or {})
        return AgentResult(
            agent_name=self.name,
            summary=self._summary,
            details=f"{self._summary} (details)",
            grounded=self._grounded,
        )


class ImageRecordingAgent(RecordingAgent):
    """Fake Image Agent — returns data={"description": ...} the same
    shape agent_orchestrator.py expects from the real image_agent.py."""

    def __init__(self):
        super().__init__("image_agent", "A photo of yellowing leaves.", grounded=False)

    def run(self, request: AgentRequest) -> AgentResult:
        self.received_contexts.append(request.context or {})
        return AgentResult(
            agent_name=self.name,
            summary=self._summary,
            details=self._summary,
            grounded=False,
            data={"description": "yellowing, slightly wilted leaves"},
        )


def make_bare_orchestrator(agent_registry: Dict[str, BaseAgent]) -> AgentOrchestrator:
    """Builds an AgentOrchestrator WITHOUT running __init__ (which needs
    a live LLM + vector DB) and installs a fake agent_registry instead."""
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.agent_registry = agent_registry
    return orchestrator


class TestSequentialCollaboration(unittest.TestCase):
    def test_each_agent_sees_every_earlier_agents_findings(self):
        disease = RecordingAgent("disease_agent", "Likely early blight.")
        weather = RecordingAgent("weather_agent", "Dry for the next 3 days.")
        soil = RecordingAgent("soil_agent", "Soil moisture is currently low.")
        fertilizer = RecordingAgent("fertilizer_agent", "Nitrogen top-dressing recommended.")

        orchestrator = make_bare_orchestrator({
            "disease_agent": disease,
            "weather_agent": weather,
            "soil_agent": soil,
            "fertilizer_agent": fertilizer,
        })

        ordered = ["disease_agent", "weather_agent", "soil_agent", "fertilizer_agent"]
        results = orchestrator._run_sequential_collaboration(
            "My tomato plants are turning yellow. Should I water them today?",
            ordered,
            context={},
        )

        # All four ran, in order.
        self.assertEqual([r.agent_name for r in results], ordered)

        # First agent in the chain sees no prior findings.
        self.assertEqual(disease.received_contexts[0]["prior_findings"], [])

        # Weather Agent (2nd) sees exactly Disease Agent's finding.
        weather_prior = weather.received_contexts[0]["prior_findings"]
        self.assertEqual(len(weather_prior), 1)
        self.assertEqual(weather_prior[0]["agent_name"], "disease_agent")
        self.assertEqual(weather_prior[0]["summary"], "Likely early blight.")

        # Soil Agent (3rd) sees Disease + Weather, in that order.
        soil_prior = soil.received_contexts[0]["prior_findings"]
        self.assertEqual([f["agent_name"] for f in soil_prior], ["disease_agent", "weather_agent"])

        # Fertilizer Agent (last) sees all three earlier findings.
        fert_prior = fertilizer.received_contexts[0]["prior_findings"]
        self.assertEqual(
            [f["agent_name"] for f in fert_prior],
            ["disease_agent", "weather_agent", "soil_agent"],
        )

    def test_image_agent_feeds_image_description_not_prior_findings(self):
        image = ImageRecordingAgent()
        disease = RecordingAgent("disease_agent", "Likely nutrient deficiency, not disease.")

        orchestrator = make_bare_orchestrator({
            "image_agent": image,
            "disease_agent": disease,
        })

        orchestrator._run_sequential_collaboration(
            "What's wrong with this plant?",
            ["image_agent", "disease_agent"],
            context={"image_base64": "fake-base64-data"},
        )

        # Disease Agent (running after Image Agent) should see the photo
        # description via context["image_description"]...
        disease_ctx = disease.received_contexts[0]
        self.assertEqual(disease_ctx["image_description"], "yellowing, slightly wilted leaves")
        # ...but the Image Agent itself should NOT appear in prior_findings
        # (see agent_orchestrator.py docstring for why).
        self.assertEqual(disease_ctx["prior_findings"], [])

    def test_unknown_agent_name_is_skipped_without_crashing(self):
        disease = RecordingAgent("disease_agent", "Likely early blight.")
        orchestrator = make_bare_orchestrator({"disease_agent": disease})

        results = orchestrator._run_sequential_collaboration(
            "question", ["disease_agent", "not_a_real_agent"], context={}
        )
        self.assertEqual([r.agent_name for r in results], ["disease_agent"])


class TestParallelModeUnchanged(unittest.TestCase):
    def test_parallel_agents_do_not_see_each_others_findings(self):
        disease = RecordingAgent("disease_agent", "Likely early blight.")
        weather = RecordingAgent("weather_agent", "Dry for the next 3 days.")

        orchestrator = make_bare_orchestrator({
            "disease_agent": disease,
            "weather_agent": weather,
        })

        orchestrator._run_parallel(
            "question", ["disease_agent", "weather_agent"], context={"crop": "tomato"}
        )

        # Phase 7 behaviour preserved: no prior_findings key injected at all.
        self.assertNotIn("prior_findings", disease.received_contexts[0])
        self.assertNotIn("prior_findings", weather.received_contexts[0])
        # Base context (e.g. crop hint) still passed through though.
        self.assertEqual(weather.received_contexts[0]["crop"], "tomato")


if __name__ == "__main__":
    unittest.main()
