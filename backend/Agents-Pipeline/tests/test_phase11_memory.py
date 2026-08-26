"""
test_phase11_memory.py
=======================
PHASE 11 — Unit tests for conversation memory.

Two suites:

  TestConversationMemoryStore — exercises `conversation_memory.py`
  directly (crop extraction, round-trip save/load, prompt rendering,
  `record_turn`'s fact extraction). No LLM / network / vector DB
  required — pure, fast, offline.

  TestOrchestratorMemoryIntegration — replays the exact "Day 1 / Day 2"
  scenario from the Phase 11 brief through `AgentOrchestrator.handle()`
  with a FAKE agent registry + a real (keyword-mode) PlannerAgent +
  a fake Report Agent, so it never needs a reachable LLM or ChromaDB
  either. Confirms that:
    Day 1: "My tomato crop has yellowing leaves" -> memory learns
           crop=Tomato and records a disease_agent finding.
    Day 2: "Should I irrigate today?" -> the SAME session already has
           `crop` in its context (no need for the farmer to repeat it)
           and the memory prompt block mentions the Day 1 finding.

Run with:
    cd backend/Agents-Pipeline
    python -m unittest tests.test_phase11_memory -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_orchestrator import AgentOrchestrator  # noqa: E402
from agent_types import AgentRequest, AgentResult  # noqa: E402
from base_agent import BaseAgent  # noqa: E402
from conversation_memory import (  # noqa: E402
    ConversationMemoryStore,
    FarmerMemory,
    extract_crop_from_text,
)
from planner_agent import PlannerAgent  # noqa: E402


class RecordingAgent(BaseAgent):
    """Same small test double style as test_phase10_collaboration.py —
    remembers the context it was called with and returns a fixed,
    easily-identifiable AgentResult."""

    def __init__(self, name: str, summary: str, grounded: bool = True, data: Dict = None):
        self.name = name
        self.description = f"Fake {name} for testing."
        self._summary = summary
        self._grounded = grounded
        self._data = data or {}
        self.received_contexts: List[Dict] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.received_contexts.append(request.context or {})
        return AgentResult(
            agent_name=self.name,
            summary=self._summary,
            details=f"{self._summary} (details)",
            grounded=self._grounded,
            data=self._data,
        )


class FakeReportAgent(BaseAgent):
    name = "report_agent"
    description = "Fake report agent for testing — just echoes what it was given."

    def __init__(self):
        self.received_contexts: List[Dict] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.received_contexts.append(request.context or {})
        agent_results = request.context.get("agent_results", [])
        return AgentResult(
            agent_name=self.name,
            summary=f"Report combining {len(agent_results)} finding(s).",
            details="fake consolidated report",
            grounded=any(r.grounded for r in agent_results),
        )


def make_bare_orchestrator(
    agent_registry: Dict[str, BaseAgent], memory_dir: Path
) -> AgentOrchestrator:
    """Builds an AgentOrchestrator WITHOUT running __init__ (which needs
    a live LLM + vector DB): installs a fake agent_registry + fake
    Report Agent, a REAL (keyword-mode) PlannerAgent (pure Python, no
    LLM needed in that mode), and a ConversationMemoryStore pointed at
    a throwaway temp directory instead of the real output/memory/."""
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.agent_registry = agent_registry
    orchestrator.planner = PlannerAgent()
    orchestrator.report_agent = FakeReportAgent()
    orchestrator.memory_store = ConversationMemoryStore(directory=memory_dir)
    return orchestrator


class TestCropExtraction(unittest.TestCase):
    def test_extracts_known_crop_from_free_text(self):
        self.assertEqual(
            extract_crop_from_text("My tomato crop in Kurunegala has yellowing leaves"),
            "Tomato",
        )

    def test_normalizes_plural_and_alias_to_canonical_name(self):
        self.assertEqual(extract_crop_from_text("my chillies are wilting"), "Chili")
        self.assertEqual(extract_crop_from_text("the paddy field flooded"), "Rice")

    def test_returns_none_when_nothing_matches(self):
        self.assertIsNone(extract_crop_from_text("Should I irrigate today?"))


class TestConversationMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="agrinova_memory_test_"))
        self.store = ConversationMemoryStore(directory=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unknown_session_returns_fresh_empty_memory(self):
        memory = self.store.get("brand-new-session")
        self.assertTrue(memory.is_empty())
        self.assertEqual(memory.to_prompt_block(), "")
        self.assertEqual(memory.known_context(), {})

    def test_save_and_reload_round_trips(self):
        memory = FarmerMemory(session_id="s1", crop="Tomato", location="Kurunegala")
        self.store.save(memory)

        reloaded = self.store.get("s1")
        self.assertEqual(reloaded.crop, "Tomato")
        self.assertEqual(reloaded.location, "Kurunegala")

    def test_record_turn_extracts_crop_from_question_text(self):
        memory = self.store.record_turn(
            session_id="s1",
            question="My tomato crop in Kurunegala has yellowing leaves.",
            context_hints={},
            agent_results=[],
        )
        self.assertEqual(memory.crop, "Tomato")

    def test_record_turn_prefers_explicit_context_over_extraction(self):
        memory = self.store.record_turn(
            session_id="s1",
            question="my chillies look sick",  # would extract "Chili"
            context_hints={"crop": "Tomato"},   # explicit hint wins
            agent_results=[],
        )
        self.assertEqual(memory.crop, "Tomato")

    def test_record_turn_remembers_disease_and_fertilizer_findings(self):
        results = [
            AgentResult(agent_name="disease_agent", summary="Likely early blight.", grounded=True),
            AgentResult(agent_name="fertilizer_agent", summary="Apply nitrogen top-dressing.", grounded=True),
        ]
        memory = self.store.record_turn(
            session_id="s1", question="q", context_hints={}, agent_results=results
        )
        self.assertEqual(memory.last_disease["summary"], "Likely early blight.")
        self.assertEqual(memory.last_fertilizer["summary"], "Apply nitrogen top-dressing.")

    def test_record_turn_skips_findings_that_errored(self):
        results = [
            AgentResult(agent_name="disease_agent", summary="", grounded=False, error="LLM unreachable"),
        ]
        memory = self.store.record_turn(
            session_id="s1", question="q", context_hints={}, agent_results=results
        )
        self.assertIsNone(memory.last_disease)

    def test_record_turn_fills_location_gap_from_weather_agent_geocoding(self):
        weather_result = AgentResult(
            agent_name="weather_agent",
            summary="Currently sunny, 30C.",
            grounded=True,
            data={"location": "Kurunegala, North Western, Sri Lanka", "latitude": 7.48, "longitude": 80.36},
        )
        memory = self.store.record_turn(
            session_id="s1", question="what's the weather like?", context_hints={},
            agent_results=[weather_result],
        )
        self.assertEqual(memory.location, "Kurunegala, North Western, Sri Lanka")
        self.assertEqual(memory.latitude, 7.48)
        self.assertEqual(len(memory.weather_history), 1)

    def test_explicit_location_is_not_overwritten_by_weather_agent_default(self):
        weather_result = AgentResult(
            agent_name="weather_agent",
            summary="Currently sunny, 30C.",
            grounded=True,
            # Simulates the tool falling back to WEATHER_DEFAULT_LOCATION_NAME
            # because the farmer's own location couldn't be geocoded.
            data={"location": "Colombo, Sri Lanka", "latitude": 6.9271, "longitude": 79.8612},
        )
        memory = self.store.record_turn(
            session_id="s1", question="q", context_hints={"location": "Kurunegala"},
            agent_results=[weather_result],
        )
        self.assertEqual(memory.location, "Kurunegala")

    def test_weather_history_is_capped(self):
        import agent_config
        total_checks = agent_config.MEMORY_MAX_WEATHER_HISTORY + 3
        memory = None
        for i in range(total_checks):
            weather_result = AgentResult(
                agent_name="weather_agent", summary=f"Forecast #{i}", grounded=True,
                data={"location": "Kurunegala"},
            )
            memory = self.store.record_turn(
                session_id="s1", question="weather?", context_hints={},
                agent_results=[weather_result],
            )
        self.assertEqual(len(memory.weather_history), agent_config.MEMORY_MAX_WEATHER_HISTORY)
        # Oldest entries are dropped, newest kept.
        self.assertEqual(memory.weather_history[-1]["summary"], f"Forecast #{total_checks - 1}")

    def test_delete_forgets_a_session(self):
        self.store.save(FarmerMemory(session_id="s1", crop="Tomato"))
        self.assertTrue(self.store.delete("s1"))
        self.assertTrue(self.store.get("s1").is_empty())
        self.assertFalse(self.store.delete("s1"))  # already gone

    def test_prompt_block_mentions_known_facts_and_is_empty_when_nothing_known(self):
        empty_memory = FarmerMemory(session_id="s1")
        self.assertEqual(empty_memory.to_prompt_block(), "")

        memory = FarmerMemory(
            session_id="s1",
            crop="Tomato",
            location="Kurunegala",
            last_disease={"summary": "Likely early blight.", "date": "2026-08-20"},
        )
        block = memory.to_prompt_block()
        self.assertIn("Tomato", block)
        self.assertIn("Kurunegala", block)
        self.assertIn("early blight", block)

    def test_session_id_is_sanitized_for_the_filesystem(self):
        memory = FarmerMemory(session_id="farmer 42/../../etc")
        self.store.save(memory)
        # Should not have escaped the memory directory.
        files = list(self.tmp_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].is_relative_to(self.tmp_dir))


class TestOrchestratorMemoryIntegration(unittest.TestCase):
    """Replays the Phase 11 brief's canonical Day 1 / Day 2 scenario."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="agrinova_memory_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_day2_question_recalls_day1_crop_without_repeating_it(self):
        disease_agent = RecordingAgent("disease_agent", "Likely early blight.")
        weather_agent = RecordingAgent(
            "weather_agent", "Dry for the next 3 days.",
            data={"location": "Kurunegala, Sri Lanka", "latitude": 7.48, "longitude": 80.36},
        )
        soil_agent = RecordingAgent("soil_agent", "Soil moisture is currently low.")
        fertilizer_agent = RecordingAgent("fertilizer_agent", "Nitrogen top-dressing recommended.")

        orchestrator = make_bare_orchestrator(
            {
                "disease_agent": disease_agent,
                "weather_agent": weather_agent,
                "soil_agent": soil_agent,
                "fertilizer_agent": fertilizer_agent,
            },
            memory_dir=self.tmp_dir,
        )

        session_id = "farmer-42"

        # --- Day 1 ---------------------------------------------------------
        day1_result = orchestrator.handle(
            "My tomato crop has yellowing leaves, what disease could this be?",
            session_id=session_id,
        )
        # Nothing was known yet on Day 1 (first-ever turn for this session).
        self.assertEqual(day1_result.recalled_memory, {})
        self.assertEqual(day1_result.session_id, session_id)

        # --- Day 2 ---------------------------------------------------------
        day2_result = orchestrator.handle("Should I irrigate today?", session_id=session_id)

        # The farmer never said "tomato" on Day 2 — it should be recalled.
        self.assertEqual(day2_result.recalled_memory.get("crop"), "Tomato")
        self.assertEqual(
            day2_result.recalled_memory.get("location"), "Kurunegala, Sri Lanka"
        )

        # Day 2's plan matched weather_agent + soil_agent (via the
        # "irrigate" keyword's reasoning chain) — both should have
        # received the recalled crop via context, with no farmer input.
        weather_ctx = weather_agent.received_contexts[-1]
        self.assertEqual(weather_ctx["crop"], "Tomato")
        self.assertIn("memory_summary", weather_ctx)
        self.assertIn("Tomato", weather_ctx["memory_summary"])
        self.assertIn("early blight", weather_ctx["memory_summary"])

        soil_ctx = soil_agent.received_contexts[-1]
        self.assertEqual(soil_ctx["crop"], "Tomato")

    def test_different_sessions_do_not_share_memory(self):
        disease_agent = RecordingAgent("disease_agent", "Likely early blight.")
        general_agent = RecordingAgent("general", "General farming guidance.")
        orchestrator = make_bare_orchestrator(
            {"disease_agent": disease_agent, "general": general_agent}, memory_dir=self.tmp_dir
        )

        orchestrator.handle("My tomato crop has yellowing leaves.", session_id="farmer-A")
        result_b = orchestrator.handle("My rice paddy field looks unhealthy.", session_id="farmer-B")

        # farmer-B's first turn should recall nothing from farmer-A.
        self.assertEqual(result_b.recalled_memory, {})

        memory_a = orchestrator.memory_store.get("farmer-A")
        memory_b = orchestrator.memory_store.get("farmer-B")
        self.assertEqual(memory_a.crop, "Tomato")
        self.assertEqual(memory_b.crop, "Rice")

    def test_no_session_id_means_no_memory_read_or_write(self):
        disease_agent = RecordingAgent("disease_agent", "Likely early blight.")
        orchestrator = make_bare_orchestrator({"disease_agent": disease_agent}, memory_dir=self.tmp_dir)

        result = orchestrator.handle("My tomato crop has yellowing leaves.")  # no session_id
        self.assertIsNone(result.session_id)
        self.assertEqual(result.recalled_memory, {})
        # Nothing persisted anywhere for an unnamed session.
        self.assertEqual(orchestrator.memory_store.list_sessions(), [])

        # And an agent's context has no memory_summary key at all in
        # that case — Phase 7-10 behaviour is byte-for-byte preserved.
        self.assertNotIn("memory_summary", disease_agent.received_contexts[-1])

    def test_explicit_context_overrides_recalled_memory_for_this_turn(self):
        disease_agent = RecordingAgent("disease_agent", "Likely early blight.")
        orchestrator = make_bare_orchestrator({"disease_agent": disease_agent}, memory_dir=self.tmp_dir)
        session_id = "farmer-42"

        orchestrator.handle("My tomato crop has yellowing leaves.", session_id=session_id)
        # Farmer explicitly plants a different crop this season.
        orchestrator.handle(
            "My rice field has yellowing leaves too.",
            context={"crop": "Rice"},
            session_id=session_id,
        )

        memory = orchestrator.memory_store.get(session_id)
        self.assertEqual(memory.crop, "Rice")


if __name__ == "__main__":
    unittest.main()
