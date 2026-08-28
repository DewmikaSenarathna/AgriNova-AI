"""
test_phase13_explainability.py
===============================
PHASE 13 — Unit tests for Explainable AI.

Two suites:

  TestExplainability — exercises `explainability.py` directly:
  section-splitting (including the no-headers fallback), the
  confidence formula's arithmetic, and reference de-duplication/
  numbering. No LLM / network required — pure, fast, offline.

  TestOrchestratorExplanationIntegration — confirms
  `AgentOrchestrator.handle()` actually attaches a Phase 13
  `Explanation` to `OrchestratedAnswer`, built from whatever the
  (fake, in-memory) Report Agent produced, mirroring the bare-
  orchestrator pattern from test_phase10_collaboration.py /
  test_phase11_memory.py.

Run with:
    cd backend/Agents-Pipeline
    python -m unittest tests.test_phase13_explainability -v
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_orchestrator import AgentOrchestrator  # noqa: E402
from agent_types import AgentRequest, AgentResult  # noqa: E402
from base_agent import BaseAgent  # noqa: E402
from conversation_memory import ConversationMemoryStore  # noqa: E402
from explainability import (  # noqa: E402
    build_explanation,
    build_references,
    compute_confidence,
    split_recommendation_and_reason,
)
from planner_agent import PlannerAgent  # noqa: E402


STRUCTURED_REPORT = """## Recommendation
Apply a nitrogen top-dressing this week and avoid overhead irrigation.

## Reason
Your tomato crop shows signs consistent with early blight [Disease Agent, Source 1]. \
Dry conditions are expected for the next 3 days [Weather Agent, Source 1], which is a \
good window to apply fertilizer without it washing away.

## Recommended next steps
- Apply urea top-dressing within 2 days.
- Avoid overhead irrigation this week.
- Monitor leaves for further spotting.
"""


class TestSplitRecommendationAndReason(unittest.TestCase):
    def test_parses_all_three_structured_sections(self):
        sections = split_recommendation_and_reason(STRUCTURED_REPORT)
        self.assertIn("nitrogen top-dressing", sections["recommendation"])
        self.assertIn("early blight", sections["reason"])
        self.assertIn("urea top-dressing", sections["next_steps"])

    def test_falls_back_to_first_paragraph_when_no_headers_present(self):
        text = "Use fertilizer X.\n\nThis is because your soil is low in nitrogen."
        sections = split_recommendation_and_reason(text)
        self.assertEqual(sections["recommendation"], "Use fertilizer X.")
        self.assertIn("low in nitrogen", sections["reason"])

    def test_handles_empty_text(self):
        sections = split_recommendation_and_reason("")
        self.assertEqual(sections["recommendation"], "")
        self.assertEqual(sections["reason"], "")

    def test_tolerates_an_unexpected_extra_heading(self):
        text = (
            "## Recommendation\nDo X.\n\n"
            "## Some Other Heading\nExtra detail here.\n\n"
            "## Reason\nBecause of Y.\n\n"
            "## Recommended next steps\n- Do X.\n"
        )
        sections = split_recommendation_and_reason(text)
        self.assertEqual(sections["recommendation"], "Do X.")
        # The unexpected heading's content should not be silently dropped.
        self.assertIn("Extra detail here.", sections["reason"])
        self.assertIn("Because of Y.", sections["reason"])


class TestComputeConfidence(unittest.TestCase):
    def test_all_grounded_with_similarity_yields_high_confidence(self):
        agent_results = [
            AgentResult(agent_name="disease_agent", summary="s", grounded=True),
            AgentResult(agent_name="weather_agent", summary="s", grounded=True),
        ]
        final_report = AgentResult(
            agent_name="report_agent", summary="s", grounded=True,
            sources=[{"similarity": 0.9}, {"similarity": 0.8}],
        )
        result = compute_confidence(agent_results, final_report)
        self.assertEqual(result["level"], "High")
        self.assertGreaterEqual(result["score"], 0.7)
        self.assertTrue(any("2 of 2" in f for f in result["factors"]))

    def test_no_grounded_findings_yields_low_confidence(self):
        agent_results = [
            AgentResult(agent_name="disease_agent", summary="s", grounded=False),
        ]
        final_report = AgentResult(agent_name="report_agent", summary="s", grounded=False)
        result = compute_confidence(agent_results, final_report)
        self.assertEqual(result["level"], "Low")
        self.assertLess(result["score"], 0.4)

    def test_errored_specialist_lowers_score_and_is_named_in_factors(self):
        agent_results = [
            AgentResult(agent_name="disease_agent", summary="s", grounded=True),
            AgentResult(agent_name="weather_agent", summary="", grounded=False, error="timeout"),
        ]
        final_report = AgentResult(agent_name="report_agent", summary="s", grounded=True)
        with_error = compute_confidence(agent_results, final_report)

        agent_results_no_error = [
            AgentResult(agent_name="disease_agent", summary="s", grounded=True),
        ]
        without_error = compute_confidence(agent_results_no_error, final_report)

        self.assertTrue(any("failed and were excluded" in f for f in with_error["factors"]))
        self.assertLessEqual(with_error["score"], without_error["score"])

    def test_score_never_goes_negative_or_above_one(self):
        agent_results = [
            AgentResult(agent_name=f"agent_{i}", summary="", grounded=False, error="down")
            for i in range(10)
        ]
        final_report = AgentResult(agent_name="report_agent", summary="s", grounded=False)
        result = compute_confidence(agent_results, final_report)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_missing_similarity_data_falls_back_to_grounded_ratio_not_zero(self):
        # A purely tool-backed answer (weather/market) has no vector
        # similarity scores at all — it shouldn't be penalized just for
        # having nothing to embed.
        agent_results = [AgentResult(agent_name="weather_agent", summary="s", grounded=True)]
        final_report = AgentResult(agent_name="report_agent", summary="s", grounded=True, sources=[])
        result = compute_confidence(agent_results, final_report)
        self.assertEqual(result["level"], "High")


class TestBuildReferences(unittest.TestCase):
    def test_numbers_sources_in_order(self):
        sources = [
            {"heading": "Early Blight Treatment", "similarity": 0.9, "agent": "disease_agent"},
            {"source": "Open-Meteo Weather API", "location": "Kurunegala", "agent": "weather_agent"},
        ]
        refs = build_references(sources)
        self.assertEqual([r["n"] for r in refs], [1, 2])
        self.assertEqual(refs[0]["label"], "Early Blight Treatment")
        self.assertEqual(refs[1]["label"], "Open-Meteo Weather API")

    def test_deduplicates_identical_chunk_sources(self):
        sources = [
            {"chunk_id": "abc123", "heading": "Soil pH Guide", "agent": "soil_agent"},
            {"chunk_id": "abc123", "heading": "Soil pH Guide", "agent": "fertilizer_agent"},
        ]
        refs = build_references(sources)
        self.assertEqual(len(refs), 1)

    def test_empty_sources_yields_empty_references(self):
        self.assertEqual(build_references([]), [])


class TestBuildExplanation(unittest.TestCase):
    def test_end_to_end_from_a_realistic_final_report(self):
        agent_results = [
            AgentResult(agent_name="disease_agent", summary="Likely early blight.", grounded=True),
            AgentResult(agent_name="weather_agent", summary="Dry for 3 days.", grounded=True),
        ]
        final_report = AgentResult(
            agent_name="report_agent",
            summary="Consolidated report.",
            details=STRUCTURED_REPORT,
            grounded=True,
            sources=[
                {"heading": "Early Blight Treatment", "similarity": 0.88, "agent": "disease_agent"},
            ],
        )
        explanation = build_explanation(agent_results, final_report)

        self.assertIn("nitrogen top-dressing", explanation.recommendation)
        self.assertIn("early blight", explanation.reason)
        self.assertIn("urea top-dressing", explanation.next_steps)
        self.assertEqual(explanation.supporting_documents, final_report.sources)
        self.assertEqual(explanation.confidence["level"], "High")
        self.assertEqual(len(explanation.references), 1)
        self.assertEqual(explanation.references[0]["n"], 1)

        as_dict = explanation.to_dict()
        self.assertEqual(set(as_dict.keys()), {
            "recommendation", "reason", "next_steps", "supporting_documents",
            "confidence", "references",
        })

    def test_falls_back_to_final_report_summary_when_text_is_unusable(self):
        final_report = AgentResult(
            agent_name="report_agent", summary="Fallback summary.", details="", grounded=False,
        )
        explanation = build_explanation([], final_report)
        self.assertEqual(explanation.recommendation, "Fallback summary.")


class RecordingAgent(BaseAgent):
    def __init__(self, name: str, summary: str, grounded: bool = True):
        self.name = name
        self.description = f"Fake {name} for testing."
        self._summary = summary
        self._grounded = grounded

    def run(self, request: AgentRequest) -> AgentResult:
        return AgentResult(
            agent_name=self.name, summary=self._summary,
            details=f"{self._summary} (details)", grounded=self._grounded,
        )


class StructuredFakeReportAgent(BaseAgent):
    """Fake Report Agent that returns the same Phase 13 structured
    text a real LLM (or the real fallback concatenation) would."""
    name = "report_agent"
    description = "Fake structured report agent for testing."

    def run(self, request: AgentRequest) -> AgentResult:
        agent_results = request.context.get("agent_results", [])
        return AgentResult(
            agent_name=self.name,
            summary="Consolidated report.",
            details=STRUCTURED_REPORT,
            grounded=any(r.grounded for r in agent_results),
            sources=[{"heading": "Early Blight Treatment", "similarity": 0.9, "agent": "disease_agent"}],
        )


def make_bare_orchestrator(agent_registry: Dict[str, BaseAgent], memory_dir: Path) -> AgentOrchestrator:
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.agent_registry = agent_registry
    orchestrator.planner = PlannerAgent()
    orchestrator.report_agent = StructuredFakeReportAgent()
    orchestrator.memory_store = ConversationMemoryStore(directory=memory_dir)
    return orchestrator


class TestOrchestratorExplanationIntegration(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="agrinova_xai_test_"))
        self._shutil = shutil

    def tearDown(self):
        self._shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_handle_attaches_a_populated_explanation(self):
        orchestrator = make_bare_orchestrator(
            {
                "disease_agent": RecordingAgent("disease_agent", "Likely early blight."),
                "weather_agent": RecordingAgent("weather_agent", "Dry for 3 days."),
                "soil_agent": RecordingAgent("soil_agent", "Soil moisture low."),
                "fertilizer_agent": RecordingAgent("fertilizer_agent", "Nitrogen recommended."),
            },
            memory_dir=self.tmp_dir,
        )

        result = orchestrator.handle("My tomato crop has yellowing leaves, what's wrong?")

        self.assertIsNotNone(result.explanation)
        self.assertIn("nitrogen top-dressing", result.explanation.recommendation)
        self.assertIn("High", result.explanation.confidence["level"])
        self.assertEqual(len(result.explanation.references), 1)

        as_dict = result.to_dict()
        self.assertIn("explanation", as_dict)
        self.assertIsNotNone(as_dict["explanation"])
        self.assertEqual(as_dict["explanation"]["recommendation"], result.explanation.recommendation)

    def test_empty_question_has_no_explanation(self):
        orchestrator = make_bare_orchestrator({}, memory_dir=self.tmp_dir)
        result = orchestrator.handle("")
        self.assertIsNone(result.explanation)
        self.assertIsNone(result.to_dict()["explanation"])


if __name__ == "__main__":
    unittest.main()
