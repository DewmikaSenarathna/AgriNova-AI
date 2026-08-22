"""
report_agent.py
"""

import logging
from typing import List, Optional

from base_agent import BaseAgent
from agent_types import AgentRequest, AgentResult
from rag_bridge import LLMClient, LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are AgriNova AI's Report Agent. You are given a farmer's original \
question and the findings of several specialist agents that already investigated it \
(disease, weather, market, government, soil, fertilizer, and/or pest specialists).

Your job:
1. Write ONE clear, well-organized report that directly answers the farmer's question, \
combining the specialists' findings — do not just concatenate them.
2. Use short headings or a short list per topic covered, in plain language a farmer can act on.
3. Preserve each specialist's citations exactly as given to you (e.g. [Disease Agent, Source 2]) \
so the farmer can still tell which finding came from where.
4. If two specialists disagree or one found nothing relevant, say so plainly rather than \
papering over it.
5. Do not invent any fact, figure, price, dosage or date that isn't present in the specialists' \
findings below.
6. End with a short "Recommended next steps" list.
"""


class ReportAgent(BaseAgent):
    name = "report_agent"
    description = "Synthesizes every specialist agent's findings into one final farmer report."

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def run(self, request: AgentRequest) -> AgentResult:
        agent_results: List[AgentResult] = (request.context or {}).get("agent_results", [])

        if not agent_results:
            return AgentResult(
                agent_name=self.name,
                summary="No specialist findings were available to build a report from.",
                grounded=False,
            )

        findings_block, combined_sources = self._build_findings_block(agent_results)
        user_prompt = (
            f"FARMER'S QUESTION: {request.query}\n\n"
            f"SPECIALIST FINDINGS:\n{findings_block}\n\n"
            f"Write the consolidated report now."
        )

        any_grounded = any(r.grounded for r in agent_results)

        try:
            report_text = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except LLMError as e:
            # Fall back to a straightforward concatenation so the farmer
            # still gets every specialist's answer even if the LLM that
            # would have merged them nicely is unreachable.
            logger.warning(f"Report Agent: LLM synthesis unavailable, falling back: {e}")
            report_text = self._fallback_concatenation(agent_results)

        return AgentResult(
            agent_name=self.name,
            summary=f"Consolidated report combining {len(agent_results)} specialist finding(s).",
            details=report_text.strip(),
            grounded=any_grounded,
            sources=combined_sources,
            data={"agents_consulted": [r.agent_name for r in agent_results]},
        )

    @staticmethod
    def _build_findings_block(agent_results: List[AgentResult]):
        lines = []
        combined_sources = []
        for result in agent_results:
            label = result.agent_name.replace("_", " ").title()
            lines.append(f"### {label}")
            lines.append(f"Grounded: {result.grounded}")
            lines.append(result.details or result.summary or "(no findings)")
            if result.error:
                lines.append(f"(Note: this agent hit an error: {result.error})")
            lines.append("")
            for source in result.sources:
                combined_sources.append({"agent": result.agent_name, **source})
        return "\n".join(lines), combined_sources

    @staticmethod
    def _fallback_concatenation(agent_results: List[AgentResult]) -> str:
        sections = []
        for result in agent_results:
            label = result.agent_name.replace("_", " ").title()
            body = result.details or result.summary or "No findings."
            sections.append(f"## {label}\n{body}")
        sections.append(
            "## Recommended next steps\nReview each section above. If anything is unclear or "
            "not grounded in a source, confirm with a local agricultural extension officer "
            "before acting."
        )
        return "\n\n".join(sections)
