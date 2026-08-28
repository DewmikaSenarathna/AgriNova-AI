"""
explainability.py
==================
PHASE 13 — Explainable AI.

"Never answer without showing evidence." Instead of handing a farmer a
bare instruction —

    Use fertilizer X.

— every final answer is broken into the shape a farmer (or an
auditor) can actually check:

    Recommendation -> Reason -> Supporting documents -> Confidence -> References

WHY THIS ISN'T ANOTHER LLM CALL
--------------------------------
`recommendation` / `reason` / `next_steps` are PARSED out of the
Report Agent's already-written text — see `report_agent.py`'s
`SYSTEM_PROMPT`, which now requires exactly that structure
(`## Recommendation` / `## Reason` / `## Recommended next steps`).
`confidence` is computed with a plain, additive, fully-inspectable
formula over `AgentResult.grounded` / retrieval similarity scores /
errors — deliberately NOT another LLM call asked to grade its own
answer. An LLM can write a fluent, confident-SOUNDING recommendation
whether or not it's actually well-supported; a farmer's trust in the
confidence score shouldn't hinge on the same model also being an
honest judge of its own homework. Every number this module produces
comes with the plain-language `factors` that explain it — the
confidence score is itself explainable, not a black box.

HOW IT FITS THE PIPELINE
-------------------------
`agent_orchestrator.py` calls `build_explanation()` once, right after
the Report Agent produces `final_report`, and attaches the result to
`OrchestratedAnswer.explanation`. `api.py` serializes it straight
through; the frontend's `RecommendationLedger.jsx` renders its five
fields as five distinct, visually separated stages instead of one wall
of prose.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agent_types import AgentResult

# Section headers the Report Agent's prompt is asked to use (see
# report_agent.py's SYSTEM_PROMPT). Matched loosely (case-insensitive,
# "##" through "####", tolerant of a little trailing punctuation) so a
# slightly-off LLM rendering doesn't silently fall through to the
# unstructured fallback below.
_RECOMMENDATION_HEADER = re.compile(r"^#{1,4}\s*recommendation\s*:?\s*$", re.IGNORECASE)
_REASON_HEADER = re.compile(r"^#{1,4}\s*reason(ing)?\s*:?\s*$", re.IGNORECASE)
_NEXT_STEPS_HEADER = re.compile(r"^#{1,4}\s*recommended next steps\s*:?\s*$", re.IGNORECASE)
_ANY_HEADER = re.compile(r"^#{1,4}\s+\S.*$")


def split_recommendation_and_reason(report_text: str) -> Dict[str, str]:
    """
    Splits the Report Agent's markdown-ish text into its three
    required sections. If none of the expected headers are found at
    all (an LLM that ignored the instruction, or the very old
    unstructured Phase 7-12 output), falls back to treating the first
    paragraph as the recommendation and everything else as the reason
    — so the UI is never left with nothing to put in the
    "Recommendation" stage.
    """
    lines = (report_text or "").replace("\r\n", "\n").split("\n")

    sections: Dict[str, List[str]] = {"recommendation": [], "reason": [], "next_steps": []}
    current = None
    saw_headers = False

    for line in lines:
        stripped = line.strip()
        if _RECOMMENDATION_HEADER.match(stripped):
            current = "recommendation"
            saw_headers = True
            continue
        if _REASON_HEADER.match(stripped):
            current = "reason"
            saw_headers = True
            continue
        if _NEXT_STEPS_HEADER.match(stripped):
            current = "next_steps"
            saw_headers = True
            continue
        if _ANY_HEADER.match(stripped):
            # Some other heading the LLM produced instead (e.g. a
            # per-specialist heading) — always treat it, and everything
            # under it, as supporting reasoning rather than letting it
            # bleed into whatever section came before (the
            # "Recommendation" section in particular should stay tight).
            current = "reason"
            sections[current].append(line)
            continue
        if current:
            sections[current].append(line)

    if not saw_headers:
        paragraphs = [p.strip() for p in (report_text or "").split("\n\n") if p.strip()]
        recommendation = paragraphs[0] if paragraphs else ""
        reason = "\n\n".join(paragraphs[1:])
        return {"recommendation": recommendation, "reason": reason, "next_steps": ""}

    return {
        "recommendation": "\n".join(sections["recommendation"]).strip(),
        "reason": "\n".join(sections["reason"]).strip(),
        "next_steps": "\n".join(sections["next_steps"]).strip(),
    }


def _collect_similarities(sources: List[Dict[str, Any]]) -> List[float]:
    return [
        float(s["similarity"])
        for s in sources
        if isinstance(s.get("similarity"), (int, float))
    ]


def compute_confidence(
    agent_results: List[AgentResult], final_report: AgentResult
) -> Dict[str, Any]:
    """
    A plain, additive formula — never a bare number with no way to
    check it:

      +0.50 x (grounded specialists / specialists that ran)
      +0.30 x (average retrieval similarity, for any vector-backed
               sources used — this weight is redistributed onto the
               grounded-ratio term instead when there's no similarity
               data at all, e.g. a purely tool-backed answer like
               weather/market, so those answers aren't unfairly capped
               just because there was nothing to embed)
      +0.20 if the consolidated report itself is grounded
      -0.15 per specialist that errored out entirely

    Clipped to [0, 1] and bucketed into Low (<0.4) / Medium (<0.7) /
    High (>=0.7). `factors` lists exactly which of the above applied,
    in plain language a farmer can read — see this module's docstring
    for why this is a formula, not another LLM call.
    """
    total = len(agent_results) or 1
    errored = [r for r in agent_results if r.error]
    usable = [r for r in agent_results if not r.error]

    factors: List[str] = []
    score = 0.0

    grounded_count = sum(1 for r in usable if r.grounded)
    grounded_ratio = grounded_count / total
    score += 0.5 * grounded_ratio
    factors.append(f"{grounded_count} of {total} specialist finding(s) were grounded in real evidence")

    similarities = _collect_similarities(final_report.sources or [])
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        score += 0.3 * avg_sim
        factors.append(
            f"retrieved knowledge-base passages matched the question with "
            f"{avg_sim:.0%} average similarity"
        )
    else:
        score += 0.3 * grounded_ratio

    if final_report.grounded:
        score += 0.2
        factors.append("the consolidated report is grounded in cited sources")
    else:
        factors.append("the consolidated report could not be fully grounded — treat with extra care")

    if errored:
        penalty = min(0.15 * len(errored), score)
        score -= penalty
        factors.append(f"{len(errored)} specialist(s) failed and were excluded from this answer")

    score = max(0.0, min(1.0, score))
    if score >= 0.7:
        level = "High"
    elif score >= 0.4:
        level = "Medium"
    else:
        level = "Low"

    return {"level": level, "score": round(score, 2), "factors": factors}


def _source_label(source: Dict[str, Any]) -> str:
    if source.get("heading"):
        return source["heading"]
    if source.get("source"):
        return source["source"]
    if source.get("crop"):
        return f"Market price — {source['crop']}"
    if source.get("location"):
        return source["location"]
    return "Untitled source"


def build_references(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    De-duplicates and numbers the combined source list into a
    bibliography-style reference list. The numbering follows the same
    order the sources were combined in by the Report Agent
    (`report_agent.py`'s `_build_findings_block`), which is also the
    order the `[Source N]` markers in the report text refer to.
    """
    seen = set()
    references: List[Dict[str, Any]] = []
    for source in sources:
        key = source.get("chunk_id") or source.get("url") or (
            _source_label(source),
            source.get("agent"),
        )
        if key in seen:
            continue
        seen.add(key)
        references.append(
            {
                "n": len(references) + 1,
                "label": _source_label(source),
                "agent": source.get("agent"),
                "similarity": source.get("similarity")
                if isinstance(source.get("similarity"), (int, float))
                else None,
                "detail": source.get("text") or source.get("url") or source.get("location"),
            }
        )
    return references


@dataclass
class Explanation:
    """
    PHASE 13 — the structured, evidence-first shape every final answer
    is broken into. Attached to `OrchestratedAnswer.explanation` (see
    agent_orchestrator.py) and serialized straight through `api.py`'s
    `AskResponse`, so the frontend never has to re-derive any of this
    from the raw report text itself.
    """
    recommendation: str
    reason: str
    next_steps: str
    supporting_documents: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    references: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "reason": self.reason,
            "next_steps": self.next_steps,
            "supporting_documents": self.supporting_documents,
            "confidence": self.confidence,
            "references": self.references,
        }


def build_explanation(agent_results: List[AgentResult], final_report: AgentResult) -> Explanation:
    """The one entry point `agent_orchestrator.py` calls."""
    sections = split_recommendation_and_reason(final_report.details or final_report.summary or "")
    confidence = compute_confidence(agent_results, final_report)
    references = build_references(final_report.sources or [])

    recommendation = sections["recommendation"] or final_report.summary or (
        "No confident recommendation could be produced from this question — please consult "
        "a local agricultural extension officer."
    )

    return Explanation(
        recommendation=recommendation,
        reason=sections["reason"],
        next_steps=sections["next_steps"],
        supporting_documents=final_report.sources or [],
        confidence=confidence,
        references=references,
    )
