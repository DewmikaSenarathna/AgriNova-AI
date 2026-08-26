"""
tool_types.py
=============
PHASE 9 — the shared "envelope" every tool speaks, so every agent can
call ANY tool through the same interface, exactly the way `AgentRequest`
/ `AgentResult` (agent_types.py) let the orchestrator treat every agent
identically without knowing how it works internally.

    Agent  ->  tool.run(**kwargs)  ->  ToolResult  ->  Agent decides what
                                                        to do with it

A tool is a thin wrapper around ONE external capability an agent needs
(an HTTP API, a local dataset, a full-text search, a vision model, a
vector database) — never an LLM call, and never a full agent. That
separation keeps tools independently testable/swappable and keeps
agents focused on *reasoning*, not on HTTP/plumbing details.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """
    What every tool hands back to the agent that called it.

    ok        -> True if the tool call succeeded and `data` is usable.
                 False means the agent should degrade gracefully (never
                 crash) — the same "never raises out of execute()"
                 philosophy as BaseAgent.execute(), just one layer down.
    data       -> structured, machine-usable output specific to this
                  tool (e.g. weather numbers, a price row, PDF search
                  hits, a vision-model description).
    text       -> a short, ready-to-embed-in-a-prompt text rendering of
                  `data`, so agents don't each re-invent their own
                  "turn this JSON into a sentence" logic.
    source     -> a citable description of where this evidence came
                  from (tool name, provider, URL, file path...), in the
                  same shape agents already put in AgentResult.sources.
    error      -> set (with ok=False) when the tool could not do its
                  job — a down API, a missing dataset, an unreachable
                  model. Agents surface this as an honest AgentResult
                  rather than pretending the tool succeeded.
    """
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "data": self.data,
            "text": self.text,
            "source": self.source,
            "error": self.error,
        }


@dataclass
class PDFSearchHit:
    """One matching document from GovernmentPDFSearchTool (or any future
    full-text-search tool over Document-Processing-Pipeline output)."""
    file_name: str
    title: str
    document_type: str
    snippet: str
    score: int  # simple keyword-hit count, see government_pdf_search_tool.py

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "title": self.title,
            "document_type": self.document_type,
            "snippet": self.snippet,
            "score": self.score,
        }


@dataclass
class ToolSpec:
    """
    A tool's self-description — lets the Planner/Report Agents (or a
    future LLM-driven tool-selection step) list "what tools exist and
    when to use them" without importing every tool module just to read
    a docstring. Mirrors `BaseAgent.name` / `BaseAgent.description`.
    """
    name: str
    description: str
    used_by: List[str] = field(default_factory=list)  # agent names

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "used_by": self.used_by}
