"""
base_tool.py
============
PHASE 9 — Connect External Tools

The contract every tool (Weather API, Market Price API, Government PDF
Search, Vector Database, Image Model) implements, mirroring
`base_agent.py`'s "single responsibility + never raise out of the
public entry point" design one layer down the stack:

    Agent
      |
      v
    tool.execute(**kwargs)   <-  public entry point, NEVER raises
      |
      v
    tool.run(**kwargs)        <-  subclasses implement their one job here
      |
      v
    ToolResult(ok, data, text, source, error)

Why tools are a separate layer from agents at all: several agents need
the SAME external capability (every KnowledgeAgent subclass needs the
vector database; the Government Agent needs both the vector database
AND a PDF full-text search). Putting the capability in a tool instead
of duplicating it inside each agent means:
  - one place to swap Open-Meteo for another weather provider,
  - one place to swap the local market JSON for a live pricing API,
  - one place to point the vision model at a different backend,
without touching the agents that call it. Agents stay focused on
*reasoning about* what a tool returns, not on how to reach it.
"""

import logging
from abc import ABC, abstractmethod

from tools.tool_types import ToolResult

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Every tool:
      - has exactly ONE external capability (name + description say
        what it is, exactly like BaseAgent)
      - takes whatever keyword arguments its capability needs, returns
        a ToolResult
      - never raises out of `execute()` — a down weather API or an
        empty vector database should degrade to an honest
        ToolResult(ok=False, error=...), not crash the agent that
        called it (which itself must not crash the whole request —
        see base_agent.py)
    """

    name: str = "base_tool"
    description: str = "Base tool — not used directly."

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Subclasses implement their one external capability here."""
        raise NotImplementedError

    def execute(self, **kwargs) -> ToolResult:
        """Public entry point agents call. Wraps `run()` so ANY
        unexpected exception (a malformed API response, a network
        error the tool's own code didn't anticipate, ...) turns into a
        clearly-labelled ToolResult instead of taking down the agent."""
        try:
            return self.run(**kwargs)
        except Exception as e:  # last-resort safety net, by design
            logger.exception(f"Tool '{self.name}' failed unexpectedly.")
            return ToolResult(ok=False, error=f"{self.name} error: {e}")
