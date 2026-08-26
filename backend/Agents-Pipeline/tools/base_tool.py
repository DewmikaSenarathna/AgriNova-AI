"""
base_tool.py
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
