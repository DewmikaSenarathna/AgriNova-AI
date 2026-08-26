"""
image_agent.py
================
PHASE 9 — Single responsibility: describe what's visible in a
farmer-submitted crop photo, using the "Image Model" tool
(tools/image_model_tool.py).

This agent deliberately does NOT diagnose a disease or pest itself —
that stays Disease Agent / Pest Agent's job, grounded in the shared
knowledge base. The Image Agent's output is honestly `grounded=False`
(a vision-model description, not a retrieved source), and
`agent_orchestrator.py` feeds that description into whichever
knowledge agents run alongside it (via
`request.context["image_description"]`, see knowledge_agent.py) so a
photo of, e.g., yellowing spotted leaves can help Disease Agent find
the right knowledge-base sources even if the farmer's own words
didn't describe the symptoms in detail.

    farmer's photo (base64, in request.context["image_base64"])
            |
            v
    ImageModelTool -> vision LLM -> plain-language description
            |
            v
    AgentResult(grounded=False, data={"description": ...})
"""

import logging
from typing import Optional

from base_agent import BaseAgent
from agent_types import AgentRequest, AgentResult
from rag_bridge import LLMClient
from tools.image_model_tool import ImageModelTool

logger = logging.getLogger(__name__)


class ImageAgent(BaseAgent):
    name = "image_agent"
    description = "Describes what's visible in a farmer-submitted crop photo using a vision model."

    def __init__(self, llm: Optional[LLMClient] = None, tool: Optional[ImageModelTool] = None):
        self.tool = tool or ImageModelTool(llm=llm)

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        image_base64 = context.get("image_base64")

        if not image_base64:
            return AgentResult(
                agent_name=self.name,
                summary="No photo was attached to this question.",
                details=(
                    "The Image Agent only runs something useful when a photo is attached "
                    "(context['image_base64']). Ask the farmer to attach a clear, close-up "
                    "photo of the affected plant part if visual symptoms are involved."
                ),
                grounded=False,
            )

        result = self.tool.execute(image_base64=image_base64, question=request.query or "")

        if not result.ok:
            return AgentResult(
                agent_name=self.name,
                summary="Couldn't analyze the attached photo.",
                details=result.error or "",
                grounded=False,
                error=result.error,
            )

        return AgentResult(
            agent_name=self.name,
            summary=result.text.strip().split("\n")[0][:200],
            details=result.text.strip(),
            # This is a vision-model INFERENCE, not a retrieved/citable
            # source — mirrors how the rest of the project reserves
            # `grounded=True` for knowledge-base or live-API evidence.
            grounded=False,
            sources=[result.source],
            data=result.data,
        )
