"""
image_agent.py
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
