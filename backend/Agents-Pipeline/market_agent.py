"""
market_agent.py
=================
Single responsibility: crop market-price information and simple
sell/hold guidance.

Two-tier strategy, with Phase 9's tools formalizing tier 1:

    Step 1: MarketPriceTool ("Market Price API" tool) -> fast, exact,
            offline-safe local lookup (data/market_prices_sample.json
            today; swap for a live pricing API inside the tool only)
            |
            v  (if the crop isn't in the dataset)
    Step 2: fall back to the shared VectorDBTool + LLM, the same way
            the other KnowledgeAgent subclasses work, for general
            market guidance the knowledge base may contain

Everything else in the pipeline only depends on this agent returning
an AgentResult, so either tier can change independently.
"""

import logging
from typing import Optional

from agent_types import AgentRequest, AgentResult
from knowledge_agent import KnowledgeAgent
from tools.market_price_tool import MarketPriceTool

logger = logging.getLogger(__name__)


class MarketAgent(KnowledgeAgent):
    name = "market_agent"
    description = "Reports crop market prices and simple sell/hold guidance."
    domain_label = "market price"
    query_hints = ["market price", "selling price", "crop price trend"]
    system_prompt = (
        "You are AgriNova AI's Market Agent, a crop-market assistant for farmers.\n\n"
        "Using ONLY the numbered SOURCE excerpts provided:\n"
        "1. Share whatever market/price guidance the sources contain, citing them as "
        "[Source N].\n"
        "2. Do not invent specific prices that are not present in the sources — market prices "
        "change daily and a wrong number here can cost a farmer money.\n"
        "3. If no price data is available, say so plainly and suggest checking the local "
        "market / agriculture department price bulletin.\n"
        "4. Keep the answer short, plain-language and actionable."
    )

    def __init__(self, *args, market_tool: Optional[MarketPriceTool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_tool = market_tool or MarketPriceTool()

    def run(self, request: AgentRequest) -> AgentResult:
        question = (request.query or "").strip()
        crop_hint = (request.context or {}).get("crop")

        result = self.market_tool.execute(crop=crop_hint, question=question)

        if result.ok:
            price = result.data
            crop_display = price["crop"]
            currency = price["currency"]
            details = (
                f"Latest available price for {crop_display}:\n"
                f"- Low: {currency} {price['low']} per kg\n"
                f"- Average: {currency} {price['average']} per kg\n"
                f"- High: {currency} {price['high']} per kg\n\n"
                f"These are indicative figures — always confirm same-day prices at your "
                f"local market or with the agriculture department's price bulletin before "
                f"deciding whether to sell now or hold."
            )
            return AgentResult(
                agent_name=self.name,
                summary=result.text,
                details=details,
                grounded=True,
                sources=[result.source],
                data=price,
            )

        # Step 2 — fall back to knowledge-base (VectorDBTool) + LLM (KnowledgeAgent.run)
        logger.info(f"Market Agent: no local price match ({result.error}); falling back to KB.")
        return super().run(request)
