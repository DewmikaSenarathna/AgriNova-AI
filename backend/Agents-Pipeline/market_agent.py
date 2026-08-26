"""
market_agent.py
"""

import json
import logging
import re
from typing import Optional

import agent_config
from agent_types import AgentRequest, AgentResult
from knowledge_agent import KnowledgeAgent

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_prices = self._load_local_prices()

    @staticmethod
    def _load_local_prices() -> dict:
        path = agent_config.MARKET_PRICE_DATA_PATH
        if not path.exists():
            logger.warning(f"Market Agent: local price dataset not found at {path}.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("prices", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Market Agent: could not read local price dataset: {e}")
            return {}

    def _match_local_crop(self, question: str) -> Optional[str]:
        """Word-boundary match of a known crop key (or its display form)
        against the farmer's question. Word-boundary (not plain substring)
        matters here — a plain substring check would match "rice" inside
        "price", misidentifying a generic price question as a rice-price
        question. Simple by design — a real deployment would use the crop
        name the Planner Agent already extracted, passed via
        `request.context['crop']`, which is checked first below."""
        question_lower = question.lower()
        for crop_key in self._local_prices:
            crop_display = crop_key.replace("_", " ")
            # Word-boundary at the START only (see planner_agent.py's
            # keyword matching for the same pattern) so "coconuts" still
            # matches "coconut", without "rice" matching inside "price".
            pattern = r"\b" + re.escape(crop_key) + r"\w*|\b" + re.escape(crop_display) + r"\w*"
            if re.search(pattern, question_lower):
                return crop_key
        return None

    def run(self, request: AgentRequest) -> AgentResult:
        question = (request.query or "").strip()
        crop_key = (request.context or {}).get("crop")
        if crop_key:
            crop_key = crop_key.strip().lower().replace(" ", "_")
            if crop_key not in self._local_prices:
                crop_key = None
        if not crop_key:
            crop_key = self._match_local_crop(question)

        if crop_key and crop_key in self._local_prices:
            price = self._local_prices[crop_key]
            crop_display = crop_key.replace("_", " ").title()
            currency = agent_config.MARKET_PRICE_CURRENCY
            summary = (
                f"{crop_display}: average {currency} {price['average']}/kg "
                f"(range {currency} {price['low']}-{price['high']}/kg)."
            )
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
                summary=summary,
                details=details,
                grounded=True,
                sources=[{
                    "source": "AgriNova AI local market price dataset (sample/demo data)",
                    "crop": crop_display,
                }],
                data={"crop": crop_display, "currency": currency, **price},
            )

        # Step 2 — fall back to knowledge-base + LLM (KnowledgeAgent.run)
        return super().run(request)
