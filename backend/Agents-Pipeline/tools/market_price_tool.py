"""
market_price_tool.py
"""

import json
import logging
import re
from typing import Optional

import agent_config
from tools.base_tool import BaseTool
from tools.tool_types import ToolResult

logger = logging.getLogger(__name__)


class MarketPriceTool(BaseTool):
    name = "market_price_api"
    description = (
        "Looks up a crop's low/average/high market price. Backed today by a local "
        "sample dataset; swap in a live market-price API here without changing any agent."
    )

    def __init__(self):
        self._prices = self._load_local_prices()
        self._currency = agent_config.MARKET_PRICE_CURRENCY

    @staticmethod
    def _load_local_prices() -> dict:
        path = agent_config.MARKET_PRICE_DATA_PATH
        if not path.exists():
            logger.warning(f"Market Price API tool: local price dataset not found at {path}.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("prices", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Market Price API tool: could not read local price dataset: {e}")
            return {}

    def match_crop(self, question: str) -> Optional[str]:
        """Word-boundary match of a known crop key (or its display form)
        against a farmer's question. Word-boundary (not plain substring)
        matters — a plain substring check would match "rice" inside
        "price", misidentifying a generic price question as a rice-price
        question."""
        question_lower = (question or "").lower()
        for crop_key in self._prices:
            crop_display = crop_key.replace("_", " ")
            pattern = r"\b" + re.escape(crop_key) + r"\w*|\b" + re.escape(crop_display) + r"\w*"
            if re.search(pattern, question_lower):
                return crop_key
        return None

    def run(self, crop: Optional[str] = None, question: str = "") -> ToolResult:
        crop_key = crop.strip().lower().replace(" ", "_") if crop else None
        if crop_key and crop_key not in self._prices:
            crop_key = None
        if not crop_key:
            crop_key = self.match_crop(question)

        if not crop_key or crop_key not in self._prices:
            return ToolResult(
                ok=False,
                error="No price data for this crop in the local market-price dataset.",
            )

        price = self._prices[crop_key]
        crop_display = crop_key.replace("_", " ").title()
        text = (
            f"{crop_display}: average {self._currency} {price['average']}/kg "
            f"(range {self._currency} {price['low']}-{price['high']}/kg)."
        )
        return ToolResult(
            ok=True,
            data={"crop": crop_display, "currency": self._currency, **price},
            text=text,
            source={
                "source": "AgriNova AI local market price dataset (sample/demo data)",
                "crop": crop_display,
            },
        )
