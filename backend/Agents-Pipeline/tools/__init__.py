"""
tools
=====
PHASE 9 — Connect External Tools.

Every external capability an agent can reach for lives in this
package, behind the shared `BaseTool` contract (see base_tool.py):

    Agent            Tool                    External capability
    --------------   ---------------------   -------------------------------
    Weather Agent    WeatherTool             Open-Meteo Weather API
    Market Agent     MarketPriceTool         Market price dataset (swap for
                                              a live pricing API)
    Government Agent GovernmentPDFSearchTool Full-text search over official
                                              PDFs (+ VectorDBTool below)
    Disease/Pest/     VectorDBTool           Shared ChromaDB vector database
    Fertilizer/Soil
    Government Agent
    Image Agent      ImageModelTool          Vision-capable LLM (crop photos)

`TOOL_REGISTRY` below is a lightweight, import-light catalogue (name ->
ToolSpec) for anything that wants to list "what tools exist" — e.g. the
`/api/tools` endpoint in api.py — without constructing every tool
(some of which open network connections or load embedding models on
init) just to read its description.
"""

from tools.base_tool import BaseTool
from tools.tool_types import PDFSearchHit, ToolResult, ToolSpec
from tools.weather_tool import WeatherTool
from tools.market_price_tool import MarketPriceTool
from tools.vector_db_tool import VectorDBTool
from tools.government_pdf_search_tool import GovernmentPDFSearchTool
from tools.image_model_tool import ImageModelTool

TOOL_REGISTRY = {
    "weather_api": ToolSpec(
        name="weather_api",
        description=WeatherTool.description,
        used_by=["weather_agent"],
    ),
    "market_price_api": ToolSpec(
        name="market_price_api",
        description=MarketPriceTool.description,
        used_by=["market_agent"],
    ),
    "government_pdf_search": ToolSpec(
        name="government_pdf_search",
        description=GovernmentPDFSearchTool.description,
        used_by=["government_agent"],
    ),
    "vector_database": ToolSpec(
        name="vector_database",
        description=VectorDBTool.description,
        used_by=[
            "disease_agent", "pest_agent", "fertilizer_agent",
            "soil_agent", "government_agent", "market_agent", "general_agent",
        ],
    ),
    "image_model": ToolSpec(
        name="image_model",
        description=ImageModelTool.description,
        used_by=["image_agent"],
    ),
}

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolSpec",
    "PDFSearchHit",
    "WeatherTool",
    "MarketPriceTool",
    "VectorDBTool",
    "GovernmentPDFSearchTool",
    "ImageModelTool",
    "TOOL_REGISTRY",
]
