"""
tools
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
