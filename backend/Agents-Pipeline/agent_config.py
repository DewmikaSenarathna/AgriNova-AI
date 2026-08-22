"""
agent_config.py
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 1. FOLDER PATHS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = OUTPUT_DIR / "logs"

for folder in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 2. PLANNER AGENT SETTINGS

# "keyword"  -> fast, free, deterministic keyword/regex routing (default).
# "llm"       -> ask the configured LLM (see rag_bridge.rag_config) to
#               choose which agents to run, as strict JSON. Slower and
#               costs one extra LLM call per question, but copes better
#               with oddly-phrased or multi-intent questions.
PLANNER_MODE = os.getenv("PLANNER_MODE", "keyword").strip().lower()

# If keyword routing finds no confident match at all, the Planner falls
# back to running the General Agent (the plain Phase 6 RAGPipeline) so
# the farmer always gets an answer instead of an empty routing table.
PLANNER_FALLBACK_AGENT = "general"

# Safety cap: even if the planner (LLM mode especially) tries to select
# more agents than this, only the top N (by keyword-match strength /
# LLM order) actually run — keeps latency and LLM spend bounded.
PLANNER_MAX_AGENTS_PER_REQUEST = int(os.getenv("PLANNER_MAX_AGENTS_PER_REQUEST", "4"))

# 3. WEATHER AGENT SETTINGS
# Open-Meteo (https://open-meteo.com) is used because it's free, requires
# NO API key, and has generous rate limits for a student/portfolio
# project — matching the "Weather API" line in the tech-stack table
# without forcing every contributor to go sign up for a key just to
# run the demo. Swap WEATHER_PROVIDER + the two URLs below to point at
# a different provider without touching weather_agent.py.
WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "open-meteo")
WEATHER_GEOCODING_URL = os.getenv(
    "WEATHER_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
WEATHER_FORECAST_URL = os.getenv("WEATHER_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")

# Used when the farmer's question doesn't mention a place and the
# Planner didn't pass one in via context (e.g. from a logged-in
# profile) — keeps the Weather Agent usable out of the box in demos.
WEATHER_DEFAULT_LOCATION_NAME = os.getenv("WEATHER_DEFAULT_LOCATION_NAME", "Colombo, Sri Lanka")
WEATHER_DEFAULT_LATITUDE = float(os.getenv("WEATHER_DEFAULT_LATITUDE", "6.9271"))
WEATHER_DEFAULT_LONGITUDE = float(os.getenv("WEATHER_DEFAULT_LONGITUDE", "79.8612"))

WEATHER_FORECAST_DAYS = int(os.getenv("WEATHER_FORECAST_DAYS", "3"))
WEATHER_REQUEST_TIMEOUT_SECONDS = int(os.getenv("WEATHER_REQUEST_TIMEOUT_SECONDS", "15"))

# 4. MARKET AGENT SETTINGS
# No free, universal, no-key crop-price API exists for every region, so
# the Market Agent checks a small local dataset FIRST (fast, exact,
# offline-friendly) and only falls back to a knowledge-base + LLM
# answer when the crop isn't in that dataset. Replace this file with a
# live market-API integration later without changing any other agent.
MARKET_PRICE_DATA_PATH = DATA_DIR / "market_prices_sample.json"
MARKET_PRICE_CURRENCY = os.getenv("MARKET_PRICE_CURRENCY", "LKR")

# 5. REPORT AGENT SETTINGS

REPORT_MAX_TOKENS = int(os.getenv("REPORT_MAX_TOKENS", "900"))

# 6. API SERVER SETTINGS (api.py)
# Runs on its own port, separate from RAG-Pipeline/api.py, so both
# Phase 6 (plain RAG) and Phase 7 (agentic) endpoints can be run side
# by side during development/demos.

API_HOST = os.getenv("AGENTS_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("AGENTS_API_PORT", "8001"))
API_CORS_ORIGINS = [o.strip() for o in os.getenv("AGENTS_API_CORS_ORIGINS", "*").split(",")]

# 7. LOGGING

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
