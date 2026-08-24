"""
weather_tool.py
"""

import logging
import re
from typing import Optional, Tuple

import requests

import agent_config
from tools.base_tool import BaseTool
from tools.tool_types import ToolResult

logger = logging.getLogger(__name__)

# WMO weather codes (used by Open-Meteo) -> short human description.
# https://open-meteo.com/en/docs
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}

# Very small helper for pulling "in <place>" / "at <place>" / "near <place>"
# out of a free-text question. Intentionally simple (not NLP) — passing
# context={"location": ...} (or lat/lon) is the more reliable path;
# this is just a helpful fallback.
LOCATION_PATTERN = re.compile(r"\b(?:in|at|near|for)\s+([A-Za-z][A-Za-z\s,]{2,40})$")


class WeatherTool(BaseTool):
    name = "weather_api"
    description = (
        "Resolves a location and fetches a short-range weather forecast "
        "(current conditions + N-day daily forecast) from Open-Meteo."
    )

    # -- Location resolution 
    def resolve_location(self, query: str, context: Optional[dict] = None) -> Tuple[float, float, str]:
        context = context or {}
        if "latitude" in context and "longitude" in context:
            return (
                float(context["latitude"]),
                float(context["longitude"]),
                context.get("location", "your location"),
            )

        place_name = context.get("location")
        if not place_name:
            match = LOCATION_PATTERN.search(query or "")
            if match:
                place_name = match.group(1).strip(" ,")

        if place_name:
            geocoded = self._geocode(place_name)
            if geocoded:
                return geocoded

        return (
            agent_config.WEATHER_DEFAULT_LATITUDE,
            agent_config.WEATHER_DEFAULT_LONGITUDE,
            agent_config.WEATHER_DEFAULT_LOCATION_NAME,
        )

    def _geocode(self, place_name: str) -> Optional[Tuple[float, float, str]]:
        try:
            response = requests.get(
                agent_config.WEATHER_GEOCODING_URL,
                params={"name": place_name, "count": 1},
                timeout=agent_config.WEATHER_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results = response.json().get("results")
            if not results:
                return None
            top = results[0]
            label = ", ".join(
                part for part in [top.get("name"), top.get("admin1"), top.get("country")] if part
            )
            return float(top["latitude"]), float(top["longitude"]), label
        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Geocoding '{place_name}' failed: {e}")
            return None

    # -- Forecast fetch 
    def _fetch_forecast(self, latitude: float, longitude: float) -> dict:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "forecast_days": agent_config.WEATHER_FORECAST_DAYS,
            "timezone": "auto",
        }
        response = requests.get(
            agent_config.WEATHER_FORECAST_URL,
            params=params,
            timeout=agent_config.WEATHER_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    # -- Public entry point 
    def run(self, query: str = "", context: Optional[dict] = None) -> ToolResult:
        latitude, longitude, location_label = self.resolve_location(query, context)

        try:
            forecast = self._fetch_forecast(latitude, longitude)
        except requests.RequestException as e:
            return ToolResult(
                ok=False,
                error=f"Could not reach the weather service for {location_label}: {e}",
            )

        current = forecast.get("current", {})
        daily = forecast.get("daily", {})
        structured_data = {
            "location": location_label,
            "latitude": latitude,
            "longitude": longitude,
            "current": {
                "temperature_c": current.get("temperature_2m"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "condition": WMO_CODES.get(current.get("weather_code"), "unknown"),
            },
            "daily_forecast": [
                {
                    "date": daily.get("time", [None])[i] if i < len(daily.get("time", [])) else None,
                    "condition": WMO_CODES.get(
                        daily.get("weather_code", [None])[i]
                        if i < len(daily.get("weather_code", []))
                        else None,
                        "unknown",
                    ),
                    "temp_max_c": daily.get("temperature_2m_max", [None])[i]
                    if i < len(daily.get("temperature_2m_max", [])) else None,
                    "temp_min_c": daily.get("temperature_2m_min", [None])[i]
                    if i < len(daily.get("temperature_2m_min", [])) else None,
                    "precipitation_sum_mm": daily.get("precipitation_sum", [None])[i]
                    if i < len(daily.get("precipitation_sum", [])) else None,
                }
                for i in range(len(daily.get("time", [])))
            ],
        }

        return ToolResult(
            ok=True,
            data=structured_data,
            text=self._format_forecast_text(structured_data),
            source={
                "source": "Open-Meteo Weather API",
                "url": "https://open-meteo.com",
                "location": location_label,
            },
        )

    @staticmethod
    def _format_forecast_text(data: dict) -> str:
        c = data["current"]
        lines = [
            f"Current conditions in {data['location']}: {c['condition']}, {c['temperature_c']}°C, "
            f"{c['humidity_pct']}% humidity, wind {c['wind_speed_kmh']} km/h, "
            f"{c['precipitation_mm']} mm precipitation.",
            "",
            f"{len(data['daily_forecast'])}-day forecast:",
        ]
        for day in data["daily_forecast"]:
            lines.append(
                f"- {day['date']}: {day['condition']}, {day['temp_min_c']}-{day['temp_max_c']}°C, "
                f"{day['precipitation_sum_mm']} mm rain expected."
            )
        return "\n".join(lines)
