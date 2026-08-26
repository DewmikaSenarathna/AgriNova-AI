"""
conversation_memory.py
=======================
PHASE 11 — Conversation Memory.

Farmers should not have to repeat themselves. Without this module,
every question starts from zero:

    Day 1   Farmer: "My tomato crop in Kurunegala has yellowing leaves."
            AI: [diagnoses, but forgets everything the moment it answers]

    Day 2   Farmer: "Should I irrigate today?"
            AI: "Which crop? Where are you located?"  <- bad experience

With this module, the second question already has the AI knowing:

    Crop:              Tomato
    Location:           Kurunegala
    Field:              (whatever the farmer told it)
    Previous disease:    Early blight (found on Day 1)
    Previous fertilizer:  (whatever the Fertilizer Agent last recommended)
    Weather history:     (the last few forecasts checked)

    Day 2   Farmer: "Should I irrigate today?"
            AI: [already knows it's Tomato, in Kurunegala, with a recent
                 early-blight diagnosis on file] -> answers directly,
                 and can even flag "since your tomatoes had early
                 blight recently, avoid overhead irrigation."

HOW IT FITS THE PIPELINE
-------------------------
This mirrors the shape Phase 10 already established for cross-agent
context (`agent_types.format_prior_findings` /
`request.context["prior_findings"]`) — just one level up, across
QUESTIONS instead of across agents within one question:

    Phase 10 — agents within ONE question share findings via
               `context["prior_findings"]`
    Phase 11 — questions within ONE farmer's conversation share facts
               via `context["memory_summary"]` + recalled context keys
               (`crop`, `location`, `latitude`, `longitude`, `field`)

A conversation is identified by a `session_id` (one per farmer /
device / login — the caller decides). `agent_orchestrator.py`:

    1. loads that session's `FarmerMemory` at the START of `handle()`,
    2. merges its known facts into `context` (explicit context passed
       in for THIS question still wins over older memory — a farmer
       switching crops mid-conversation should not get stuck on the
       old one),
    3. hands agents a rendered `context["memory_summary"]` block (see
       `FarmerMemory.to_prompt_block`) the same way `prior_findings`
       is rendered, so agents/Report Agent can say "since your tomato
       crop had early blight recently..." instead of asking again,
    4. after the answer is built, extracts whatever NEW facts this
       turn revealed (from explicit context hints, simple keyword
       extraction on the question text, and the specialist agents'
       own grounded results — e.g. the Weather Agent's resolved
       location becomes memory) and persists the updated memory.

Storage is deliberately simple: one small JSON file per session under
`agent_config.MEMORY_DIR` (default `output/memory/`) — no database
server to stand up for a portfolio/demo project, easy to inspect by
hand, and trivial to swap for a real database later since every
caller only ever talks to `ConversationMemoryStore`, never the files
directly.
"""

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import agent_config

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, types only
    from agent_types import AgentResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Lightweight crop extraction — NOT NLP, just a keyword table. Passing
# context={"crop": "..."} explicitly (e.g. from a farmer's saved profile
# or a frontend dropdown) is always the more reliable path; this is a
# best-effort fallback so a crop mentioned in plain conversation
# ("my tomato leaves are yellowing") still gets remembered.
# ---------------------------------------------------------------------------
_CROP_ALIASES: Dict[str, str] = {}


def _register_crop(canonical: str, *aliases: str):
    _CROP_ALIASES[canonical.lower()] = canonical
    for alias in aliases:
        _CROP_ALIASES[alias.lower()] = canonical


_register_crop("Tomato", "tomatoes")
_register_crop("Chili", "chilli", "chillies", "chilies", "chili pepper")
_register_crop("Rice", "paddy")
_register_crop("Maize", "corn")
_register_crop("Onion", "onions", "big onion", "red onion")
_register_crop("Potato", "potatoes")
_register_crop("Cabbage", "cabbages")
_register_crop("Brinjal", "eggplant", "aubergine")
_register_crop("Cucumber", "cucumbers")
_register_crop("Beans", "green beans", "long beans")
_register_crop("Tea", "tea leaves")
_register_crop("Coconut", "coconuts")
_register_crop("Banana", "bananas", "plantain")
_register_crop("Mango", "mangoes", "mangos")
_register_crop("Okra", "ladies finger", "lady's finger", "ladies' finger")
_register_crop("Carrot", "carrots")
_register_crop("Cinnamon")
_register_crop("Pepper", "black pepper")
_register_crop("Wheat")
_register_crop("Sugarcane")
_register_crop("Cotton")
_register_crop("Groundnut", "peanut", "peanuts")
_register_crop("Soybean", "soya bean", "soybeans")
_register_crop("Watermelon", "watermelons")
_register_crop("Pumpkin", "pumpkins")
_register_crop("Papaya", "papayas", "pawpaw")
_register_crop("Pineapple", "pineapples")
_register_crop("Ginger")
_register_crop("Turmeric")
_register_crop("Garlic")
_register_crop("Cauliflower")

# Longest alias first, so "green beans" matches before the bare "beans"
# fragment inside it, etc.
_SORTED_ALIASES = sorted(_CROP_ALIASES.keys(), key=len, reverse=True)


def extract_crop_from_text(text: str) -> Optional[str]:
    """Best-effort keyword match for a crop name mentioned in free text.
    Returns the canonical display name (e.g. "tomatoes" -> "Tomato"), or
    None if nothing in `_CROP_ALIASES` matches."""
    text_lower = (text or "").lower()
    for alias in _SORTED_ALIASES:
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            return _CROP_ALIASES[alias]
    return None


# Names of the specialist AgentResults that turn into a remembered
# "last <domain> finding" — deliberately excludes market/government
# (price/policy findings go stale fast and re-asking is cheap+safer)
# and report/general/image/weather (weather is handled separately below,
# since it accumulates a short HISTORY rather than a single "latest").
_REMEMBERED_FINDING_AGENTS = {
    "disease_agent": "last_disease",
    "fertilizer_agent": "last_fertilizer",
    "pest_agent": "last_pest",
    "soil_agent": "last_soil_note",
}


@dataclass
class FarmerMemory:
    """
    Everything AgriNova AI remembers about ONE farmer's ongoing
    conversation (identified by `session_id`).

    Structured facts (`crop`, `location`, `latitude`, `longitude`,
    `field`) are merged straight into `AgentRequest.context` for the
    NEXT question, so every existing agent that already reads
    `context.get("crop")` / `context.get("location")` / etc. (Market
    Agent, Weather Agent — see market_agent.py / tools/weather_tool.py)
    benefits automatically, with no changes needed to those agents.

    `last_disease` / `last_fertilizer` / `last_pest` / `last_soil_note`
    and `weather_history` are richer than a single context key, so
    they're instead rendered into `to_prompt_block()` for agents/the
    Report Agent to read as prose context (mirrors
    `agent_types.format_prior_findings`'s "findings from teammates"
    block, just spanning turns instead of agents).
    """

    session_id: str
    crop: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    field_name: Optional[str] = None

    last_disease: Optional[Dict[str, str]] = None
    last_fertilizer: Optional[Dict[str, str]] = None
    last_pest: Optional[Dict[str, str]] = None
    last_soil_note: Optional[Dict[str, str]] = None
    weather_history: List[Dict[str, str]] = dc_field(default_factory=list)

    # Short rolling log of past turns, for light conversational
    # continuity ("Recently asked: ..."). Capped at
    # agent_config.MEMORY_MAX_TURNS — this is NOT a full transcript
    # store, just enough for the "don't repeat yourself" experience.
    turns: List[Dict[str, Any]] = dc_field(default_factory=list)

    created_at: str = dc_field(default_factory=_now_iso)
    updated_at: str = dc_field(default_factory=_now_iso)

    # -- Serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FarmerMemory":
        known_fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known_fields})

    # -- What the orchestrator merges into AgentRequest.context ---------------
    def known_context(self) -> Dict[str, Any]:
        """Only the plain structured facts, ready to spread into
        `context` for the next question — never overrides what the
        CALLER explicitly passes for that question (see
        agent_orchestrator.handle: `{**recalled, **explicit_context}`)."""
        ctx: Dict[str, Any] = {}
        if self.crop:
            ctx["crop"] = self.crop
        if self.location:
            ctx["location"] = self.location
        if self.latitude is not None and self.longitude is not None:
            ctx["latitude"] = self.latitude
            ctx["longitude"] = self.longitude
        if self.field_name:
            ctx["field"] = self.field_name
        return ctx

    def is_empty(self) -> bool:
        return not (
            self.crop or self.location or self.field_name
            or self.last_disease or self.last_fertilizer
            or self.last_pest or self.last_soil_note
            or self.weather_history or self.turns
        )

    # -- What agents/Report Agent read as prose context -----------------------
    def to_prompt_block(self) -> str:
        """Renders known facts as a short prompt block, the Phase 11
        counterpart to `agent_types.format_prior_findings`. Returns ""
        when nothing is known yet (a brand-new session), so every
        agent's prompt is byte-for-byte unchanged from Phase 10 in
        that case."""
        if self.is_empty():
            return ""

        lines: List[str] = []
        if self.crop:
            lines.append(f"- Crop: {self.crop}")
        if self.location:
            lines.append(f"- Location: {self.location}")
        if self.field_name:
            lines.append(f"- Field / plot: {self.field_name}")
        if self.last_disease:
            lines.append(
                f"- Previous disease discussed ({self.last_disease.get('date', 'earlier')}): "
                f"{self.last_disease.get('summary', '')}"
            )
        if self.last_pest:
            lines.append(
                f"- Previous pest issue discussed ({self.last_pest.get('date', 'earlier')}): "
                f"{self.last_pest.get('summary', '')}"
            )
        if self.last_fertilizer:
            lines.append(
                f"- Previous fertilizer guidance given ({self.last_fertilizer.get('date', 'earlier')}): "
                f"{self.last_fertilizer.get('summary', '')}"
            )
        if self.last_soil_note:
            lines.append(
                f"- Previous soil note ({self.last_soil_note.get('date', 'earlier')}): "
                f"{self.last_soil_note.get('summary', '')}"
            )
        if self.weather_history:
            latest = self.weather_history[-1]
            lines.append(
                f"- Weather last checked ({latest.get('date', 'earlier')}): {latest.get('summary', '')}"
            )
        recent_turns = self.turns[-3:]
        if recent_turns:
            lines.append("- Recently asked:")
            for t in recent_turns:
                lines.append(f"  - \"{t.get('question', '')}\" ({t.get('date', 'earlier')})")

        header = (
            "WHAT'S ALREADY KNOWN ABOUT THIS FARMER FROM EARLIER TURNS IN THIS "
            "CONVERSATION (do not ask the farmer to repeat any of this — only ask "
            "again if it seems to have changed, is missing, or the farmer's current "
            "question contradicts it):"
        )
        return header + "\n" + "\n".join(lines) + "\n\n"


class ConversationMemoryStore:
    """
    File-backed store for `FarmerMemory`, one small JSON file per
    session under `agent_config.MEMORY_DIR`. Deliberately simple
    (no database server) — every caller (agent_orchestrator.py,
    api.py, main.py) only ever talks to this class, so the storage
    backend can change later without touching any of them.
    """

    _SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory or agent_config.MEMORY_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)
        # One RE-ENTRANT lock for the whole store — sessions are
        # lightweight and request volume for a demo/portfolio project
        # is low, so a single lock keeps this file simple instead of
        # managing a lock per session_id. Must be reentrant because
        # `record_turn()` holds the lock for its whole read-modify-write
        # cycle and calls `self.get()` (which also locks) inside it.
        self._lock = threading.RLock()

    def _sanitize_session_id(self, session_id: str) -> str:
        session_id = (session_id or "").strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string.")
        return self._SAFE_ID_PATTERN.sub("_", session_id)[:200]

    def _path_for(self, session_id: str) -> Path:
        return self.directory / f"{self._sanitize_session_id(session_id)}.json"

    def get(self, session_id: str) -> FarmerMemory:
        """Loads a session's memory, or returns a fresh, empty one if
        this is the farmer's first turn (or the file is unreadable —
        a corrupt memory file should never take down a request)."""
        path = self._path_for(session_id)
        if not path.exists():
            return FarmerMemory(session_id=session_id)
        try:
            with self._lock:
                raw = json.loads(path.read_text(encoding="utf-8"))
            return FarmerMemory.from_dict(raw)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(f"Conversation memory for session '{session_id}' unreadable ({e}); starting fresh.")
            return FarmerMemory(session_id=session_id)

    def save(self, memory: FarmerMemory) -> None:
        path = self._path_for(memory.session_id)
        with self._lock:
            path.write_text(json.dumps(memory.to_dict(), indent=2), encoding="utf-8")

    def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        with self._lock:
            if path.exists():
                path.unlink()
                return True
            return False

    def list_sessions(self) -> List[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))

    # -- The main entry point: fold one finished turn into memory -------------
    def record_turn(
        self,
        session_id: str,
        question: str,
        context_hints: Dict[str, Any],
        agent_results: List["AgentResult"],
        final_report: Optional["AgentResult"] = None,
    ) -> FarmerMemory:
        """
        Called once per question, AFTER the orchestrator has a final
        answer. Extracts whatever new structured facts this turn
        revealed and appends them to the session's memory:

          - crop / location / lat+lon / field: explicit `context_hints`
            win; crop also falls back to keyword-matching `question`
            (see `extract_crop_from_text`) so a crop mentioned only in
            plain conversation still gets remembered.
          - Weather Agent's RESOLVED location (from real geocoding,
            see tools/weather_tool.py) fills in location/lat/lon if
            nothing more explicit is already on file — geocoded output
            is more reliable than guessing from free text.
          - Disease / Pest / Fertilizer / Soil Agent findings become
            `last_<domain>`, so the NEXT turn's memory block can say
            "since your tomatoes had early blight recently...".
          - A short rolling log of past questions for light
            conversational continuity.
        """
        with self._lock:
            memory = self.get(session_id)
            now = _now_iso()
            context_hints = context_hints or {}

            crop = context_hints.get("crop") or extract_crop_from_text(question)
            if crop:
                memory.crop = crop

            location = context_hints.get("location")
            if location:
                memory.location = location

            lat, lon = context_hints.get("latitude"), context_hints.get("longitude")
            if lat is not None and lon is not None:
                try:
                    memory.latitude, memory.longitude = float(lat), float(lon)
                except (TypeError, ValueError):
                    pass

            field_name = context_hints.get("field")
            if field_name:
                memory.field_name = field_name

            for result in agent_results or []:
                if result.error:
                    continue

                remembered_key = _REMEMBERED_FINDING_AGENTS.get(result.agent_name)
                if remembered_key:
                    summary = (result.summary or (result.details or "")[:200]).strip()
                    if summary:
                        setattr(memory, remembered_key, {"summary": summary, "date": now})
                    continue

                if result.agent_name == "weather_agent" and result.grounded:
                    weather_location = result.data.get("location")
                    weather_lat = result.data.get("latitude")
                    weather_lon = result.data.get("longitude")
                    # Only fill gaps — an explicit farmer-given location/
                    # coordinates for THIS turn already took precedence
                    # above, and should not be overwritten by whatever
                    # the Weather Agent defaulted to.
                    if weather_location and not memory.location:
                        memory.location = weather_location
                    if weather_lat is not None and weather_lon is not None and memory.latitude is None:
                        memory.latitude, memory.longitude = weather_lat, weather_lon
                    summary = (result.summary or "").strip()
                    if summary:
                        memory.weather_history.append({"summary": summary, "date": now})
                        memory.weather_history = memory.weather_history[-agent_config.MEMORY_MAX_WEATHER_HISTORY:]

            turn_entry: Dict[str, Any] = {
                "question": question,
                "date": now,
                "agents_run": [r.agent_name for r in (agent_results or [])],
            }
            if final_report is not None:
                turn_entry["final_summary"] = final_report.summary
            memory.turns.append(turn_entry)
            memory.turns = memory.turns[-agent_config.MEMORY_MAX_TURNS:]

            memory.updated_at = now
            self._write_locked(memory)
            return memory

    def _write_locked(self, memory: FarmerMemory) -> None:
        """Same as `save()`, but assumes the caller already holds
        `self._lock` (used internally by `record_turn` to keep the
        read-modify-write cycle atomic w.r.t. other threads)."""
        path = self._path_for(memory.session_id)
        path.write_text(json.dumps(memory.to_dict(), indent=2), encoding="utf-8")
