# AgriNova AI — Frontend (Phase 12)

A React + Vite web dashboard for AgriNova AI — the interface described
in the Phase 12 brief:

```
Dashboard
  ↓
Ask Question   (text)
  ↓
Voice Input     (speech-to-text, in the same composer)
  ↓
Upload Leaf Image (attach a photo, in the same composer)
  ↓
Weather Card
  ↓
Market Prices
  ↓
Recommendations  (the Report Agent's consolidated, source-cited answer)
  ↓
Chat History
```

It's one scrollable dashboard, not five separate pages — the sidebar's
nav links smooth-scroll to each section, matching the flow above. Ask,
Voice Input and Upload Leaf Image are one composer (a farmer types,
speaks, or attaches a photo — or combines all three in one question),
because that's how the backend already treats them: one
`POST /api/agents/ask` call with an optional `image_base64` field.

## Run it

```bash
cd frontend
npm install
cp .env.example .env       # defaults already point at localhost:8001/:8000
npm run dev                 # http://localhost:5173
```

You'll need at least the **Agents-Pipeline API** running (the
dashboard's Ask/Weather/Market/Recommendations/Memory all go through
it):

```bash
cd ../backend/Agents-Pipeline
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```

The **RAG-Pipeline API** (`:8000`) is optional — the dashboard only
uses it for the "RAG API" status dot in the top bar.

```bash
npm run build      # production build -> dist/
npm run preview     # serve that build locally
npm run lint         # ESLint
```

## How each part of the brief maps to this code

| Brief | Where |
|---|---|
| Dashboard | `App.jsx` — the shell: sidebar, top bar, and the scroll flow below |
| Ask Question | `components/Composer.jsx` — textarea, Enter to send |
| Voice Input | `components/Composer.jsx` — the mic button uses the browser's `SpeechRecognition` API to transcribe into the same textarea (progressive enhancement: disabled with a clear message in browsers that don't support it, e.g. Firefox) |
| Upload Leaf Image | `components/Composer.jsx` — the leaf-clip button attaches a photo, previewed as a removable chip, sent as `image_base64` |
| Weather Card | `components/WeatherCard.jsx` — reads the Weather Agent's `AgentResult.data` (current conditions + short forecast) whenever it ran for the latest question |
| Market Prices | `components/MarketCard.jsx` — reads the Market Agent's `AgentResult.data` (low/average/high) whenever it ran |
| Recommendations | `components/RecommendationLedger.jsx` — the Report Agent's final, source-cited answer, plus the **graft line**: a visualization of the Planner's agent chain (see below) |
| Chat History | `components/ChatHistory.jsx` — every question asked in the current session, most recent first, expandable, cached in `localStorage` per session |

Two more pieces tie the whole thing to what the backend actually does:

* **`components/MemoryPanel.jsx`** — surfaces Phase 11 conversation
  memory (`GET /api/memory/:session_id`): crop, location, previous
  disease/fertilizer findings, recent weather. This is what lets
  "should I irrigate today?" be answered without repeating the crop
  and location from an earlier question.
* **Session switcher** (top bar) — a `session_id` is how the backend
  ties one farmer's questions together (see
  `backend/Agents-Pipeline/conversation_memory.py`). The dashboard
  generates one automatically, persists it in `localStorage`, and lets
  you create/switch/forget sessions — "forget" also clears that
  session's memory on the backend (`DELETE /api/memory/:session_id`).

## The "graft line"

The Recommendation card's signature visual is a small branching
connector down the left edge of the agent chain — not decoration, it
encodes the same thing the backend's Planner produces
(`plan.steps`, see `backend/Agents-Pipeline/planner_agent.py`): which
specialist ran, in what order, and whether its finding was grounded
(solid green node), ungrounded (hollow clay node), or errored (red
node) — mirroring Phase 10's sequential collaboration, where each
agent is shown building on the ones before it.

## Design system

`src/styles.css` — hand-written CSS variables, no UI framework. Named
"Paddy & Monsoon": paddy green (the crop), turmeric gold (a cash crop
and the accent color), monsoon indigo (text — the sky that decides
half of these farming decisions), and a warm rice-paper neutral
background chosen for daylight/outdoor legibility rather than a dark
"AI chat" theme. Type: **Fraunces** for headings, **Inter** for UI
text, **IBM Plex Mono** for data (temperatures, prices, timestamps,
session IDs).

## Project structure

```
frontend/
  index.html
  vite.config.js
  .env.example
  src/
    main.jsx              # React entry point
    App.jsx                # layout, state, orchestration
    api.js                  # every fetch call, one place
    session.js               # session id + local chat-history persistence
    markdownLite.jsx          # tiny renderer for the Report Agent's markdown-ish text
    format.js                  # relative time / name formatting helpers
    icons.jsx                   # small hand-authored SVG icon set
    styles.css                   # the whole design system
    components/
      Sidebar.jsx
      TopBar.jsx
      SessionSwitcher.jsx
      Composer.jsx                # Ask + Voice Input + Upload Leaf Image
      RecommendationLedger.jsx     # final report + graft line + sources
      WeatherCard.jsx
      MarketCard.jsx
      MemoryPanel.jsx               # Phase 11 conversation memory
      ChatHistory.jsx
      EmptyState.jsx
      ErrorBanner.jsx
```

## Notes / known limitations

* **Voice input** uses the Web Speech API (`SpeechRecognition` /
  `webkitSpeechRecognition`), which is Chrome/Edge-only today — Firefox
  and Safari fall back to a clear inline message rather than a silent
  no-op. Text and photo input always work everywhere.
* **Chat history** is cached in the browser (`localStorage`) per
  session for a snappy UI, but the backend's Phase 11 memory (crop,
  location, previous findings) is the actual source of truth for what
  AgriNova AI "remembers" — the Memory panel always reflects a fresh
  `GET /api/memory/:session_id` call, never the local cache.
* No build-time secrets are needed — both backend URLs are public,
  read-only API base URLs (`VITE_AGENTS_API_URL` / `VITE_RAG_API_URL`),
  safe to bake into a static build.
