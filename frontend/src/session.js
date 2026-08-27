/**
 * session.js
 * ==========
 * A "session" here is the same concept the backend's Phase 11
 * conversation memory uses: an opaque `session_id` string that ties a
 * farmer's questions together so AgriNova AI doesn't ask them to
 * repeat their crop/location/previous findings every turn (see
 * backend/Agents-Pipeline/conversation_memory.py).
 *
 * This module owns:
 *   - generating/reading/switching the ACTIVE session id
 *   - a small registry of every session this browser has used, so the
 *     "Session" switcher in the top bar can list them by name
 *   - per-session chat history, kept client-side (localStorage) purely
 *     for a snappy UI — the SOURCE OF TRUTH for what AgriNova AI
 *     actually remembers is always the backend (GET /api/memory/:id),
 *     never this cache.
 */

const ACTIVE_SESSION_KEY = "agrinova.activeSessionId";
const SESSION_REGISTRY_KEY = "agrinova.sessions"; // { [id]: { label, createdAt } }
const HISTORY_KEY_PREFIX = "agrinova.history."; // + sessionId
const MAX_HISTORY_PER_SESSION = 50;

function safeParse(raw, fallback) {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function randomId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID().slice(0, 8);
  }
  return Math.random().toString(36).slice(2, 10);
}

export function getSessionRegistry() {
  return safeParse(localStorage.getItem(SESSION_REGISTRY_KEY), {});
}

function saveSessionRegistry(registry) {
  localStorage.setItem(SESSION_REGISTRY_KEY, JSON.stringify(registry));
}

export function registerSession(id, label) {
  const registry = getSessionRegistry();
  if (!registry[id]) {
    registry[id] = { label: label || id, createdAt: new Date().toISOString() };
    saveSessionRegistry(registry);
  }
  return registry;
}

export function renameSession(id, label) {
  const registry = getSessionRegistry();
  if (registry[id]) {
    registry[id] = { ...registry[id], label };
    saveSessionRegistry(registry);
  }
  return registry;
}

export function forgetSessionLocally(id) {
  const registry = getSessionRegistry();
  delete registry[id];
  saveSessionRegistry(registry);
  localStorage.removeItem(HISTORY_KEY_PREFIX + id);
}

export function createSession(label) {
  const id = `farmer-${randomId()}`;
  registerSession(id, label);
  return id;
}

export function getActiveSessionId() {
  let id = localStorage.getItem(ACTIVE_SESSION_KEY);
  if (!id) {
    id = createSession("My farm");
    setActiveSessionId(id);
  } else {
    registerSession(id); // no-op if already registered
  }
  return id;
}

export function setActiveSessionId(id) {
  localStorage.setItem(ACTIVE_SESSION_KEY, id);
}

// -- Per-session chat history (client-side cache only) --------------------

export function getHistory(sessionId) {
  return safeParse(localStorage.getItem(HISTORY_KEY_PREFIX + sessionId), []);
}

export function appendHistory(sessionId, turn) {
  const history = getHistory(sessionId);
  const next = [...history, turn].slice(-MAX_HISTORY_PER_SESSION);
  localStorage.setItem(HISTORY_KEY_PREFIX + sessionId, JSON.stringify(next));
  return next;
}

export function clearHistory(sessionId) {
  localStorage.removeItem(HISTORY_KEY_PREFIX + sessionId);
}
