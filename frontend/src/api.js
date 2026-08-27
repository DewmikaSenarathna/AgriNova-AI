/**
 * api.js
 * ======
 * Every fetch call the dashboard makes, in one place — mirrors how the
 * backend keeps one choke point (rag_bridge.py) instead of scattering
 * HTTP calls through every component.
 *
 * Two backends:
 *   AGENTS_API_URL — backend/Agents-Pipeline/api.py (default :8001).
 *                     The main API: /api/agents/ask, /api/memory/*.
 *   RAG_API_URL     — backend/RAG-Pipeline/api.py (default :8000).
 *                     Only used for the top bar's "RAG online" status dot.
 */

export const AGENTS_API_URL = (
  import.meta.env.VITE_AGENTS_API_URL || "http://localhost:8001"
).replace(/\/$/, "");

export const RAG_API_URL = (
  import.meta.env.VITE_RAG_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, { status, cause } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.cause = cause;
  }
}

async function request(baseUrl, path, options = {}) {
  let response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new ApiError(
      `Couldn't reach ${baseUrl}. Is the backend running and CORS-enabled?`,
      { cause: err }
    );
  }

  let body = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  if (!response.ok) {
    const detail =
      (body && typeof body === "object" && (body.detail || body.error)) ||
      (typeof body === "string" && body) ||
      `Request failed (${response.status}).`;
    throw new ApiError(String(detail), { status: response.status });
  }

  return body;
}

// -- Health --------------------------------------------------------------

export function getAgentsHealth() {
  return request(AGENTS_API_URL, "/health");
}

export function getRagHealth() {
  return request(RAG_API_URL, "/health");
}

// -- The main agentic ask endpoint ----------------------------------------

/**
 * @param {object} params
 * @param {string} params.question
 * @param {string} [params.sessionId] — Phase 11 conversation memory key.
 * @param {object} [params.context] — optional hints, e.g. {crop, location}.
 * @param {string} [params.imageBase64] — optional attached photo.
 */
export function askAgents({ question, sessionId, context, imageBase64 }) {
  return request(AGENTS_API_URL, "/api/agents/ask", {
    method: "POST",
    body: JSON.stringify({
      question,
      session_id: sessionId || null,
      context: context || null,
      image_base64: imageBase64 || null,
    }),
  });
}

// -- Conversation memory (Phase 11) ---------------------------------------

export function getMemory(sessionId) {
  return request(AGENTS_API_URL, `/api/memory/${encodeURIComponent(sessionId)}`);
}

export function deleteMemory(sessionId) {
  return request(AGENTS_API_URL, `/api/memory/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

// -- Directory endpoints (used by the "how this works" panel) -------------

export function listAgents() {
  return request(AGENTS_API_URL, "/api/agents");
}

export function listTools() {
  return request(AGENTS_API_URL, "/api/tools");
}
