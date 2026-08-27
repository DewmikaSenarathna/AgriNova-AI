import React, { useCallback, useEffect, useRef, useState } from "react";

import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import Composer from "./components/Composer.jsx";
import RecommendationLedger from "./components/RecommendationLedger.jsx";
import WeatherCard from "./components/WeatherCard.jsx";
import MarketCard from "./components/MarketCard.jsx";
import MemoryPanel from "./components/MemoryPanel.jsx";
import ChatHistory from "./components/ChatHistory.jsx";
import EmptyState from "./components/EmptyState.jsx";
import ErrorBanner from "./components/ErrorBanner.jsx";
import Footer from "./components/Footer.jsx";

import { askAgents, getAgentsHealth, getRagHealth, getMemory, ApiError } from "./api.js";
import {
  getActiveSessionId,
  setActiveSessionId,
  getSessionRegistry,
  createSession,
  forgetSessionLocally,
  getHistory,
  appendHistory,
  clearHistory,
} from "./session.js";

const HEALTH_POLL_MS = 20000;

export default function App() {
  const [sessionId, setSessionId] = useState(getActiveSessionId);
  const [sessions, setSessions] = useState(getSessionRegistry);
  const [history, setHistory] = useState(() => getHistory(sessionId));

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [memory, setMemory] = useState(null);
  const [agentsStatus, setAgentsStatus] = useState("checking");
  const [ragStatus, setRagStatus] = useState("checking");

  const [activeSection, setActiveSection] = useState("ask");

  const sectionRefs = {
    ask: useRef(null),
    recommendation: useRef(null),
    weather: useRef(null),
    market: useRef(null),
    history: useRef(null),
  };

  // -- Health polling ------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const health = await getAgentsHealth();
        if (!cancelled) setAgentsStatus(health.status === "ok" ? "ok" : "degraded");
      } catch {
        if (!cancelled) setAgentsStatus("down");
      }
      try {
        const health = await getRagHealth();
        if (!cancelled) setRagStatus(health.status === "ok" ? "ok" : "degraded");
      } catch {
        if (!cancelled) setRagStatus("down");
      }
    }

    poll();
    const timer = setInterval(poll, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // -- Memory recall, refreshed whenever the session changes --------------
  const refreshMemory = useCallback(async (id) => {
    try {
      const mem = await getMemory(id);
      setMemory(mem);
    } catch {
      setMemory(null);
    }
  }, []);

  useEffect(() => {
    setHistory(getHistory(sessionId));
    setResult(null);
    refreshMemory(sessionId);
  }, [sessionId, refreshMemory]);

  // -- Ask -------------------------------------------------------------------
  const handleAsk = useCallback(
    async (question, imageBase64) => {
      setLoading(true);
      setError("");
      try {
        const response = await askAgents({ question, sessionId, imageBase64 });
        setResult(response);
        const turn = {
          id: `${Date.now()}`,
          question,
          askedAt: new Date().toISOString(),
          result: response,
        };
        setHistory(appendHistory(sessionId, turn));
        refreshMemory(sessionId);
        sectionRefs.recommendation.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.message
            : "Something went wrong reaching AgriNova AI.";
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [sessionId, refreshMemory]
  );

  // -- Session management ----------------------------------------------------
  const handleSwitchSession = (id) => {
    setActiveSessionId(id);
    setSessionId(id);
  };

  const handleCreateSession = () => {
    const id = createSession(`Farm ${Object.keys(sessions).length + 1}`);
    setSessions(getSessionRegistry());
    handleSwitchSession(id);
  };

  const handleForgetSession = (id) => {
    forgetSessionLocally(id);
    const updated = getSessionRegistry();
    setSessions(updated);
    const remaining = Object.keys(updated);
    if (remaining.length === 0) {
      handleCreateSession();
    } else if (id === sessionId) {
      handleSwitchSession(remaining[0]);
    }
  };

  const handleClearHistory = () => {
    clearHistory(sessionId);
    setHistory([]);
  };

  const scrollToSection = (id) => {
    setActiveSection(id);
    sectionRefs[id]?.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="app-shell">
      <Sidebar activeSection={activeSection} onNavigate={scrollToSection} />

      <div className="main">
        <TopBar
          section={activeSection}
          agentsStatus={agentsStatus}
          ragStatus={ragStatus}
          sessionId={sessionId}
          sessions={sessions}
          onSwitchSession={handleSwitchSession}
          onCreateSession={handleCreateSession}
          onForgetSession={handleForgetSession}
        />

        <div className="content">
          <section ref={sectionRefs.ask} id="ask">
            <Composer onSubmit={handleAsk} busy={loading} disabled={agentsStatus === "down"} />
            <div style={{ marginTop: 12 }}>
              <ErrorBanner message={error} />
            </div>
          </section>

          {!result && !loading && <EmptyState />}

          {(result || loading) && (
            <div className="result-grid">
              <section ref={sectionRefs.recommendation} id="recommendation">
                <RecommendationLedger result={result} loading={loading} />
              </section>

              <div className="context-stack">
                <section ref={sectionRefs.weather} id="weather">
                  <WeatherCard agentResults={result?.agent_results} />
                </section>
                <section ref={sectionRefs.market} id="market">
                  <MarketCard agentResults={result?.agent_results} />
                </section>
                <MemoryPanel memory={memory} />
              </div>
            </div>
          )}

          <section ref={sectionRefs.history} id="history">
            <ChatHistory
              history={history}
              onAskAgain={(question) => handleAsk(question, null)}
              onClear={handleClearHistory}
            />
          </section>
        </div>

        <Footer />
      </div>
    </div>
  );
}
