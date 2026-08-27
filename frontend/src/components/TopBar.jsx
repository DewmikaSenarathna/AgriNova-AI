import React from "react";
import SessionSwitcher from "./SessionSwitcher.jsx";

const TITLES = {
  ask: "Dashboard",
  recommendation: "Recommendation",
  weather: "Weather",
  market: "Market prices",
  history: "Chat history",
};

function StatusDot({ label, status }) {
  // status: "ok" | "degraded" | "down" | "checking"
  const cls = status === "checking" ? "" : status;
  return (
    <span className={`status-dot ${cls}`} title={`${label}: ${status}`}>
      <span className="dot" />
      {label}
    </span>
  );
}

export default function TopBar({
  section,
  agentsStatus,
  ragStatus,
  sessionId,
  sessions,
  onSwitchSession,
  onCreateSession,
  onForgetSession,
}) {
  return (
    <header className="topbar">
      <div className="topbar-title">{TITLES[section] || "Dashboard"}</div>
      <div className="topbar-right">
        <div className="status-group">
          <StatusDot label="Agents API" status={agentsStatus} />
          <StatusDot label="RAG API" status={ragStatus} />
        </div>
        <SessionSwitcher
          sessionId={sessionId}
          sessions={sessions}
          onSwitch={onSwitchSession}
          onCreate={onCreateSession}
          onForget={onForgetSession}
        />
      </div>
    </header>
  );
}
