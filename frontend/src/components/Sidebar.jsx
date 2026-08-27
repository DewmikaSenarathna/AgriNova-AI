import React from "react";
import { SparkIcon, SunIcon, TagIcon, ClipboardIcon, HistoryIcon } from "../icons.jsx";

const SECTIONS = [
  { id: "ask", label: "Ask a question", icon: SparkIcon },
  { id: "recommendation", label: "Recommendation", icon: ClipboardIcon },
  { id: "weather", label: "Weather", icon: SunIcon },
  { id: "market", label: "Market prices", icon: TagIcon },
  { id: "history", label: "Chat history", icon: HistoryIcon },
];

export default function Sidebar({ activeSection, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">
          <img src="/AgriNovaAI_logo.png" alt="AgriNovaAI logo" />
        </span>
        <div>
          <div className="brand-name">AgriNova AI</div>
          <div className="brand-tag">Field dashboard</div>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Dashboard sections">
        {SECTIONS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`sidebar-link${activeSection === id ? " is-active" : ""}`}
            onClick={() => onNavigate(id)}
          >
            <Icon width={16} height={16} />
            {label}
          </button>
        ))}
      </nav>

      <p className="sidebar-footer">
        Planner → specialist agents → Report Agent, with weather, market and
        memory built in.
      </p>
    </aside>
  );
}
