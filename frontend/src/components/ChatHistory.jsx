import React, { useState } from "react";
import { HistoryIcon, ChevronRightIcon } from "../icons.jsx";
import { relativeTime, formatAgentName } from "../format.js";
import MarkdownLite from "../markdownLite.jsx";

function HistoryItem({ turn, onAskAgain }) {
  const [open, setOpen] = useState(false);
  const agentNames = turn.result?.agent_results?.map((r) => r.agent_name) || [];

  return (
    <div className={`history-item${open ? " is-open" : ""}`}>
      <button
        type="button"
        className="history-item-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <ChevronRightIcon width={14} height={14} className="history-chevron" />
        <span className="history-question">{turn.question}</span>
        <span className="history-time">{relativeTime(turn.askedAt)}</span>
      </button>

      {open && (
        <div className="history-item-body">
          {agentNames.length > 0 && (
            <div className="history-agents">
              {agentNames.map((name, i) => (
                <span className="history-agent-tag" key={`${name}-${i}`}>
                  {formatAgentName(name)}
                </span>
              ))}
            </div>
          )}
          <MarkdownLite
            text={turn.result?.final_report?.details || turn.result?.final_report?.summary}
          />
          <div style={{ marginTop: 10 }}>
            <button type="button" className="btn btn-ghost" onClick={() => onAskAgain(turn.question)}>
              Ask again
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ChatHistory({ history, onAskAgain, onClear }) {
  return (
    <div className="history-section">
      <div className="history-head">
        <div className="card-title">
          <HistoryIcon width={16} height={16} />
          Chat history
        </div>
        {history.length > 0 && (
          <button type="button" className="btn btn-ghost" onClick={onClear}>
            Clear
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <p className="card-empty">
          Questions you ask in this session will appear here, most recent
          first.
        </p>
      ) : (
        <div className="history-list">
          {[...history].reverse().map((turn) => (
            <HistoryItem key={turn.id} turn={turn} onAskAgain={onAskAgain} />
          ))}
        </div>
      )}
    </div>
  );
}
