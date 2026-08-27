import React, { useEffect, useRef, useState } from "react";
import { ChevronDownIcon, PlusIcon, TrashIcon, CopyIcon } from "../icons.jsx";

export default function SessionSwitcher({
  sessionId,
  sessions,
  onSwitch,
  onCreate,
  onForget,
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    function onClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const entries = Object.entries(sessions);
  const currentLabel = sessions[sessionId]?.label || sessionId;

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API may be unavailable (e.g. insecure context) — the
      // session id is still visible in the menu for manual copying.
    }
  };

  return (
    <div className="session-pill" ref={ref}>
      <button
        type="button"
        className="session-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="true"
        title="Switch farmer / conversation"
      >
        {currentLabel}
        <ChevronDownIcon width={13} height={13} />
      </button>

      {open && (
        <div className="session-menu" role="menu">
          <div className="eyebrow">Conversation</div>
          <div className="session-menu-list">
            {entries.map(([id, meta]) => (
              <button
                key={id}
                type="button"
                role="menuitemradio"
                aria-checked={id === sessionId}
                className={`session-menu-item${id === sessionId ? " is-active" : ""}`}
                onClick={() => {
                  onSwitch(id);
                  setOpen(false);
                }}
              >
                <span>{meta.label || id}</span>
                <span className="id">{id}</span>
              </button>
            ))}
          </div>

          <div className="session-menu-actions">
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                onCreate();
                setOpen(false);
              }}
            >
              <PlusIcon width={14} height={14} />
              New
            </button>
            <button type="button" className="btn btn-ghost" onClick={copyId}>
              <CopyIcon width={14} height={14} />
              {copied ? "Copied" : "Copy ID"}
            </button>
            {entries.length > 1 && (
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => {
                  onForget(sessionId);
                  setOpen(false);
                }}
              >
                <TrashIcon width={14} height={14} />
                Forget
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
