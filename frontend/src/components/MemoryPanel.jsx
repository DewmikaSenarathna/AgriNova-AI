import React from "react";
import { ClipboardIcon } from "../icons.jsx";
import { relativeTime } from "../format.js";

const FINDING_LABELS = {
  last_disease: "Disease",
  last_pest: "Pest",
  last_fertilizer: "Fertilizer",
  last_soil_note: "Soil",
};

export default function MemoryPanel({ memory }) {
  const hasChips = memory && (memory.crop || memory.location || memory.field_name);
  const findings = memory
    ? Object.entries(FINDING_LABELS).filter(([key]) => memory[key])
    : [];
  const hasAnything =
    hasChips || findings.length > 0 || (memory?.weather_history || []).length > 0;

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">
          <ClipboardIcon width={16} height={16} />
          Remembered about you
        </div>
      </div>

      {!hasAnything ? (
        <p className="card-empty">
          Nothing remembered yet for this session - mention your crop and
          location once, and AgriNova AI won't ask again (Phase 11
          conversation memory).
        </p>
      ) : (
        <>
          {hasChips && (
            <div className="memory-chips">
              {memory.crop && (
                <span className="memory-chip">
                  <span className="label">Crop</span>
                  {memory.crop}
                </span>
              )}
              {memory.location && (
                <span className="memory-chip">
                  <span className="label">Location</span>
                  {memory.location}
                </span>
              )}
              {memory.field_name && (
                <span className="memory-chip">
                  <span className="label">Field</span>
                  {memory.field_name}
                </span>
              )}
            </div>
          )}

          {(findings.length > 0 || memory.weather_history?.length > 0) && (
            <div className="memory-findings">
              {findings.map(([key, label]) => (
                <div className="memory-finding" key={key}>
                  <span className="mf-label">
                    {label} · {relativeTime(memory[key].date)}
                  </span>
                  {memory[key].summary}
                </div>
              ))}
              {memory.weather_history?.length > 0 && (
                <div className="memory-finding">
                  <span className="mf-label">
                    Weather last checked ·{" "}
                    {relativeTime(memory.weather_history[memory.weather_history.length - 1].date)}
                  </span>
                  {memory.weather_history[memory.weather_history.length - 1].summary}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
