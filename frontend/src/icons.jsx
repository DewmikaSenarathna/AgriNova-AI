import React from "react";

const base = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export const MicIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
    <path d="M9 21h6" />
  </svg>
);

export const MicOffIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
    <path d="M9 21h6" />
    <line x1="3" y1="3" x2="21" y2="21" />
  </svg>
);

export const LeafClipIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M4 13c0-6 5-9 12-9-1 8-4 13-11 13" />
    <path d="M4 13c2 0 3-1 4-3" />
  </svg>
);

export const SendIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M3.5 12 20 4l-6 16-3-7-7-1Z" />
  </svg>
);

export const CloseIcon = (props) => (
  <svg {...base} {...props}>
    <line x1="5" y1="5" x2="19" y2="19" />
    <line x1="19" y1="5" x2="5" y2="19" />
  </svg>
);

export const ChevronDownIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M6 9l6 6 6-6" />
  </svg>
);

export const SunIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
  </svg>
);

export const CloudIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M7 18a4.5 4.5 0 0 1-.5-8.98A5.5 5.5 0 0 1 17 8.5a4 4 0 0 1-1 7.9H7Z" />
  </svg>
);

export const RainIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M7 15a4.5 4.5 0 0 1-.5-8.98A5.5 5.5 0 0 1 17 5.5a4 4 0 0 1-1 7.9H7Z" />
    <path d="M8 19l-1 2M12 19l-1 2M16 19l-1 2" />
  </svg>
);

export const StormIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M7 13a4.5 4.5 0 0 1-.5-8.98A5.5 5.5 0 0 1 17 3.5a4 4 0 0 1-1 7.9H7Z" />
    <path d="M12 14l-2 4h3l-2 4" />
  </svg>
);

export const DropletIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z" />
  </svg>
);

export const WindIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M3 8h11a3 3 0 1 0-3-3" />
    <path d="M3 16h13a3 3 0 1 1-3 3" />
  </svg>
);

export const TagIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3h6a1 1 0 0 1 1 1v6l-9 9-8-8 9-9Z" />
    <circle cx="15.5" cy="7.5" r="1.2" fill="currentColor" stroke="none" />
  </svg>
);

export const ClipboardIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="6" y="4" width="12" height="17" rx="2" />
    <path d="M9 4V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1" />
    <path d="M9 10h6M9 14h6M9 18h3" />
  </svg>
);

export const HistoryIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M3 12a9 9 0 1 0 3-6.7" />
    <path d="M3 4v5h5" />
    <path d="M12 8v4l3 2" />
  </svg>
);

export const SparkIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M17.5 17.5 6 6M17.5 6.5 15 9" />
  </svg>
);

export const ChevronRightIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M9 6l6 6-6 6" />
  </svg>
);

export const PlusIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const TrashIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13" />
  </svg>
);

export const CopyIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h10" />
  </svg>
);

export function conditionIcon(condition = "") {
  const c = condition.toLowerCase();
  if (c.includes("thunder") || c.includes("hail")) return StormIcon;
  if (c.includes("rain") || c.includes("drizzle") || c.includes("shower")) return RainIcon;
  if (c.includes("cloud") || c.includes("overcast") || c.includes("fog")) return CloudIcon;
  return SunIcon;
}
