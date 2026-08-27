/**
 * markdownLite.jsx
 * ================
 * The Report Agent (backend/Agents-Pipeline/report_agent.py) is asked
 * to write "short headings or a short list per topic" — plain-ish
 * markdown, not a full spec. Rather than pull in a markdown dependency
 * for a handful of patterns, this renders exactly what that prompt
 * actually produces: `## Heading` / `### Heading` lines, `- ` or `* `
 * bullet lists, `**bold**` spans, and blank-line-separated paragraphs.
 * Anything else is shown as plain text — never as raw, un-rendered
 * markdown syntax.
 */
import React from "react";

function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={`${keyPrefix}-b${i}`}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={`${keyPrefix}-t${i}`}>{part}</React.Fragment>;
  });
}

export default function MarkdownLite({ text }) {
  const source = (text || "").replace(/\r\n/g, "\n").trim();
  if (!source) return null;

  const lines = source.split("\n");
  const blocks = [];
  let listBuffer = [];

  const flushList = (key) => {
    if (listBuffer.length) {
      blocks.push(
        <ul className="md-list" key={`ul-${key}`}>
          {listBuffer.map((item, i) => (
            <li key={i}>{renderInline(item, `li-${key}-${i}`)}</li>
          ))}
        </ul>
      );
      listBuffer = [];
    }
  };

  lines.forEach((rawLine, idx) => {
    const line = rawLine.trim();

    if (!line) {
      flushList(idx);
      return;
    }

    const heading = line.match(/^(#{2,4})\s+(.*)$/);
    if (heading) {
      flushList(idx);
      const level = heading[1].length; // 2, 3 or 4
      const Tag = level === 2 ? "h3" : level === 3 ? "h4" : "h5";
      blocks.push(
        <Tag className="md-heading" key={`h-${idx}`}>
          {renderInline(heading[2], `h-${idx}`)}
        </Tag>
      );
      return;
    }

    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      listBuffer.push(bullet[1]);
      return;
    }

    flushList(idx);
    blocks.push(
      <p className="md-p" key={`p-${idx}`}>
        {renderInline(line, `p-${idx}`)}
      </p>
    );
  });
  flushList("end");

  return <div className="md-lite">{blocks}</div>;
}
