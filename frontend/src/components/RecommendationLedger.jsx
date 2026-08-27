import React, { useState } from "react";
import MarkdownLite from "../markdownLite.jsx";
import { ChevronDownIcon } from "../icons.jsx";
import { formatAgentName, truncate } from "../format.js";

function nodeStatus(agentResult, isReportNode, finalGrounded) {
  if (isReportNode) return finalGrounded ? "grounded" : "ungrounded";
  if (!agentResult) return "pending";
  if (agentResult.error) return "errored";
  return agentResult.grounded ? "grounded" : "ungrounded";
}

function GraftLine({ plan, agentResults, finalReport }) {
  const steps = plan?.steps?.length
    ? plan.steps
    : (plan?.agents_to_run || []).map((agent) => ({ need: null, agent, reason: null }));

  if (!steps.length) return null;

  return (
    <div className="graft-line" aria-label="Agent collaboration chain">
      {steps.map((step, idx) => {
        const isReportNode = step.agent === "report_agent";
        const agentResult = isReportNode
          ? null
          : agentResults.find((r) => r.agent_name === step.agent);
        const status = nodeStatus(agentResult, isReportNode, finalReport?.grounded);
        const summary = isReportNode ? finalReport?.summary : agentResult?.summary;

        return (
          <div className="graft-node" key={`${step.agent}-${idx}`}>
            <div className="graft-stem">
              <span className={`graft-dot ${status}`} />
            </div>
            <div className="graft-content">
              <div className="graft-agent">
                {isReportNode ? "Report Agent" : formatAgentName(step.agent)}
              </div>
              {step.need && <div className="graft-need">Need: {step.need}</div>}
              {summary && <div className="graft-summary">{truncate(summary, 160)}</div>}
              {agentResult?.error && (
                <div className="graft-summary" style={{ color: "var(--alert)" }}>
                  {agentResult.error}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function renderSource(source, idx) {
  const heading = source.heading || source.file_name || source.crop || source.source;
  const detail = source.text || source.location || source.url;
  const similarity =
    typeof source.similarity === "number" ? `${Math.round(source.similarity * 100)}%` : null;

  return (
    <div className="source-item" key={idx}>
      <div className="source-heading">
        <span>{heading || `Source ${idx + 1}`}</span>
        {similarity && <span className="source-sim">{similarity} match</span>}
      </div>
      {detail && <div>{truncate(String(detail), 200)}</div>}
      {source.agent && !heading && <div>via {formatAgentName(source.agent)}</div>}
    </div>
  );
}

export default function RecommendationLedger({ result, loading }) {
  const [showChain, setShowChain] = useState(true);
  const [showSources, setShowSources] = useState(false);

  if (loading) {
    return (
      <div className="ledger">
        <div className="thinking-row">
          <span className="thinking-dots">
            <span />
            <span />
            <span />
          </span>
          Planner is routing your question to the right specialists…
        </div>
      </div>
    );
  }

  if (!result) return null;

  const { question, plan, agent_results: agentResults = [], final_report: finalReport } = result;
  const sources = finalReport?.sources || [];

  return (
    <div className="ledger">
      <div className="ledger-head">
        <div className="ledger-question">
          You asked: <strong>{question}</strong>
        </div>
      </div>

      <div className="ledger-body">
        <div className="ledger-report">
          <MarkdownLite text={finalReport?.details || finalReport?.summary} />
        </div>
      </div>

      <div className="ledger-footer">
        <span className={`grounded-badge ${finalReport?.grounded ? "yes" : "no"}`}>
          <span className="dot" />
          {finalReport?.grounded ? "Grounded in sources" : "Not grounded — verify locally"}
        </span>

        <button
          type="button"
          className="sources-toggle"
          onClick={() => setShowChain((v) => !v)}
        >
          <ChevronDownIcon
            width={13}
            height={13}
            style={{ transform: showChain ? "rotate(180deg)" : "none" }}
          />
          {showChain ? "Hide" : "Show"} agent chain ({agentResults.length})
        </button>

        {sources.length > 0 && (
          <button
            type="button"
            className="sources-toggle"
            onClick={() => setShowSources((v) => !v)}
          >
            <ChevronDownIcon
              width={13}
              height={13}
              style={{ transform: showSources ? "rotate(180deg)" : "none" }}
            />
            {showSources ? "Hide" : "Show"} sources ({sources.length})
          </button>
        )}
      </div>

      {showChain && (
        <div style={{ padding: "0 20px 18px" }}>
          <GraftLine plan={plan} agentResults={agentResults} finalReport={finalReport} />
        </div>
      )}

      {showSources && (
        <div style={{ padding: "0 20px 20px" }}>
          <div className="sources-list">{sources.map(renderSource)}</div>
        </div>
      )}
    </div>
  );
}
