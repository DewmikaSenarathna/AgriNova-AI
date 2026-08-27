import React from "react";
import { TagIcon } from "../icons.jsx";

export default function MarketCard({ agentResults }) {
  const market = (agentResults || []).find(
    (r) => r.agent_name === "market_agent" && typeof r.data?.average === "number"
  );

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">
          <TagIcon width={16} height={16} />
          Market prices
        </div>
      </div>

      {!market ? (
        <p className="card-empty">
          Ask about selling your crop: "what's today's price for coconuts?"
          and the low / average / high range shows up here.
        </p>
      ) : (
        <MarketBody data={market.data} />
      )}
    </div>
  );
}

function MarketBody({ data }) {
  const { crop, currency, low, average, high } = data;
  const span = Math.max(high - low, 1);
  const avgPct = Math.min(100, Math.max(0, ((average - low) / span) * 100));

  return (
    <>
      <div className="market-crop">
        <span className="market-crop-name">{crop}</span>
        <span className="market-currency">{currency} / kg</span>
      </div>

      <div className="market-range">
        <span className="avg-marker" style={{ left: `calc(${avgPct}% - 1.5px)` }} />
      </div>
      <div className="market-figures">
        <span>Low {low}</span>
        <span className="avg">Avg {average}</span>
        <span>High {high}</span>
      </div>

      <p className="market-note">
        Indicative figures - confirm same-day prices at your local market
        before deciding whether to sell now or hold.
      </p>
    </>
  );
}
