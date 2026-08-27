import React from "react";
import { SparkIcon } from "../icons.jsx";

export default function EmptyState() {
  return (
    <div className="empty-state">
      <SparkIcon />
      <h2>Ask AgriNova AI anything about your farm</h2>
      <p>
        Type, speak or attach a photo of a leaf. The Planner routes your
        question to the right specialists disease, weather, soil,
        fertilizer, pests, market prices or government schemes and
        combines their findings into one grounded recommendation.
      </p>
    </div>
  );
}
