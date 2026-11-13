import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function UsagePanel({ machineId }) {
  const [usage, setUsage] = useState(null);
  const [compare, setCompare] = useState([]);

  async function refresh() {
    if (!machineId) return;
    const [u, c] = await Promise.all([
      api.usage(machineId),
      api.compareWorkflows(machineId),
    ]);
    setUsage(u);
    setCompare(c);
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [machineId]);

  if (!usage) return null;

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Cost and quality by workflow</h2>
        <button onClick={refresh}>Refresh</button>
      </div>

      <p className="cost" style={{ marginTop: 8 }}>
        {usage.queries} queries · {usage.total_tokens} LLM tokens · $
        {Number(usage.cost_usd).toFixed(6)} · ingestion ${Number(usage.ingest.cost_usd).toFixed(6)}{" "}
        <span className="badge warn">{usage.ingest.cost_source}</span>
      </p>

      {compare.length === 0 ? (
        <p className="muted">Ask a question to start comparing flows.</p>
      ) : (
        <table style={{ marginTop: 8 }}>
          <thead>
            <tr>
              <th>Flow</th>
              <th>Runs</th>
              <th>Avg cost</th>
              <th>Avg tokens</th>
              <th>Avg latency</th>
              <th>Total cost</th>
            </tr>
          </thead>
          <tbody>
            {compare.map((row) => (
              <tr key={row.workflow_id}>
                <td>{row.workflow_name}</td>
                <td>{row.runs}</td>
                <td>${Number(row.avg_cost_usd).toFixed(6)}</td>
                <td>{Math.round(row.avg_total_tokens)}</td>
                <td>{Math.round(row.avg_latency_ms)} ms</td>
                <td>${Number(row.total_cost_usd).toFixed(6)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {usage.by_model.length > 0 && (
        <>
          <h3 style={{ fontSize: 13, margin: "14px 0 4px" }}>By model</h3>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Calls</th>
                <th>In</th>
                <th>Out</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {usage.by_model.map((m) => (
                <tr key={m.model_id}>
                  <td>{m.model_id}</td>
                  <td>{m.calls}</td>
                  <td>{m.prompt_tokens}</td>
                  <td>{m.completion_tokens}</td>
                  <td>${Number(m.cost_usd).toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}