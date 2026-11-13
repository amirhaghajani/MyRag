import React from "react";
import { api } from "../api.js";

function money(value) {
  return `$${Number(value || 0).toFixed(6)}`;
}

export default function MachineList({ machines, selectedId, onSelect, onRefresh }) {
  async function remove(id) {
    if (!window.confirm("Delete this machine and all of its documents?")) return;
    await api.deleteMachine(id);
    onRefresh();
  }

  if (!machines.length) {
    return <p className="muted">No machines yet. Create one on the left.</p>;
  }

  return (
    <div>
      {machines.map((machine) => (
        <div
          key={machine.id}
          className={`machine-item ${machine.id === selectedId ? "active" : ""}`}
        >
          <div className="row" style={{ justifyContent: "space-between" }}>
            <button
              className="machine-item"
              style={{ border: "none", background: "transparent", padding: 0 }}
              onClick={() => onSelect(machine)}
            >
              <strong>{machine.title}</strong>
            </button>
            <button onClick={() => remove(machine.id)} title="Delete machine">
              ✕
            </button>
          </div>
          <div className="cost">
            {machine.document_count} docs · {machine.chunk_count} chunks ·{" "}
            {machine.query_count} queries · {money(machine.total_cost_usd)}
          </div>
          <div className="cost">embed: {machine.embed_model_id}</div>
        </div>
      ))}
    </div>
  );
}