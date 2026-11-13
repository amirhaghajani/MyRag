import React from "react";

export default function WorkflowPicker({ workflows, value, onChange }) {
  if (!workflows.length) {
    return <span className="muted">No workflows seeded.</span>;
  }

  return (
    <label className="row" style={{ gap: 6 }}>
      <span className="muted">Retrieval flow</span>
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {workflows
          .filter((w) => w.enabled)
          .map((w) => (
            <option key={w.id} value={w.id} title={w.description}>
              {w.name}
            </option>
          ))}
      </select>
    </label>
  );
}