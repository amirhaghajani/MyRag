import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function MachineEmbedSettings({ machine, models, onChanged }) {
  const configured = models?.embeddings || [];
  const serverDim = models?.embed_dim;
  const [modelId, setModelId] = useState(machine.embed_model_id);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const list = models?.embeddings || [];
    const exists = list.some((m) => m.id === machine.embed_model_id);
    setModelId(exists ? machine.embed_model_id : "");
  }, [machine.id, machine.embed_model_id, models]);

  const mismatch = serverDim != null && machine.embed_dim !== serverDim;
  const modelMissing = !configured.some((m) => m.id === machine.embed_model_id);
  const selected = configured.find((m) => m.id === modelId);
  const selectionMatches = !selected || selected.dim === serverDim;

  async function apply() {
    if (!modelId || modelId === machine.embed_model_id || !selectionMatches) return;
    const reembed = machine.chunk_count > 0;
    const question = reembed
      ? `Re-embed all ${machine.chunk_count} chunks of "${machine.title}" with ${modelId}?\n\nThis calls the embedding API again and may cost money.`
      : `Use ${modelId} for new documents of "${machine.title}"?`;
    if (!window.confirm(question)) return;

    setBusy(true);
    setError("");
    try {
      await api.updateMachine(machine.id, { embed_model_id: modelId, reembed });
      onChanged();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Embedding</h2>
      <p className="cost">
        Server EMBED_DIM {serverDim ?? "?"} · {machine.embed_model_id} (dim{" "}
        {machine.embed_dim}) · {machine.chunk_count} chunks
      </p>

      {mismatch && (
        <div className="error" style={{ marginBottom: 8 }}>
          This machine uses dim {machine.embed_dim} but the server is configured for dim{" "}
          {serverDim}. Uploads and chat are blocked until you pick a {serverDim}-dim model and
          re-embed its documents.
        </div>
      )}

      {modelMissing && !mismatch && (
        <div className="error" style={{ marginBottom: 8 }}>
          The embed model “{machine.embed_model_id}” is no longer configured on the server.
          Pick one below and re-embed this machine's documents.
        </div>
      )}

      <div className="row" style={{ gap: 8 }}>
        <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
          <option value="" disabled>
            Choose a model…
          </option>
          {configured.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id} — dim {m.dim}
            </option>
          ))}
        </select>
        <button
          onClick={apply}
          disabled={
            busy ||
            !modelId ||
            modelId === machine.embed_model_id ||
            !selectionMatches
          }
        >
          {busy
            ? "Re-embedding…"
            : machine.chunk_count > 0
              ? "Change and re-embed"
              : "Use for new documents"}
        </button>
      </div>

      {selected && selected.dim !== serverDim && (
        <p className="cost" style={{ marginTop: 6 }}>
          This model's dim ({selected.dim}) does not match the server ({serverDim}). Change
          EMBED_DIM in .env and run the re-embed command first.
        </p>
      )}
      {error && <div className="error" style={{ marginTop: 6 }}>{error}</div>}
    </div>
  );
}