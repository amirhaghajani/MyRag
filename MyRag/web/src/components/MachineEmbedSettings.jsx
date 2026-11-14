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

const modelMissing = !configured.some((m) => m.id === machine.embed_model_id);
  const current = configured.find((m) => m.id === machine.embed_model_id);
  const selected = configured.find((m) => m.id === modelId);

  async function apply() {
    if (!modelId || modelId === machine.embed_model_id) return;
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
        {machine.embed_model_id} (dim {machine.embed_dim}) · {machine.chunk_count} chunks ·
        server default {models?.default_embed_model} (dim {serverDim ?? "?"})
      </p>

      {modelMissing && (
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
            modelId === machine.embed_model_id
          }
        >
          {busy
            ? "Re-embedding…"
            : machine.chunk_count > 0
              ? "Change and re-embed"
              : "Use for new documents"}
        </button>
      </div>

      {selected && selected.dim !== current?.dim && current && (
        <p className="cost" style={{ marginTop: 6 }}>
          Switching to dim {selected.dim} re-embeds this machine's chunks into a different
          vector column; other machines keep theirs.
        </p>
      )}
      {error && <div className="error" style={{ marginTop: 6 }}>{error}</div>}
    </div>
  );
}