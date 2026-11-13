import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function MachineCreate({ models, onCreated }) {
  const [title, setTitle] = useState("");
  const [embedModelId, setEmbedModelId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!embedModelId && models?.default_embed_model) {
      setEmbedModelId(models.default_embed_model);
    }
  }, [models, embedModelId]);

  async function submit(event) {
    event.preventDefault();
    if (!title.trim()) return;

    setBusy(true);
    setError("");
    try {
      const machine = await api.createMachine({
        title: title.trim(),
        embed_model_id: embedModelId || undefined,
      });
      setTitle("");
      onCreated(machine);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card" onSubmit={submit}>
      <h2>New machine</h2>
      <div style={{ display: "grid", gap: 8 }}>
        <input
          placeholder="Machine title, e.g. Roller Mill #2"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <select value={embedModelId} onChange={(e) => setEmbedModelId(e.target.value)}>
          {(models?.embeddings || []).map((m) => (
            <option key={m.id} value={m.id}>
              {m.id} (dim {m.dim})
            </option>
          ))}
        </select>
        <button type="submit" disabled={busy || !title.trim()}>
          {busy ? "Creating…" : "Create machine"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
    </form>
  );
}