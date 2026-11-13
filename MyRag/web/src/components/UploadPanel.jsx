import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function UploadPanel({ machineId, onChanged }) {
  const [documents, setDocuments] = useState([]);
  const [busy, setBusy] = useState(false);
  const [retrying, setRetrying] = useState(null);
  const [error, setError] = useState("");

  async function refresh() {
    if (!machineId) return;
    setDocuments(await api.listDocuments(machineId));
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [machineId]);

  const indexing = documents.some((d) => d.status === "indexing");

  useEffect(() => {
    if (!indexing) return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexing, machineId]);

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const doc = await api.uploadDocument(machineId, file);
      await api.indexDocument(doc.id);
      await refresh();
      onChanged();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function remove(id) {
    await api.deleteDocument(id);
    await refresh();
    onChanged();
  }

  async function retry(doc) {
    setRetrying(doc.id);
    setError("");
    try {
      await api.indexDocument(doc.id);
      await refresh();
      onChanged();
    } catch (err) {
      setError(err.message);
    } finally {
      setRetrying(null);
    }
  }

  return (
    <div className="card">
      <h2>Machine data</h2>
      <div className="row">
        <input
          type="file"
          accept=".pdf,.txt,application/pdf,text/plain"
          onChange={upload}
          disabled={busy || !machineId}
        />
        {busy && <span className="muted">Uploading and indexing…</span>}
      </div>
      {error && <div className="error">{error}</div>}

      {documents.length > 0 && (
        <table style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>File</th>
              <th>Status</th>
              <th>Chunks</th>
              <th>Tokens</th>
              <th>Cost</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.id}>
                <td>{doc.filename}</td>
                <td>
                  <span
                    className={`badge ${
                      doc.status === "ready"
                        ? "ok"
                        : doc.status === "failed"
                          ? "err"
                          : doc.status === "indexing"
                            ? "warn"
                            : ""
                    }`}
                  >
                    {doc.status}
                  </span>
                </td>
                <td>{doc.chunk_count}</td>
                <td>{doc.token_estimate}</td>
                <td>${Number(doc.embed_cost_usd || 0).toFixed(6)}</td>
                <td>
                  {doc.status === "failed" && (
                    <button
                      onClick={() => retry(doc)}
                      disabled={retrying === doc.id}
                      title={doc.error || "Re-index from the stored file"}
                    >
                      {retrying === doc.id ? "…" : "Retry"}
                    </button>
                  )}{" "}
                  <button onClick={() => remove(doc.id)}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {documents
        .filter((d) => d.status === "failed" && d.error)
        .map((d) => (
          <div className="error" key={`err-${d.id}`}>
            {d.filename}: {d.error}
          </div>
        ))}

      {documents.some((d) => d.status === "indexing") && (
        <button style={{ marginTop: 8 }} onClick={refresh}>
          Refresh status
        </button>
      )}
    </div>
  );
}