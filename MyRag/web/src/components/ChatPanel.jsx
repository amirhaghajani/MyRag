import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { confirmReembed, embedIsStale, staleError } from "../reembed.js";

export default function ChatPanel({
  machineId,
  machine,
  models,
  workflows,
  workflowId,
  onWorkflowChange,
  onMachineUpdated,
}) {
  const [llmModelId, setLlmModelId] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [stream, setStream] = useState(null);
  const [trace, setTrace] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!llmModelId && models?.default_llm_model) {
      setLlmModelId(models.default_llm_model);
    }
  }, [models, llmModelId]);

  async function refresh() {
    if (!machineId) return;
    setMessages(await api.listMessages(machineId));
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [machineId]);

  async function streamAnswer(asked) {
    await api.chatStream(
      machineId,
      {
        question: asked,
        workflow_id: workflowId,
        llm_model_id: llmModelId,
      },
      (event) => {
        if (event.type === "status") {
          setStream((prev) => (prev ? { ...prev, stage: event.stage } : prev));
        } else if (event.type === "token") {
          setStream((prev) =>
            prev ? { ...prev, stage: "answering", answer: prev.answer + event.text } : prev
          );
        } else if (event.type === "reasoning") {
          setStream((prev) =>
            prev ? { ...prev, reasoning: prev.reasoning + event.text } : prev
          );
        }
      }
    );
    setQuestion("");
    await refresh();
    setStream(null);
  }

  async function ask(event) {
    event.preventDefault();
    if (!question.trim() || !machineId) return;

    setBusy(true);
    setError("");
    const asked = question.trim();
    setStream({ question: asked, answer: "", reasoning: "", stage: "retrieving" });
    try {
      if (embedIsStale(machine, models)) {
        const updated = await confirmReembed(machine, models);
        if (!updated) {
          setStream(null);
          return;
        }
        onMachineUpdated?.(updated);
      }
      try {
        await streamAnswer(asked);
      } catch (err) {
        if (!staleError(err)) throw err;
        const updated = await confirmReembed(machine, models);
        if (!updated) {
          setStream(null);
          return;
        }
        onMachineUpdated?.(updated);
        await streamAnswer(asked);
      }
    } catch (err) {
      setError(err.message);
      setStream(null);
    } finally {
      setBusy(false);
    }
  }

  async function toggleLog(message) {
    const id = message.query_id;
    if (trace && trace.queryId === id) {
      setTrace(null);
      return;
    }
    setTrace({ queryId: id, loading: true, error: "", events: null });
    try {
      const data = await api.queryLog(id);
      setTrace({ queryId: id, loading: false, error: "", events: data.events || [] });
    } catch (err) {
      setTrace({ queryId: id, loading: false, error: err.message, events: null });
    }
  }

  return (
    <div className="card">
      <h2>Chat with machine data</h2>

      <div className="row" style={{ marginBottom: 10 }}>
        <label className="row" style={{ gap: 6 }}>
          <span className="muted">Model</span>
          <select value={llmModelId} onChange={(e) => setLlmModelId(e.target.value)}>
            {(models?.llms || []).map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>
        </label>
        {workflowId ? null : <span className="badge err">select a flow</span>}
      </div>

      <form onSubmit={ask} className="row">
        <input
          style={{ flex: 1, minWidth: 240 }}
          placeholder="e.g. What is the rated capacity of this mill?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={busy || !machineId}
        />
        <button type="submit" disabled={busy || !question.trim() || !machineId}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>

      {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

      <div className="messages" style={{ marginTop: 14 }}>
        {stream && (
          <div className="msg stream">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <strong>{stream.question}</strong>
              <span className="cost">
                {llmModelId} ·{" "}
                {stream.stage === "retrieving"
                  ? "searching documents…"
                  : stream.reasoning && !stream.answer
                    ? "thinking…"
                    : "writing…"}
              </span>
            </div>
            {stream.reasoning && (
              <details className="reasoning" open>
                <summary className="muted">Reasoning</summary>
                <div className="reasoning-body">{stream.reasoning}</div>
              </details>
            )}
            <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
              {stream.answer}
              <span className="cursor">▍</span>
            </div>
          </div>
        )}
        {messages.length === 0 && !stream && <p className="muted">No questions yet.</p>}
        {messages.map((m) => (
          <div key={m.query_id} className={`msg ${m.error ? "user" : ""}`}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <strong>{m.question}</strong>
              <span className="cost">
                {m.workflow_name} · {m.llm_model_id}
              </span>
            </div>
            <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
              {m.error ? <span className="error">{m.error}</span> : m.answer}
            </div>
            {m.reasoning && (
              <details className="reasoning">
                <summary className="muted">Reasoning</summary>
                <div className="reasoning-body">{m.reasoning}</div>
              </details>
            )}
            <div className="cost" style={{ marginTop: 6 }}>
              {m.total_tokens} tokens ({m.prompt_tokens} in / {m.completion_tokens} out) · $
              {Number(m.cost_usd || 0).toFixed(6)} · {m.cost_source} · {m.total_ms} ms
              {m.retrieved_chunk_ids?.length
                ? ` · chunks ${m.retrieved_chunk_ids.join(", ")}`
                : ""}
            </div>
            <div className="cost">
              <button className="link" onClick={() => toggleLog(m)}>
                {trace?.queryId === m.query_id ? "Hide log" : "Show log"}
              </button>
            </div>
            {trace?.queryId === m.query_id && (
              <div className="trace">
                {trace.loading && <p className="muted">Loading log…</p>}
                {trace.error && <div className="error">{trace.error}</div>}
                {trace.events?.map((event, i) =>
                  event.kind === "retrieval" ? (
                    <div className="trace-block" key={`retrieval-${i}`}>
                      <h4>Retrieved data ({event.payload.chunks?.length || 0} chunks)</h4>
                      {event.payload.chunks?.length ? (
                        event.payload.chunks.map((chunk) => (
                          <details key={chunk.chunk_id}>
                            <summary>
                              #{chunk.rank + 1} · chunk {chunk.chunk_id} ·{" "}
                              {chunk.filename || "document"} · score {chunk.score}
                            </summary>
                            <pre>{chunk.text}</pre>
                          </details>
                        ))
                      ) : (
                        <p className="muted">No chunks retrieved.</p>
                      )}
                    </div>
                  ) : (
                    <div className="trace-block" key={`prompt-${i}`}>
                      <h4>Last LLM prompt</h4>
                      {event.payload.messages?.map((msg, j) => (
                        <div className="trace-msg" key={j}>
                          <span className="muted">{msg.role}</span>
                          <pre>{msg.content}</pre>
                        </div>
                      ))}
                    </div>
                  )
                )}
                {trace.events?.length === 0 && (
                  <p className="muted">No log recorded for this query.</p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}