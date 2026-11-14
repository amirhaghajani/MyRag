import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Login from "./components/Login.jsx";
import MachineCreate from "./components/MachineCreate.jsx";
import MachineEmbedSettings from "./components/MachineEmbedSettings.jsx";
import MachineList from "./components/MachineList.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import WorkflowPicker from "./components/WorkflowPicker.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import UsagePanel from "./components/UsagePanel.jsx";

export default function App() {
  const [machines, setMachines] = useState([]);
  const [selected, setSelected] = useState(null);
  const [models, setModels] = useState(null);
  const [workflows, setWorkflows] = useState([]);
  const [workflowId, setWorkflowId] = useState(null);
  const [error, setError] = useState("");
  const [auth, setAuth] = useState({ ready: false, enabled: false, authenticated: true });

  useEffect(() => {
    api
      .authStatus()
      .then((status) =>
        setAuth({
          ready: true,
          enabled: status.enabled,
          authenticated: status.authenticated,
        })
      )
      .catch(() => setAuth({ ready: true, enabled: false, authenticated: true }));
  }, []);

  useEffect(() => {
    const onUnauthorized = () =>
      setAuth((prev) => ({ ...prev, authenticated: false }));
    window.addEventListener("myrag:unauthorized", onUnauthorized);
    return () => window.removeEventListener("myrag:unauthorized", onUnauthorized);
  }, []);

  const loadMachines = useCallback(async () => {
    try {
      setMachines(await api.listMachines());
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    if (!auth.ready || (auth.enabled && !auth.authenticated)) return;
    loadMachines();
    api.listModels().then(setModels).catch((e) => setError(e.message));
    api
      .listWorkflows()
      .then((rows) => {
        setWorkflows(rows);
        if (rows.length) setWorkflowId(rows[0].id);
      })
      .catch((e) => setError(e.message));
  }, [loadMachines, auth.ready, auth.enabled, auth.authenticated]);

  function refreshAll() {
    loadMachines();
  }

  if (!auth.ready) {
    return <p className="muted" style={{ padding: 24 }}>Loading…</p>;
  }

  if (auth.enabled && !auth.authenticated) {
    return (
      <Login
        onSuccess={() => setAuth((prev) => ({ ...prev, authenticated: true }))}
      />
    );
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>MyRag</h1>
        {auth.enabled && (
          <button
            className="link"
            onClick={() => {
              api.logout();
              setAuth((prev) => ({ ...prev, authenticated: false }));
            }}
          >
            Sign out
          </button>
        )}
        {error && <div className="error" style={{ marginBottom: 10 }}>{error}</div>}
        <MachineCreate models={models} onCreated={refreshAll} />
        <h2>Machines</h2>
        <MachineList
          machines={machines}
          selectedId={selected?.id}
          onSelect={(m) => setSelected(m)}
          onRefresh={refreshAll}
        />
      </aside>

      <main className="main">
        {!selected ? (
          <p className="muted">Select a machine to upload data and start chatting.</p>
        ) : (
          <>
            <h1 style={{ marginBottom: 4 }}>{selected.title}</h1>
            <p className="cost">
              embed model {selected.embed_model_id} · dim {selected.embed_dim} · chunk{" "}
              {selected.chunk_size}/{selected.chunk_overlap} · {selected.chunk_count} chunks
            </p>

            {models && (
              <MachineEmbedSettings
                machine={selected}
                models={models}
                onChanged={loadMachines}
              />
            )}

            <UploadPanel
              machine={selected}
              models={models}
              onChanged={refreshAll}
              onMachineUpdated={setSelected}
            />

            <div className="card">
              <WorkflowPicker
                workflows={workflows}
                value={workflowId}
                onChange={setWorkflowId}
              />
            </div>

            <ChatPanel
              machineId={selected.id}
              machine={selected}
              models={models}
              workflows={workflows}
              workflowId={workflowId}
              onWorkflowChange={setWorkflowId}
              onMachineUpdated={setSelected}
            />

            <UsagePanel machineId={selected.id} />
          </>
        )}
      </main>
    </div>
  );
}