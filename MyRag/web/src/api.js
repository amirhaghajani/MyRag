const BASE = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }

  if (response.status === 204) return null;
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export const api = {
  listMachines: () => request("/machines"),
  createMachine: (payload) =>
    request("/machines", { method: "POST", body: JSON.stringify(payload) }),
  getMachine: (id) => request(`/machines/${id}`),
  deleteMachine: (id) => request(`/machines/${id}`, { method: "DELETE" }),

  listDocuments: (machineId) => request(`/machines/${machineId}/documents`),
  uploadDocument: async (machineId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/machines/${machineId}/documents`, {
      method: "POST",
      headers: {},
      body: form,
    });
  },
  indexDocument: (documentId) =>
    request(`/documents/${documentId}/index`, { method: "POST" }),
  deleteDocument: (documentId) =>
    request(`/documents/${documentId}`, { method: "DELETE" }),

  listWorkflows: () => request("/workflows"),
  listModels: () => request("/models"),

  chat: (machineId, payload) =>
    request(`/machines/${machineId}/chat`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  chatStream: async (machineId, payload, onEvent) => {
    const response = await fetch(`${BASE}/machines/${machineId}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || JSON.stringify(body);
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const event = JSON.parse(line.slice(5).trim());
        onEvent(event);
        if (event.type === "done" || event.type === "error") return event;
      }
    }
    return null;
  },
  listMessages: (machineId) => request(`/machines/${machineId}/messages`),

  usage: (machineId) => request(`/machines/${machineId}/usage`),
  compareWorkflows: (machineId) => request(`/usage/compare?machine_id=${machineId}`),
};