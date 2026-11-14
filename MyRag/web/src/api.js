const BASE = "/api";
const AUTH_KEY = "myrag.auth";

export function getAuthHeader() {
  try {
    const token = sessionStorage.getItem(AUTH_KEY);
    return token ? { Authorization: `Basic ${token}` } : {};
  } catch {
    return {};
  }
}

export function setAuth(username, password) {
  sessionStorage.setItem(AUTH_KEY, btoa(`${username}:${password}`));
}

export function clearAuth() {
  sessionStorage.removeItem(AUTH_KEY);
}

function rejectUnauthorized() {
  clearAuth();
  window.dispatchEvent(new Event("myrag:unauthorized"));
}

async function request(path, options = {}) {
  const isForm = typeof FormData !== "undefined" && options.body instanceof FormData;
  const response = await fetch(`${BASE}${path}`, {
    headers: {
      ...(isForm ? {} : { "Content-Type": "application/json" }),
      ...getAuthHeader(),
      ...(options.headers || {}),
    },
    ...options,
  });

  if (response.status === 401) {
    rejectUnauthorized();
    throw new Error("Login required");
  }

  if (!response.ok) {
    let detail = response.statusText;
    let payload = null;
    try {
      payload = await response.json();
      detail = payload.detail ?? JSON.stringify(payload);
    } catch {
      /* non-JSON error body */
    }
    if (typeof detail !== "string") detail = JSON.stringify(detail);
    const error = new Error(detail);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  if (response.status === 204) return null;
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export const api = {
  authStatus: () => request("/auth/status"),
  setAuth,
  checkAuth: async (username, password) => {
    const response = await fetch(`${BASE}/auth/status`, {
      headers: { Authorization: `Basic ${btoa(`${username}:${password}`)}` },
    });
    const body = await response.json().catch(() => ({}));
    return { ok: response.ok && Boolean(body.authenticated), body };
  },
  logout: () => clearAuth(),

  listMachines: () => request("/machines"),
  createMachine: (payload) =>
    request("/machines", { method: "POST", body: JSON.stringify(payload) }),
  getMachine: (id) => request(`/machines/${id}`),
  updateMachine: (id, payload) =>
    request(`/machines/${id}`, { method: "PATCH", body: JSON.stringify(payload) }),
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
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader(),
      },
      body: JSON.stringify(payload),
    });
    if (response.status === 401) {
      rejectUnauthorized();
      throw new Error("Login required");
    }
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
  queryLog: (queryId) => request(`/queries/${queryId}/log`),

  usage: (machineId) => request(`/machines/${machineId}/usage`),
  compareWorkflows: (machineId) => request(`/usage/compare?machine_id=${machineId}`),
};