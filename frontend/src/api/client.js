// Thin wrapper around the backend REST API. Every backend call goes through here.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
// The API router is mounted at /v1 (backend/app/main.py); /health is not.
const API = `${BASE_URL}/v1`;
// Sent when the backend runs with API_AUTH_ENABLED=true. Baked into the bundle
// at build time, so it identifies this client rather than being a secret.
const API_KEY = import.meta.env.VITE_API_KEY || "";
const KEY_HEADERS = API_KEY ? { "X-API-Key": API_KEY } : {};

export const UPLOAD_SUFFIXES = [".pdf", ".csv", ".xlsx", ".xlsm"];
export const MAX_UPLOAD_MB = 20;

// FastAPI reports errors as {detail}; surface that rather than a bare status.
async function errorFrom(res) {
  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    // Non-JSON error body: fall back to the status line.
  }
  const err = new Error(detail || `Request failed: ${res.status} ${res.statusText}`);
  err.status = res.status;
  return err;
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the backend");
  }
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

export function getHealth() {
  return request(`${BASE_URL}/health`);
}

export function listAgents() {
  return request(`${API}/agents`, { headers: KEY_HEADERS });
}

export function chat(question, sessionId) {
  return request(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...KEY_HEADERS },
    body: JSON.stringify({ question, session_id: sessionId ?? null }),
  });
}

export function getSession(sessionId) {
  return request(`${API}/sessions/${encodeURIComponent(sessionId)}`, {
    headers: KEY_HEADERS,
  });
}

// XHR rather than fetch: fetch has no upload progress events.
export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/upload`);
    if (API_KEY) xhr.setRequestHeader("X-API-Key", API_KEY);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // Leave body null; handled below.
      }
      if (xhr.status >= 200 && xhr.status < 300 && body) return resolve(body);
      const detail = body?.detail;
      const err = new Error(
        typeof detail === "string" ? detail : `Upload failed: ${xhr.status}`
      );
      err.status = xhr.status;
      reject(err);
    };
    xhr.onerror = () => reject(new Error("Cannot reach the backend"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
