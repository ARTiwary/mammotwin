/**
 * Thin fetch wrapper around the FastAPI backend (backend/main.py).
 * Assumes the backend is running at API_BASE (default: localhost:8000,
 * FastAPI's default uvicorn port) -- override with VITE_API_BASE if needed.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON -- keep statusText */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse(res);
}

export async function predictUpload(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict/upload`, { method: "POST", body: form });
  return handleResponse(res);
}

export async function getExplanation(imageId) {
  const res = await fetch(`${API_BASE}/explain/${imageId}`);
  return handleResponse(res);
}

export async function getUncertainty(imageId) {
  const res = await fetch(`${API_BASE}/uncertainty/${imageId}`);
  return handleResponse(res);
}

export async function compareLongitudinal(priorFile, currentFile) {
  const form = new FormData();
  form.append("prior", priorFile);
  form.append("current", currentFile);
  const res = await fetch(`${API_BASE}/longitudinal/compare`, { method: "POST", body: form });
  return handleResponse(res);
}
