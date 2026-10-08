// api.js - every HTTP call the UI makes to the Express backend lives here.
// Components import these functions instead of calling fetch() directly.

// Base URL of the backend. Empty string = "same origin as the page"
// (the Vite proxy in development, nginx in production). Set VITE_API_URL
// at build time to point the UI at a backend on another domain.
const API_BASE = import.meta.env.VITE_API_URL ?? "";

// Send one request to /api/<path> and return the parsed JSON (or throw an Error).
async function request(path, { method = "GET", body } = {}) {
  // Holds the server's reply once it arrives.
  let response;
  try {
    // Send the request. Only requests with a body need a Content-Type header.
    response = await fetch(`${API_BASE}/api${path}`, {
      // GET, POST, PATCH or DELETE.
      method,
      // Tell Express the body is JSON so express.json() parses it.
      headers: body ? { "Content-Type": "application/json" } : undefined,
      // Turn the JavaScript object into a JSON string.
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    // fetch() only throws when no reply came back at all (server down, no network).
    throw new Error("Cannot reach the server. Is the backend running?");
  }

  // 204 No Content (successful delete) has no body to parse.
  if (response.status === 204) return null;

  // Parse the JSON body; fall back to null if it isn't JSON (e.g. a proxy error page).
  const data = await response.json().catch(() => null);

  // Any status outside 200-299 is a failure.
  if (!response.ok) {
    // Our API always sends { error: "..." }; use it when present.
    if (data?.error) throw new Error(data.error);
    // A 5xx without our JSON usually means the proxy (Vite or nginx) couldn't reach Express.
    if (response.status >= 500) throw new Error(`The server is unavailable (HTTP ${response.status}). Is the backend running?`);
    // Anything else: report the status code.
    throw new Error(`Request failed with status ${response.status}`);
  }

  // Success: hand the data to the caller.
  return data;
}

// GET /api/tasks -> array of tasks, newest first.
export function getTasks() {
  return request("/tasks");
}

// POST /api/tasks -> the newly created task.
export function createTask(title) {
  return request("/tasks", { method: "POST", body: { title } });
}

// PATCH /api/tasks/:id -> the updated task. `changes` is e.g. { completed: true }.
export function updateTask(id, changes) {
  return request(`/tasks/${encodeURIComponent(id)}`, { method: "PATCH", body: changes });
}

// DELETE /api/tasks/:id -> null.
export function deleteTask(id) {
  return request(`/tasks/${encodeURIComponent(id)}`, { method: "DELETE" });
}
