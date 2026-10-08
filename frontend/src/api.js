// api.js - every HTTP call the UI makes to the FastAPI backend lives here.
// Components import these functions instead of calling fetch() directly.

// Base URL of the backend. Empty string = "same origin as the page"
// (the Vite proxy in development, nginx in production). Set VITE_API_BASE_URL
// at build time to point the UI at a backend on another domain.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

// Turn a fetch Response into JSON, or throw a readable Error for HTTP failures.
async function parseResponse(response) {
  // 2xx status codes mean success.
  if (response.ok) {
    // 204 No Content has no body to parse.
    return response.status === 204 ? null : response.json();
  }
  // Start with the generic status text, e.g. "Bad Request".
  let message = response.statusText || `HTTP ${response.status}`;
  // FastAPI puts error details in {"detail": ...}; try to read it.
  try {
    // Parse the error body.
    const body = await response.json();
    // `detail` is a string for our own errors...
    if (typeof body.detail === "string") message = body.detail;
    // ...or a list of validation problems for 422 errors.
    else if (Array.isArray(body.detail)) message = body.detail.map((d) => d.msg).join("; ");
  } catch {
    // The body was not JSON - keep the status text.
  }
  // Throw so the calling component can show the message.
  throw new Error(message);
}

// GET /api/health -> { status, model, embedding, documents }
export async function getHealth() {
  // Send the request and parse the result.
  return parseResponse(await fetch(`${API_BASE}/api/health`));
}

// GET /api/documents -> array of document metadata
export async function listDocuments() {
  // Send the request and parse the result.
  return parseResponse(await fetch(`${API_BASE}/api/documents`));
}

// POST /api/documents (multipart upload of ONE file) -> the new document's metadata
export async function uploadDocument(file) {
  // FormData builds a multipart/form-data body, the format used for file uploads.
  const form = new FormData();
  // The field name "file" must match the FastAPI parameter name.
  form.append("file", file);
  // Don't set Content-Type manually: the browser adds it with the right boundary.
  return parseResponse(await fetch(`${API_BASE}/api/documents`, { method: "POST", body: form }));
}

// DELETE /api/documents/{id}
export async function deleteDocument(id) {
  // encodeURIComponent guards against special characters in the ID.
  return parseResponse(await fetch(`${API_BASE}/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }));
}

// POST /api/chat -> Server-Sent Events. This is an ASYNC GENERATOR: the caller
// writes `for await (const event of streamChat(...))` and receives each event
// object ({type: "token", text: "..."} etc.) the moment it arrives.
export async function* streamChat({ question, history, signal }) {
  // Start the request. `signal` lets the user cancel it (AbortController).
  const response = await fetch(`${API_BASE}/api/chat`, {
    // POST because we send a JSON body.
    method: "POST",
    // Tell the server the body is JSON.
    headers: { "Content-Type": "application/json" },
    // The question and earlier conversation turns.
    body: JSON.stringify({ question, history }),
    // Hook for cancellation.
    signal,
  });
  // For HTTP errors (e.g. 422), parseResponse throws a readable Error.
  if (!response.ok) await parseResponse(response);

  // Read the body as a stream of text pieces instead of waiting for all of it.
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  // Holds text received but not yet processed (an event may arrive split in two).
  let buffer = "";
  // Keep reading until the server closes the stream.
  while (true) {
    // Wait for the next piece of text.
    const { value, done } = await reader.read();
    // `done` means the response has ended.
    if (done) break;
    // Append the new text to the buffer.
    buffer += value;
    // SSE events are separated by a blank line ("\n\n"); process each complete one.
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      // Cut one complete event out of the buffer...
      const rawEvent = buffer.slice(0, boundary);
      // ...and keep the remainder for the next round.
      buffer = buffer.slice(boundary + 2);
      // An event may contain several lines; we only use "data: ..." lines.
      for (const line of rawEvent.split("\n")) {
        // Parse the JSON after "data: " and hand it to the caller.
        if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
      }
    }
  }
}
