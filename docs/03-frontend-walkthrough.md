# 3. Frontend walkthrough (React / Vite)

The frontend is a small single-page application: a document panel on the left and a streaming chat on the right. Every `.jsx`/`.js`/`.css` file is commented line by line. This guide explains the files that **can't** contain comments (JSON), then the core ideas: streaming, state updates and rendering.

```
index.html ─► main.jsx ─► App.jsx ─┬─► DocumentPanel.jsx ──► api.js (upload/list/delete)
                                   └─► ChatPanel.jsx ──────► api.js (streamChat)
                                          └─► Message.jsx ──► SourceList.jsx
```

---

## 3.1 `package.json`, line by line

JSON does not allow comments, so here is every line explained:

| Line | Meaning |
|---|---|
| `"name": "knowrag-ai-frontend"` | Package name (only used locally) |
| `"private": true` | Prevents accidentally publishing this app to the npm registry |
| `"version": "1.0.0"` | Our app's version |
| `"type": "module"` | `.js` files use modern `import`/`export` syntax (ES modules) |
| `"scripts"` → `"dev": "vite"` | `npm run dev` starts the development server with hot reload |
| `"build": "vite build"` | `npm run build` produces optimised static files in `dist/` |
| `"preview": "vite preview"` | `npm run preview` serves the built `dist/` locally to test it |
| `"dependencies"` → `react`, `react-dom` | React itself, and its renderer for web pages |
| `react-markdown` | Renders Claude's Markdown answers (lists, tables, code) safely |
| `"devDependencies"` → `vite` | The dev server and bundler (only needed while building) |
| `@vitejs/plugin-react` | Teaches Vite to compile JSX and enables Fast Refresh |

`package-lock.json` records the **exact** installed versions of every package and sub-package, so `npm ci` (used in Docker) reproduces the same build everywhere.

---

## 3.2 `vite.config.js`: the dev proxy

```js
server: { port: 5173, proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } } }
```

During development the UI runs on port 5173 and the API on port 8000. The proxy forwards every `/api/...` request from the Vite server to FastAPI. As far as the browser knows, everything comes from **one origin**, so `fetch("/api/chat")` works without CORS complications and without hard-coding URLs. In production, nginx plays the same role (see `nginx.conf`).

---

## 3.3 `index.html` and `main.jsx`: booting React

- `index.html` contains just `<div id="root"></div>` and `<script type="module" src="/src/main.jsx">`.
- `main.jsx` calls `createRoot(document.getElementById("root")).render(<App />)`, which hands that div to React.
- `<StrictMode>` runs extra development-only checks (for example, it deliberately runs effects twice to expose bugs). It has no effect in production builds.

---

## 3.4 `api.js`: talking to the backend

All network code lives here, so components stay focused on the UI.

### `parseResponse(response)`
This is one place to turn HTTP failures into readable errors. FastAPI sends errors as `{"detail": "..."}`, or for validation errors (422) as `{"detail": [{msg: ...}, ...]}`. Both shapes are handled, and the function throws `new Error(message)` so components can simply `catch` and display it.

### `uploadDocument(file)`
```js
const form = new FormData();
form.append("file", file);   // "file" must match the FastAPI parameter name
fetch("/api/documents", { method: "POST", body: form });
```
We **don't** set `Content-Type` ourselves. For `FormData` the browser generates `multipart/form-data; boundary=...` with a unique boundary string. Setting the header manually would omit the boundary and break the upload.

### `streamChat()`: reading Server-Sent Events from a POST
The browser's built-in `EventSource` only supports **GET** requests, but we need to **POST** a JSON body containing the question and history. So we read the response stream ourselves.

`streamChat` is an **async generator** (`async function*`). It `yield`s each event as soon as it is parsed, and the caller consumes events with `for await (const event of streamChat(...))`.

Line by line:

```js
const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
```
`response.body` is a stream of raw **bytes**. `TextDecoderStream` converts them to **text**, correctly handling multi-byte characters (such as emoji) that are split across network packets. `getReader()` lets us pull pieces one at a time.

```js
let buffer = "";
while (true) {
  const { value, done } = await reader.read();   // wait for the next piece of text
  if (done) break;                               // server closed the stream
  buffer += value;
```
Network packets don't align with events. One `read()` may return half an event, or three and a half events. So we **accumulate** text in a buffer.

```js
  let boundary;
  while ((boundary = buffer.indexOf("\n\n")) !== -1) {   // a blank line ends an SSE event
    const rawEvent = buffer.slice(0, boundary);            // one complete event
    buffer = buffer.slice(boundary + 2);                   // keep the rest for later
    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
    }
  }
}
```
Every complete event (up to a blank line) is cut out of the buffer and parsed. An incomplete tail stays in the buffer until more text arrives.

The `signal` option links the request to an `AbortController`. Calling `controller.abort()` (the **Stop** button) cancels the fetch. The backend notices the closed connection, which ends its request to Claude, so you stop paying for tokens.

---

## 3.5 `App.jsx`: layout and shared state

- `documents` lives here (not in `DocumentPanel`) because **two** components need it: the panel shows the list, and the chat uses `documents.length > 0` to decide which hint to show. Keeping shared state in the closest common parent is called "lifting state up".
- `refresh()` calls `listDocuments()` and `getHealth()` **in parallel** with `Promise.all`. It is wrapped in `useCallback(..., [])` so it keeps the same identity across renders, and the `useEffect(() => refresh(), [refresh])` therefore runs only once, on page load.
- If the backend is unreachable, a red banner explains why instead of showing a blank page.

---

## 3.6 `DocumentPanel.jsx`: uploads

- A visible **drop zone** plus a hidden `<input type="file" multiple>`. Clicking the zone calls `inputRef.current.click()`, which opens the native file picker.
- Drag and drop: `onDragOver` must call `e.preventDefault()`, otherwise the browser never fires `onDrop` and instead navigates to the dropped file.
- Files upload **one at a time**. Each upload is CPU-heavy on the server (parsing and embedding), and sequential uploads give clear per-file progress ("Indexing report.pdf…") and per-file errors.
- After choosing files, `e.target.value = ""` resets the input. Otherwise picking the *same* file again would not fire `onChange`.
- The drop zone has `role="button"`, `tabIndex={0}` and an Enter/Space handler, so keyboard users can operate it too.

---

## 3.7 `ChatPanel.jsx`: conversation and streaming

### The message model
Each message in `messages` is a plain object:

```js
{ id, role: "user" | "assistant", content, sources, query, notices, status, error, meta }
```
- `sources: null` means "still searching". An array (even an empty one) means retrieval finished. `Message.jsx` uses this to switch between "Searching your documents…" and "Thinking…".
- `status: "streaming"` or `"complete"`.
- `meta` is the `done` event (model name and token counts).

### Why `updateMessage` uses a *functional* state update
```js
setMessages((previous) => previous.map((m) => (m.id === id ? { ...m, ...changes } : m)));
```
During streaming, dozens of `token` events arrive per second. If we wrote `setMessages(messages.map(...))` using the `messages` variable captured when `ask()` started, every update would start from that **stale snapshot** and overwrite the previous token. The functional form `setMessages(prev => ...)` always receives the **latest** state, so each token is appended correctly. This is the most important React detail in the app.

For tokens, `changes` is itself a function of the current message:
```js
updateMessage(assistantId, (m) => ({ content: m.content + event.text }));
```

### The `ask()` flow
1. Build `history` from earlier messages. Failed and empty messages are skipped, and only the last 20 are kept (the backend accepts at most 40) to keep requests small.
2. Append the user message and an empty assistant placeholder in **one** state update.
3. Create an `AbortController` and store it in a `useRef`. Refs persist across renders without causing re-renders, which is ideal for this kind of handle.
4. `for await (const event of streamChat(...))` dispatches on `event.type`: `sources`, `token`, `notice`, `done` or `error`.
5. `catch`: an `AbortError` means the user pressed Stop, so we keep the partial answer and add "Stopped.". Any other error is displayed.
6. `finally`: mark the message complete, unlock the input and clear the controller, whatever happened.

### Details
- **Enter sends, Shift+Enter adds a new line.** `event.nativeEvent.isComposing` is checked so that users typing Chinese, Japanese or Korean with an input-method editor don't send half-composed text.
- **Auto-scroll:** an empty `<div ref={bottomRef}>` sits after the last message, and a `useEffect` on `[messages]` scrolls it into view on every token.
- **Suggestions:** the clickable starter questions appear only when documents exist.

---

## 3.8 `Message.jsx` and `SourceList.jsx`: rendering

- **User text** is rendered as plain text with `white-space: pre-wrap`, so line breaks are preserved.
- **Assistant text** goes through `<ReactMarkdown>`. React-markdown builds React elements from Markdown and does **not** render raw HTML by default. Even if a document tricked the model into emitting `<script>`, it would appear as text, never execute. This is an important XSS safeguard for any app that displays LLM output.
- `SourceList` uses native `<details>`/`<summary>` elements, which give collapsible sections with keyboard support and no JavaScript. Each source shows its `[n]` number (matching the citations in the answer), file name, page and cosine-similarity score.

---

## 3.9 `styles.css`: theming and layout

- **Design tokens:** colours, radius and font are CSS custom properties on `:root` (`--bg`, `--accent`, …). Components only reference tokens, so the whole theme changes in one place.
- **Dark mode:** `@media (prefers-color-scheme: dark)` redefines the same tokens, and `color-scheme` makes native scrollbars and inputs match.
- **Layout:** CSS Grid with `grid-template-columns: 320px 1fr`. `min-height: 0` on grid and flex children is the classic fix that lets an inner area (`.messages`) scroll instead of stretching the page.
- **Mobile:** below 800 px the grid becomes one column, with a 16 px side gutter and no horizontal scrolling.
