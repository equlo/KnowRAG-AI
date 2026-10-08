// App.jsx - the top-level component: page layout + shared document state.

// React hooks: useState stores data, useEffect runs side effects,
// useCallback keeps a function identity stable between renders.
import { useCallback, useEffect, useState } from "react";
// API helpers.
import { getHealth, listDocuments } from "./api.js";
// The left-hand knowledge-base panel.
import DocumentPanel from "./components/DocumentPanel.jsx";
// The right-hand chat panel.
import ChatPanel from "./components/ChatPanel.jsx";

export default function App() {
  // The list of indexed documents (shared by both panels).
  const [documents, setDocuments] = useState([]);
  // Backend status from /api/health (null until loaded).
  const [health, setHealth] = useState(null);
  // An error message if the backend could not be reached.
  const [loadError, setLoadError] = useState("");

  // Reload documents and health info from the backend.
  const refresh = useCallback(async () => {
    // Network calls can fail, so guard them.
    try {
      // Run both requests in parallel and wait for both.
      const [docs, status] = await Promise.all([listDocuments(), getHealth()]);
      // Store the results; React re-renders automatically.
      setDocuments(docs);
      setHealth(status);
      // Clear any previous error.
      setLoadError("");
    } catch (error) {
      // Show why the backend is unreachable.
      setLoadError(`Cannot reach the backend: ${error.message}`);
    }
  }, []);

  // Load data when the page first opens. `refresh` was created with
  // useCallback(..., []) so it never changes, and this effect runs only once.
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Describe what the page looks like.
  return (
    <div className="app">
      {/* Top bar: app name and backend status */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">K</span>
          <div>
            <h1>KnowRAG AI</h1>
            <p className="tagline">Chat with your documents · Retrieval-Augmented Generation</p>
          </div>
        </div>
        {/* Show model + embedding info once health has loaded */}
        {health && (
          <div className="status" title={`Embeddings: ${health.embedding}`}>
            <span className="status-dot" aria-hidden="true" />
            {health.model}
          </div>
        )}
      </header>

      {/* A banner when the backend cannot be reached */}
      {loadError && <div className="banner error">{loadError}</div>}

      {/* Two-column layout: documents on the left, chat on the right */}
      <main className="layout">
        <DocumentPanel documents={documents} onChange={refresh} />
        <ChatPanel hasDocuments={documents.length > 0} />
      </main>
    </div>
  );
}
