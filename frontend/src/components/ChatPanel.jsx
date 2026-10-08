// ChatPanel.jsx - the conversation: message list, input box, and streaming logic.

// useEffect auto-scrolls, useRef holds DOM nodes/controllers, useState holds data.
import { useEffect, useRef, useState } from "react";
// The streaming chat API helper.
import { streamChat } from "../api.js";
// Renders one chat bubble.
import Message from "./Message.jsx";

// Example questions shown before the first message.
const SUGGESTIONS = [
  "Summarize the key points of my documents.",
  "What are the main risks or open questions mentioned?",
  "List the important dates and deadlines.",
];

// Only the most recent turns are sent as history (keeps requests small).
const MAX_HISTORY_MESSAGES = 20;

// A simple counter gives every message a unique `id` (React needs stable keys).
let nextId = 0;
const newId = () => ++nextId;

// `hasDocuments` tells us whether to show the "upload first" hint.
export default function ChatPanel({ hasDocuments }) {
  // All messages in the conversation, oldest first.
  const [messages, setMessages] = useState([]);
  // Current contents of the input box.
  const [input, setInput] = useState("");
  // True while an answer is streaming (disables sending another question).
  const [busy, setBusy] = useState(false);
  // The AbortController of the running request, so "Stop" can cancel it.
  const abortRef = useRef(null);
  // An invisible element at the end of the list, used for auto-scrolling.
  const bottomRef = useRef(null);

  // Whenever messages change (including each new token), scroll to the bottom.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Update one message by id. `changes` is an object or a function of the old message.
  function updateMessage(id, changes) {
    // Functional setState: always works on the latest state, even mid-stream.
    setMessages((previous) =>
      previous.map((message) =>
        message.id === id
          ? { ...message, ...(typeof changes === "function" ? changes(message) : changes) }
          : message,
      ),
    );
  }

  // Send a question and stream the answer into a new assistant message.
  async function ask(question) {
    // Build the history from earlier messages: skip failed/empty ones, keep the latest few.
    const history = messages
      .filter((m) => m.content && !m.error)
      .slice(-MAX_HISTORY_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content }));
    // Reserve an id for the assistant's reply.
    const assistantId = newId();
    // Add the user's question and an empty assistant message to the list.
    setMessages((previous) => [
      ...previous,
      { id: newId(), role: "user", content: question },
      // `sources: null` means "still searching"; `status` tracks streaming.
      { id: assistantId, role: "assistant", content: "", sources: null, notices: [], status: "streaming" },
    ]);
    // Lock the input.
    setBusy(true);
    // Create a controller so the request can be cancelled.
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // Consume server events one by one as they arrive.
      for await (const event of streamChat({ question, history, signal: controller.signal })) {
        // Retrieved chunks arrive first.
        if (event.type === "sources") updateMessage(assistantId, { sources: event.sources, query: event.query });
        // Each token is appended to the answer text.
        else if (event.type === "token") updateMessage(assistantId, (m) => ({ content: m.content + event.text }));
        // Informational notices are collected in a list.
        else if (event.type === "notice") updateMessage(assistantId, (m) => ({ notices: [...m.notices, event.text] }));
        // "done" carries model name and token usage.
        else if (event.type === "done") updateMessage(assistantId, { meta: event });
        // "error" carries a message to show instead of an answer.
        else if (event.type === "error") updateMessage(assistantId, { error: event.message });
      }
    } catch (error) {
      // The user pressed "Stop": keep the partial answer and add a note.
      if (error.name === "AbortError") updateMessage(assistantId, (m) => ({ notices: [...m.notices, "Stopped."] }));
      // Network/HTTP failure: show it.
      else updateMessage(assistantId, { error: error.message });
    } finally {
      // Whatever happened, the message is no longer streaming...
      updateMessage(assistantId, { status: "complete" });
      // ...the input unlocks...
      setBusy(false);
      // ...and there is nothing left to cancel.
      abortRef.current = null;
    }
  }

  // Form submit handler (button click or Enter key).
  function handleSubmit(event) {
    // Prevent the browser's default full-page form submission.
    event?.preventDefault();
    // Remove surrounding whitespace.
    const question = input.trim();
    // Ignore empty input and double submissions.
    if (!question || busy) return;
    // Clear the box and ask.
    setInput("");
    ask(question);
  }

  // Enter sends; Shift+Enter inserts a newline.
  function handleKeyDown(event) {
    // `isComposing` is true while typing with an input method (e.g. Chinese/Japanese).
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) handleSubmit(event);
  }

  // Describe the panel.
  return (
    <section className="panel chat-panel">
      {/* Scrollable message area */}
      <div className="messages">
        {/* Empty state: welcome text and suggested questions */}
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Ask anything about your documents</h2>
            <p className="muted">
              {hasDocuments
                ? "Answers are grounded in your files and cite their sources like [1]."
                : "Start by uploading a PDF, Word, text or Markdown file on the left."}
            </p>
            {/* Suggestions are clickable only when there is something to search */}
            {hasDocuments && (
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="chip" onClick={() => ask(s)} disabled={busy}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {/* One <Message> per message */}
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        {/* Scroll target */}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          // Controlled input: React state is the source of truth.
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          // Enter sends, Shift+Enter adds a line (see handleKeyDown).
          placeholder="Ask a question about your documents…"
          rows={2}
          aria-label="Your question"
        />
        {/* While streaming the button becomes "Stop"; otherwise it sends */}
        {busy ? (
          <button type="button" className="button secondary" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="button" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </section>
  );
}
