// Message.jsx - renders one chat bubble (user question or assistant answer).

// Converts Markdown text (headings, lists, tables, code) into React elements.
// It does NOT render raw HTML, so model output cannot inject scripts into the page.
import ReactMarkdown from "react-markdown";
// The collapsible list of retrieved sources under an answer.
import SourceList from "./SourceList.jsx";

export default function Message({ message }) {
  // Is this the user's own message?
  const isUser = message.role === "user";
  // Still waiting for the first token (and no error yet)?
  const waiting = message.status === "streaming" && !message.content && !message.error;

  return (
    // The role becomes a CSS class so user/assistant bubbles look different.
    <div className={`message ${message.role}`}>
      <div className="bubble">
        {/* User text is shown as-is; assistant text is rendered as Markdown */}
        {isUser ? <p className="user-text">{message.content}</p> : <ReactMarkdown>{message.content}</ReactMarkdown>}

        {/* Animated indicator: first "searching", then "thinking" once sources arrived */}
        {waiting && (
          <div className="typing">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
            <span>{message.sources === null ? "Searching your documents…" : "Thinking…"}</span>
          </div>
        )}

        {/* Informational notices (fallback model used, truncated answer, stopped) */}
        {message.notices?.map((notice, index) => (
          <div key={index} className="notice">
            {notice}
          </div>
        ))}

        {/* An error replaces the answer */}
        {message.error && <div className="banner error small">{message.error}</div>}

        {/* The sources used for this answer */}
        {!isUser && message.sources?.length > 0 && <SourceList sources={message.sources} query={message.query} />}

        {/* Model and token usage, once the answer is complete */}
        {message.meta && (
          <div className="meta">
            {message.meta.model} · {message.meta.input_tokens.toLocaleString()} in /{" "}
            {message.meta.output_tokens.toLocaleString()} out tokens
          </div>
        )}
      </div>
    </div>
  );
}
