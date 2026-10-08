// SourceList.jsx - shows the chunks retrieved for an answer, so users can verify citations.

// `sources` = retrieved chunks; `query` = the (possibly rewritten) search query.
export default function SourceList({ sources, query }) {
  return (
    // <details> is a native collapsible element: closed by default, click to open.
    <details className="sources">
      {/* The always-visible header line */}
      <summary>
        {sources.length} source{sources.length === 1 ? "" : "s"} used
      </summary>
      {/* Show the search query that was actually used (helpful for follow-ups) */}
      {query && <p className="muted small">Search query: “{query}”</p>}
      <ol>
        {/* One entry per source, numbered to match the [n] citations */}
        {sources.map((source) => (
          <li key={source.number}>
            {/* Each source is itself collapsible so long chunks don't flood the chat */}
            <details>
              <summary>
                {/* Citation number */}
                <span className="source-number">[{source.number}]</span>
                {/* File name, plus page for PDFs */}
                <span className="source-file">
                  {source.filename}
                  {source.page != null && ` · p. ${source.page}`}
                </span>
                {/* Cosine similarity: 1.00 = identical meaning, around 0 = unrelated */}
                <span className="source-score" title="Cosine similarity between your question and this chunk">
                  similarity {source.score.toFixed(2)}
                </span>
              </summary>
              {/* The full chunk text */}
              <blockquote>{source.text}</blockquote>
            </details>
          </li>
        ))}
      </ol>
    </details>
  );
}
