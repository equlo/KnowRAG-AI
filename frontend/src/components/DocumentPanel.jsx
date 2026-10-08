// DocumentPanel.jsx - upload, list and delete documents in the knowledge base.

// useRef points at the hidden <input type="file">; useState tracks UI state.
import { useRef, useState } from "react";
// API helpers for uploading and deleting.
import { deleteDocument, uploadDocument } from "../api.js";

// File types the backend accepts (used by the file picker's filter).
const ACCEPTED = ".pdf,.txt,.md,.docx";

// `documents` = current list; `onChange` = callback to reload it after changes.
export default function DocumentPanel({ documents, onChange }) {
  // Reference to the hidden file input so a button can open it.
  const inputRef = useRef(null);
  // Name of the file currently uploading ("" when idle).
  const [uploading, setUploading] = useState("");
  // Error messages from failed uploads/deletes.
  const [errors, setErrors] = useState([]);
  // True while a file is dragged over the drop zone (for highlighting).
  const [dragging, setDragging] = useState(false);

  // Upload a list of files one after another.
  async function handleFiles(fileList) {
    // Convert the browser's FileList into a normal array.
    const files = Array.from(fileList);
    // Nothing selected -> nothing to do.
    if (files.length === 0) return;
    // Collect errors for this batch.
    const newErrors = [];
    // Upload sequentially (embedding is CPU-heavy on the server).
    for (const file of files) {
      // Show which file is in progress.
      setUploading(file.name);
      // Upload, remembering any failure.
      try {
        await uploadDocument(file);
      } catch (error) {
        newErrors.push(`${file.name}: ${error.message}`);
      }
    }
    // Back to idle.
    setUploading("");
    // Show this batch's errors (if any).
    setErrors(newErrors);
    // Reload the document list from the server.
    await onChange();
  }

  // Delete one document after asking for confirmation.
  async function handleDelete(doc) {
    // Native confirmation dialog; stop if the user clicks "Cancel".
    if (!window.confirm(`Remove "${doc.filename}" from the knowledge base?`)) return;
    // Delete, showing any error.
    try {
      await deleteDocument(doc.id);
      setErrors([]);
    } catch (error) {
      setErrors([`${doc.filename}: ${error.message}`]);
    }
    // Reload the list.
    await onChange();
  }

  // Called when files are dropped onto the drop zone.
  function handleDrop(event) {
    // Stop the browser from opening the file itself.
    event.preventDefault();
    // Remove the highlight.
    setDragging(false);
    // Upload the dropped files.
    handleFiles(event.dataTransfer.files);
  }

  // Describe the panel.
  return (
    <aside className="panel documents-panel">
      <h2>Knowledge base</h2>

      {/* Drop zone: click to pick files, or drag files onto it */}
      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        // Make the div keyboard-accessible like a button.
        role="button"
        tabIndex={0}
        // Clicking opens the hidden file picker.
        onClick={() => inputRef.current?.click()}
        // Enter/Space also open the picker (keyboard users).
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        // preventDefault on dragover is required for drop to fire.
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        // Remove the highlight when the drag leaves.
        onDragLeave={() => setDragging(false)}
        // Handle dropped files.
        onDrop={handleDrop}
      >
        {/* Show progress while uploading, instructions otherwise */}
        {uploading ? (
          <span>Indexing {uploading}…</span>
        ) : (
          <span>
            <strong>Drop files here</strong> or click to browse
            <br />
            <small>PDF, DOCX, TXT, MD</small>
          </span>
        )}
      </div>

      {/* The real file input, hidden; `multiple` allows selecting several files */}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        multiple
        hidden
        onChange={(e) => {
          // Upload the chosen files...
          handleFiles(e.target.files);
          // ...and reset the input so choosing the same file again still triggers onChange.
          e.target.value = "";
        }}
      />

      {/* Upload/delete errors */}
      {errors.map((message, index) => (
        <div key={index} className="banner error small">
          {message}
        </div>
      ))}

      {/* The document list (or a hint when it is empty) */}
      {documents.length === 0 ? (
        <p className="muted">No documents yet. Upload some to start asking questions.</p>
      ) : (
        <ul className="document-list">
          {/* One list item per document; `key` helps React track items */}
          {documents.map((doc) => (
            <li key={doc.id}>
              <div className="doc-info">
                {/* File name (full name shown on hover) */}
                <span className="doc-name" title={doc.filename}>
                  {doc.filename}
                </span>
                {/* Chunk and character counts ("1 chunk" vs "2 chunks") */}
                <span className="doc-meta">
                  {doc.num_chunks} chunk{doc.num_chunks === 1 ? "" : "s"} · {doc.num_characters.toLocaleString()} chars
                </span>
              </div>
              {/* Delete button */}
              <button className="icon-button" onClick={() => handleDelete(doc)} aria-label={`Delete ${doc.filename}`}>
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
