// TaskItem.jsx - one task: a checkbox, its title, and Edit/Delete buttons.
// Clicking Edit swaps the title for a text box until you save or cancel.

// useState tracks edit mode and the text being edited.
import { useState } from "react";

// `task` is the task to show; the callbacks come from App.
export default function TaskItem({ task, onToggle, onRename, onDelete }) {
  // Is this row currently being edited?
  const [editing, setEditing] = useState(false);
  // The edited title (only used in edit mode).
  const [draft, setDraft] = useState(task.title);

  // Enter edit mode, starting from the current title.
  function startEditing() {
    setDraft(task.title);
    setEditing(true);
  }

  // Save the new title (Enter key or the Save button).
  async function handleSave(event) {
    // Stop the browser from reloading the page.
    event.preventDefault();
    // Ignore leading/trailing spaces.
    const trimmed = draft.trim();
    // Empty or unchanged: just leave edit mode without calling the API.
    if (!trimmed || trimmed === task.title) return setEditing(false);
    // Leave edit mode only if the server accepted the change.
    if (await onRename(task, trimmed)) setEditing(false);
  }

  // Edit mode: a small form instead of the title.
  if (editing) {
    return (
      <li className="task-item">
        <form className="edit-form" onSubmit={handleSave}>
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            // Escape cancels editing.
            onKeyDown={(event) => event.key === "Escape" && setEditing(false)}
            aria-label="Edit task title"
            maxLength={200}
            autoFocus
          />
          <button type="submit">Save</button>
          <button type="button" className="secondary" onClick={() => setEditing(false)}>
            Cancel
          </button>
        </form>
      </li>
    );
  }

  // Normal mode. The "done" class strikes the title through.
  return (
    <li className={task.completed ? "task-item done" : "task-item"}>
      {/* Wrapping the checkbox in a <label> makes the title clickable too. */}
      <label className="task-check">
        <input type="checkbox" checked={task.completed} onChange={() => onToggle(task)} />
        <span className="task-title">{task.title}</span>
      </label>
      <div className="task-actions">
        {/* aria-label tells screen-reader users WHICH task the button acts on. */}
        <button type="button" className="secondary" onClick={startEditing} aria-label={`Edit "${task.title}"`}>
          Edit
        </button>
        <button type="button" className="danger" onClick={() => onDelete(task)} aria-label={`Delete "${task.title}"`}>
          Delete
        </button>
      </div>
    </li>
  );
}
