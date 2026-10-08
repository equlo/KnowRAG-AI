// TaskForm.jsx - the text box and button for adding a new task.

// useState holds the text being typed.
import { useState } from "react";

// `onAdd(title)` comes from App and resolves to true if the task was saved.
export default function TaskForm({ onAdd }) {
  // What the user has typed so far.
  const [title, setTitle] = useState("");
  // True while the request is in flight (prevents double submits).
  const [saving, setSaving] = useState(false);

  // Runs when the form is submitted (Enter key or the Add button).
  async function handleSubmit(event) {
    // Stop the browser from reloading the page, its default for forms.
    event.preventDefault();
    // Ignore leading/trailing spaces.
    const trimmed = title.trim();
    // Nothing to add, or a save is already running.
    if (!trimmed || saving) return;
    // Disable the Add button while saving.
    setSaving(true);
    // Ask App to create the task.
    const saved = await onAdd(trimmed);
    // Re-enable the button.
    setSaving(false);
    // Clear the box only if it worked, so the text isn't lost on an error.
    if (saved) setTitle("");
  }

  return (
    <form className="task-form" onSubmit={handleSubmit}>
      {/* A "controlled" input: React state is the single source of truth for its value. */}
      <input
        value={title}
        onChange={(event) => setTitle(event.target.value)}
        placeholder="What needs to be done?"
        aria-label="New task title"
        maxLength={200}
        autoFocus
      />
      {/* Disabled while saving or when the box is empty. */}
      <button type="submit" disabled={saving || !title.trim()}>
        {saving ? "Adding…" : "Add"}
      </button>
    </form>
  );
}
