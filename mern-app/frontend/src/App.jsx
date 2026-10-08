// App.jsx - the top-level component: the "R" (React) side of MERN.
// It owns the list of tasks, loads it from the API, and passes data and
// callbacks down to the smaller components.

// useState stores values that re-render the UI when they change; useEffect runs side effects.
import { useEffect, useState } from "react";
// The functions that talk to the Express API.
import { createTask, deleteTask, getTasks, updateTask } from "./api.js";
// The "add a task" input.
import TaskForm from "./components/TaskForm.jsx";
// The list of tasks.
import TaskList from "./components/TaskList.jsx";

// The filter buttons: each name maps to a test a task must pass to be shown.
const FILTERS = {
  // Show everything.
  all: () => true,
  // Only tasks that are not done yet.
  active: (task) => !task.completed,
  // Only finished tasks.
  completed: (task) => task.completed,
};

// The App component. React calls this function to get the UI to display.
export default function App() {
  // Every task loaded from the server.
  const [tasks, setTasks] = useState([]);
  // True until the first load finishes.
  const [loading, setLoading] = useState(true);
  // The latest error message to show ("" means no error).
  const [error, setError] = useState("");
  // Which filter button is selected.
  const [filter, setFilter] = useState("all");

  // Load the tasks once, when the component first appears on screen.
  useEffect(() => {
    // Set to true if the component goes away before the request finishes.
    let ignore = false;
    // GET /api/tasks...
    getTasks()
      // ...store the result...
      .then((data) => !ignore && setTasks(data))
      // ...or show what went wrong...
      .catch((err) => !ignore && setError(err.message))
      // ...and stop showing "Loading" either way.
      .finally(() => !ignore && setLoading(false));
    // Cleanup function: React runs it when the component unmounts.
    return () => {
      ignore = true;
    };
    // The empty array means "run only once, after the first render".
  }, []);

  // Run an API action. Clears the error on success, shows it on failure,
  // and returns true/false so callers know whether it worked.
  async function run(action) {
    try {
      // Wait for the request and the state update inside `action`.
      await action();
      // Success: hide any old error.
      setError("");
      return true;
    } catch (err) {
      // Failure: show the message from api.js.
      setError(err.message);
      return false;
    }
  }

  // Swap one task in the list for the updated copy the server sent back.
  function replaceTask(updated) {
    // Use the "updater" form of setTasks so we always work on the latest list.
    setTasks((current) => current.map((task) => (task._id === updated._id ? updated : task)));
  }

  // Create a task and put it at the top of the list.
  const handleAdd = (title) =>
    run(async () => {
      const task = await createTask(title);
      setTasks((current) => [task, ...current]);
    });

  // Flip a task between done and not done.
  const handleToggle = (task) => run(async () => replaceTask(await updateTask(task._id, { completed: !task.completed })));

  // Give a task a new title.
  const handleRename = (task, title) => run(async () => replaceTask(await updateTask(task._id, { title })));

  // Delete a task and remove it from the list.
  const handleDelete = (task) =>
    run(async () => {
      await deleteTask(task._id);
      setTasks((current) => current.filter((item) => item._id !== task._id));
    });

  // The tasks that pass the selected filter.
  const visibleTasks = tasks.filter(FILTERS[filter]);
  // How many tasks are still to do.
  const remaining = tasks.filter(FILTERS.active).length;

  // Describe the UI in JSX (HTML-like syntax that compiles to JavaScript).
  return (
    <main className="app">
      {/* Page title. */}
      <header className="app-header">
        <h1>MERN Tasks</h1>
        <p className="subtitle">MongoDB · Express · React · Node.js</p>
      </header>

      {/* The main card. */}
      <section className="card">
        {/* Input for new tasks; it calls handleAdd on submit. */}
        <TaskForm onAdd={handleAdd} />

        {/* Error banner, shown only when there is an error. role="alert" makes screen readers announce it. */}
        {error && (
          <div className="error" role="alert">
            <span>{error}</span>
            <button type="button" className="icon" onClick={() => setError("")} aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}

        {/* While loading show a message, afterwards show the list. */}
        {loading ? (
          <p className="empty">Loading tasks…</p>
        ) : (
          <TaskList
            tasks={visibleTasks}
            emptyMessage={tasks.length === 0 ? "No tasks yet. Add one above!" : `No ${filter} tasks.`}
            onToggle={handleToggle}
            onRename={handleRename}
            onDelete={handleDelete}
          />
        )}

        {/* Counter and filter buttons. */}
        <footer className="card-footer">
          <span>
            {remaining} {remaining === 1 ? "task" : "tasks"} left
          </span>
          <div className="filters" role="group" aria-label="Filter tasks">
            {/* One button per filter name. `key` helps React track list items. */}
            {Object.keys(FILTERS).map((name) => (
              <button
                key={name}
                type="button"
                className={name === filter ? "active" : ""}
                aria-pressed={name === filter}
                onClick={() => setFilter(name)}
              >
                {name}
              </button>
            ))}
          </div>
        </footer>
      </section>
    </main>
  );
}
