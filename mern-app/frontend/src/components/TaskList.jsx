// TaskList.jsx - renders the list of tasks, or a message when there are none.

// One row of the list.
import TaskItem from "./TaskItem.jsx";

// Props come from App: the tasks to show, a fallback message, and the action callbacks.
export default function TaskList({ tasks, emptyMessage, onToggle, onRename, onDelete }) {
  // Nothing to show: display the message instead of an empty list.
  if (tasks.length === 0) return <p className="empty">{emptyMessage}</p>;

  return (
    <ul className="task-list">
      {/* One <TaskItem> per task. The MongoDB _id is a stable, unique key. */}
      {tasks.map((task) => (
        <TaskItem key={task._id} task={task} onToggle={onToggle} onRename={onRename} onDelete={onDelete} />
      ))}
    </ul>
  );
}
