from dataclasses import dataclass

TaskStatus = str # "pending", "in_progress", "completed", "failed"

@dataclass
class Task:
    """A single task in the plan."""
    id: str
    description: str
    status: TaskStatus = "pending"


class TaskManager:
    """Manages a list of tasks for the agent."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []
        self._next_id: int = 1

    def create(self, description: str) -> str:
        """Create a new task and return its ID."""
        task_id = f"task_{self._next_id}"
        self._next_id += 1
        self._tasks.append(Task(id=task_id, description=description))
        return task_id

    def update(self, task_id: str, status: TaskStatus) -> bool:
        """Update the status of a task. Returns False if not found."""
        for task in self._tasks:
            if task.id == task_id:
                task.status = status
                return True

        return False

    def get(self, task_id: str) -> Task | None:
        """Get a single task by ID."""
        for task in self._tasks:
            if task.id == task_id:
                return task

        return None

    def list(self, status: TaskStatus | None = None) -> list[Task]:
        """List all tasks, optionally filtered by status."""
        if status:
            return [t for t in self._tasks if t.status == status]
        return list(self._tasks)

    def format_for_llm(self) -> str:
        """Format tasks as a readable string for LLM context."""
        if not self._tasks:
            return "(no tasks)"
        lines = []
        for t in self._tasks:
            icon = {
                "completed": "[x]",
                "in_progress": "[~]",
                "failed": "[!]",
            }.get(t.status, "[ ]")
            lines.append(f"{icon} {t.id}: {t.description}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()
        self._next_id = 1

    @property
    def length(self) -> int:
        return len(self._tasks)
