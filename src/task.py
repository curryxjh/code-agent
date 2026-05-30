from __future__ import annotations

from dataclasses import dataclass

from src.llm import Tool

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

# Tool definitions for task management
TASK_CREATE_TOOL_DEFINITION = Tool(
    name="task_create",
    description="Create a new task in the plan. Returns the task ID.",
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Description of the task to create",
            },
        },
        "required": ["description"],
    },
)

TASK_UPDATE_TOOL_DEFINITION = Tool(
    name="task_update",
    description=(
        'Update the status of an existing task. '
        'Status can be "pending", "in_progress", "completed", or "failed".'
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The task ID to update",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "failed"],
                "description": "The new status for the task",
            },
        },
        "required": ["id", "status"],
    },
)

TASK_LIST_TOOL_DEFINITION = Tool(
    name="task_list",
    description=(
        "List all tasks in the current plan with their status. "
        "Optionally filter by status."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "failed"],
                "description": "Filter tasks by status (optional)",
            },
        },
    },
)

def execute_task_tool(
        manager: TaskManager, name: str, input: dict
) -> str:
    """Execute a task management tool."""
    if name == "task_create":
        desc = input.get("description", "")
        if not desc:
            return "Error: description is required"
        task_id = manager.create(desc)
        return f"Created {task_id}: {desc}"
    elif name == "task_update":
        task_id = input.get("id", "")
        status = input.get("status", "")
        if not task_id or not status:
            return "Error: id and status are required"
        ok = manager.update(task_id, status)
        return f"Updated {task_id} → {status}" if ok else f"Error: task {task_id} not found"
    elif name == "task_list":
        status = input.get("status")
        tasks = manager.list(status)
        if not tasks:
            return "(no tasks)"
        return "\n".join(f"{t.id} [{t.status}]: {t.description}" for t in tasks)
    else:
        return f'Error: unknown task tool "{name}"'

TASK_TOOLS = [
    TASK_CREATE_TOOL_DEFINITION,
    TASK_UPDATE_TOOL_DEFINITION,
    TASK_LIST_TOOL_DEFINITION,
]

