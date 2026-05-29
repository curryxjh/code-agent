from __future__ import annotations

import pytest

from src.task import (
    TaskManager,
    execute_task_tool,
    TASK_CREATE_TOOL_DEFINITION,
    TASK_UPDATE_TOOL_DEFINITION,
    TASK_LIST_TOOL_DEFINITION,
    TASK_TOOLS,
)


class TestTaskManager:
    def setup_method(self):
        self.manager = TaskManager()

    def test_create_tasks_with_incremental_ids(self):
        id1 = self.manager.create("First task")
        id2 = self.manager.create("Second task")
        assert id1 == "task_1"
        assert id2 == "task_2"
        assert self.manager.length == 2

    def test_create_tasks_with_pending_status(self):
        self.manager.create("My task")
        task = self.manager.get("task_1")
        assert task is not None
        assert task.status == "pending"

    def test_update_task_status(self):
        self.manager.create("My task")
        ok = self.manager.update("task_1", "in_progress")
        assert ok is True
        assert self.manager.get("task_1").status == "in_progress"

    def test_return_false_for_nonexistent_task(self):
        assert self.manager.update("task_999", "completed") is False

    def test_list_all_tasks(self):
        self.manager.create("Task A")
        self.manager.create("Task B")
        assert len(self.manager.list()) == 2

    def test_filter_tasks_by_status(self):
        self.manager.create("Task A")
        self.manager.create("Task B")
        self.manager.update("task_1", "completed")
        assert len(self.manager.list("completed")) == 1
        assert len(self.manager.list("pending")) == 1

    def test_format_for_llm(self):
        self.manager.create("Read the file")
        self.manager.create("Write the output")
        self.manager.update("task_1", "completed")
        formatted = self.manager.format_for_llm()
        assert "[x] task_1: Read the file" in formatted
        assert "[ ] task_2: Write the output" in formatted

    def test_format_empty_tasks(self):
        assert self.manager.format_for_llm() == "(no tasks)"

    def test_format_in_progress_and_failed(self):
        self.manager.create("In progress")
        self.manager.create("Failed")
        self.manager.update("task_1", "in_progress")
        self.manager.update("task_2", "failed")
        formatted = self.manager.format_for_llm()
        assert "[~] task_1" in formatted
        assert "[!] task_2" in formatted

    def test_clear_all_tasks(self):
        self.manager.create("Task A")
        self.manager.create("Task B")
        self.manager.clear()
        assert self.manager.length == 0
        new_id = self.manager.create("New task")
        assert new_id == "task_1"

    def test_get_nonexistent_task(self):
        assert self.manager.get("task_999") is None


class TestExecuteTaskTool:
    def setup_method(self):
        self.manager = TaskManager()

    def test_create_task(self):
        result = execute_task_tool(
            self.manager, "task_create", {"description": "Write tests"}
        )
        assert result == "Created task_1: Write tests"
        assert self.manager.length == 1

    def test_error_for_missing_description(self):
        result = execute_task_tool(self.manager, "task_create", {})
        assert "Error" in result

    def test_update_task(self):
        self.manager.create("My task")
        result = execute_task_tool(
            self.manager, "task_update", {"id": "task_1", "status": "completed"}
        )
        assert "Updated task_1" in result

    def test_error_for_nonexistent_update(self):
        result = execute_task_tool(
            self.manager, "task_update", {"id": "task_999", "status": "completed"}
        )
        assert "not found" in result

    def test_list_tasks(self):
        self.manager.create("Task A")
        self.manager.create("Task B")
        result = execute_task_tool(self.manager, "task_list", {})
        assert "task_1" in result
        assert "task_2" in result

    def test_list_filtered_tasks(self):
        self.manager.create("Task A")
        self.manager.create("Task B")
        self.manager.update("task_1", "completed")
        result = execute_task_tool(
            self.manager, "task_list", {"status": "completed"}
        )
        assert "task_1" in result
        assert "task_2" not in result

    def test_empty_list(self):
        result = execute_task_tool(self.manager, "task_list", {})
        assert result == "(no tasks)"

    def test_unknown_tool(self):
        result = execute_task_tool(self.manager, "unknown_tool", {})
        assert "Error" in result


class TestToolDefinitions:
    def test_correct_tool_names(self):
        assert TASK_CREATE_TOOL_DEFINITION.name == "task_create"
        assert TASK_UPDATE_TOOL_DEFINITION.name == "task_update"
        assert TASK_LIST_TOOL_DEFINITION.name == "task_list"

    def test_all_tools_in_list(self):
        assert len(TASK_TOOLS) == 3
        assert [t.name for t in TASK_TOOLS] == [
            "task_create",
            "task_update",
            "task_list",
        ]