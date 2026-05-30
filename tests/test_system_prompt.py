from src.system_prompt import (
    SystemPromptBuilder,
    create_coding_assistant_prompt,
)
from src.llm.types import Tool


SAMPLE_TOOLS = [
    Tool(name="read_file", description="Read a file"),
    Tool(name="write_file", description="Write a file"),
]


class TestSystemPromptBuilder:
    def test_build_empty(self):
        builder = SystemPromptBuilder()
        assert builder.build() == ""

    def test_sections_sorted_by_priority(self):
        prompt = (
            SystemPromptBuilder()
            .add_section("Low", "low content", 10)
            .add_section("High", "high content", 90)
            .add_section("Mid", "mid content", 50)
            .build()
        )
        lines = prompt.split("\n")
        headings = [l for l in lines if l.startswith("## ")]
        assert headings == ["## High", "## Mid", "## Low"]

    def test_set_role_highest_priority(self):
        prompt = (
            SystemPromptBuilder()
            .add_section("Other", "other", 50)
            .set_role("You are a helper.")
            .build()
        )
        assert prompt.startswith("## Role")
        assert "You are a helper." in prompt

    def test_add_rules_as_bullet_list(self):
        prompt = SystemPromptBuilder().add_rules(["Rule 1", "Rule 2"]).build()
        assert "- Rule 1" in prompt
        assert "- Rule 2" in prompt

    def test_add_tool_guide(self):
        prompt = SystemPromptBuilder().add_tool_guide(SAMPLE_TOOLS).build()
        assert "**read_file**: Read a file" in prompt
        assert "**write_file**: Write a file" in prompt

    def test_set_output_constraints(self):
        prompt = SystemPromptBuilder().set_output_constraints("Be brief.").build()
        assert "## Output Format" in prompt
        assert "Be brief." in prompt

    def test_method_chaining(self):
        builder = (
            SystemPromptBuilder()
            .set_role("Helper")
            .add_rules(["Rule"])
            .add_tool_guide(SAMPLE_TOOLS)
            .set_output_constraints("Format")
        )
        assert len(builder.get_sections()) == 4

    def test_get_sections_returns_copy(self):
        builder = SystemPromptBuilder().add_section("A", "a")
        sections = builder.get_sections()
        sections.append(None)  # type: ignore
        assert len(builder.get_sections()) == 1

    def test_clear(self):
        builder = SystemPromptBuilder().set_role("Test").add_rules(["r"])
        builder.clear()
        assert len(builder.get_sections()) == 0
        assert builder.build() == ""

    def test_default_priority(self):
        builder = SystemPromptBuilder().add_section("A", "a").add_section("B", "b")
        sections = builder.get_sections()
        assert sections[0].priority == 0
        assert sections[1].priority == 0


class TestBuildWithBudget:
    def test_include_all_under_budget(self):
        prompt = (
            SystemPromptBuilder()
            .add_section("A", "short", 10)
            .add_section("B", "short", 20)
            .build_with_budget(10000)
        )
        assert "## A" in prompt
        assert "## B" in prompt

    def test_drop_low_priority_over_budget(self):
        prompt = (
            SystemPromptBuilder()
            .add_section("Important", "x" * 50, 100)
            .add_section("Nice", "y" * 50, 50)
            .add_section("Optional", "z" * 50, 10)
            .build_with_budget(130)
        )
        assert "## Important" in prompt
        assert "## Nice" in prompt
        assert "## Optional" not in prompt

    def test_always_include_first_section(self):
        prompt = (
            SystemPromptBuilder()
            .add_section("Big", "x" * 1000, 100)
            .build_with_budget(10)
        )
        assert "## Big" in prompt

    def test_empty_builder(self):
        assert SystemPromptBuilder().build_with_budget(100) == ""


class TestCreateCodingAssistantPrompt:
    def test_all_sections_present(self):
        prompt = create_coding_assistant_prompt(SAMPLE_TOOLS)
        assert "## Role" in prompt
        assert "## Rules" in prompt
        assert "## Available Tools" in prompt
        assert "## Output Format" in prompt
        assert "read_file" in prompt

    def test_role_first(self):
        prompt = create_coding_assistant_prompt(SAMPLE_TOOLS)
        assert prompt.startswith("## Role")

    def test_empty_tools(self):
        prompt = create_coding_assistant_prompt([])
        assert "## Role" in prompt
        assert "## Available Tools" in prompt