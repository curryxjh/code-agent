from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.types import Tool


@dataclass
class PromptSection:
    """A named section in the system prompt."""

    title: str
    content: str
    priority: int = 0  # higher = more important, kept first when truncating


class SystemPromptBuilder:
    """
    Build structured system prompts from composable sections.

    Sections are rendered in priority order (highest first), each with
    a markdown heading. Tool descriptions can be auto-generated from
    Tool definitions.
    """

    def __init__(self) -> None:
        self._sections: list[PromptSection] = []

    def add_section(self, title: str, content: str, priority: int = 0) -> "SystemPromptBuilder":
        """Add a named section with optional priority (default 0)."""
        self._sections.append(PromptSection(title=title, content=content, priority=priority))
        return self

    def set_role(self, role: str) -> "SystemPromptBuilder":
        """Set the agent's role/identity."""
        return self.add_section("Role", role, 100)

    def add_rules(self, rules: list[str]) -> "SystemPromptBuilder":
        """Add behavioral rules."""
        content = "\n".join(f"- {r}" for r in rules)
        return self.add_section("Rules", content, 80)

    def add_tool_guide(self, tools: list[Tool]) -> "SystemPromptBuilder":
        """Auto-generate tool usage guide from Tool definitions."""
        lines = [f"- **{t.name}**: {t.description}" for t in tools]
        return self.add_section("Available Tools", "\n".join(lines), 60)

    def set_output_constraints(self, constraints: str) -> "SystemPromptBuilder":
        """Add output format constraints."""
        return self.add_section("Output Format", constraints, 40)

    def build(self) -> str:
        """Build the final prompt string, sorted by priority (high -> low)."""
        sorted_sections = sorted(self._sections, key=lambda s: s.priority, reverse=True)
        return "\n\n".join(f"## {s.title}\n{s.content}" for s in sorted_sections)

    def build_with_budget(self, max_chars: int) -> str:
        """Build with a character budget — drop lowest-priority sections if over limit."""
        sorted_sections = sorted(self._sections, key=lambda s: s.priority, reverse=True)

        parts: list[str] = []
        total = 0

        for section in sorted_sections:
            block = f"## {section.title}\n{section.content}"
            if total + len(block) + 2 > max_chars and parts:
                break  # stop adding sections when budget exceeded
            parts.append(block)
            total += len(block) + 2  # +2 for "\n\n" separator

        return "\n\n".join(parts)

    def get_sections(self) -> list[PromptSection]:
        """Get all sections (for inspection/testing)."""
        return list(self._sections)

    def clear(self) -> "SystemPromptBuilder":
        """Clear all sections."""
        self._sections.clear()
        return self


def create_coding_assistant_prompt(tools: list[Tool]) -> str:
    """Create a pre-configured system prompt for a coding assistant."""
    return (
        SystemPromptBuilder()
        .set_role(
            "You are a coding assistant. Help the user with software engineering tasks "
            "by reading files, writing code, and running commands. Be concise and accurate."
        )
        .add_rules([
            "Always read a file before modifying it.",
            "Explain what you are about to do before using tools.",
            "If a task is complex, break it into steps and track progress with task tools.",
            "Never execute destructive commands without confirmation.",
        ])
        .add_tool_guide(tools)
        .set_output_constraints(
            "Respond in the user's language. Use markdown for code blocks. "
            "Keep explanations brief and focused."
        )
        .build()
    )