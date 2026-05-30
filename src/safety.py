"""
Security and project awareness:
- File system sandbox
- Dangerous command detection
- Project config reading (CLAUDE.md)
- Git repository context injection
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


class FileSystemSandbox:
    """
    File system sandbox — restrict file operations to allowed directories.

    Prevents the agent from reading/writing sensitive paths like ~/.ssh,
    /etc/passwd, or files outside the project directory.
    """

    DEFAULT_BLOCKED = [
        re.compile(r"\.env($|\.)"),             # .env files
        re.compile(r"/(\.ssh|\.gnupg)/"),        # SSH/GPG keys
        re.compile(r"/\.git/config$"),            # git credentials
        re.compile(r"/(passwd|shadow)$"),         # system auth files
        re.compile(r"/credentials\.json$"),       # cloud credentials
        re.compile(r"/\.aws/"),                   # AWS config
    ]

    def __init__(
        self, allowed_paths: list[str], extra_blocked: list[re.Pattern] | None = None
    ) -> None:
        self._allowed_paths = [os.path.abspath(p) for p in allowed_paths]
        self._blocked_patterns = self.DEFAULT_BLOCKED + (extra_blocked or [])

    def check(self, file_path: str) -> str | None:
        """
        Check if a file path is allowed for read/write.
        Returns an error message if blocked, or None if allowed.
        """
        resolved = os.path.abspath(file_path)

        # Check blocked patterns first
        for pattern in self._blocked_patterns:
            if pattern.search(resolved):
                return f'Blocked: "{file_path}" matches a sensitive file pattern.'

        # Check if within allowed directories
        in_allowed = any(
            resolved == allowed or resolved.startswith(allowed + os.sep)
            for allowed in self._allowed_paths
        )

        if not in_allowed:
            return f'Blocked: "{file_path}" is outside allowed directories.'

        return None  # allowed

    def is_allowed(self, file_path: str) -> bool:
        """Check if a path is allowed (returns boolean)."""
        return self.check(file_path) is None


# Dangerous command patterns that require user confirmation
_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+(-[rf]+\s+|.*--no-preserve-root)"), "Recursive/forced file deletion"),
    (re.compile(r"\bgit\s+push\s+.*--force"), "Force push may overwrite remote history"),
    (re.compile(r"\bgit\s+reset\s+--hard"), "Hard reset discards uncommitted changes"),
    (re.compile(r"\bchmod\s+777\b"), "Sets world-writable permissions"),
    (re.compile(r"\bcurl\s+.*\|\s*(sh|bash)\b"), "Piping remote script to shell"),
    (re.compile(r"\bsudo\s+"), "Elevated privilege execution"),
    (re.compile(r"\b(DROP|DELETE\s+FROM|TRUNCATE)\b", re.IGNORECASE), "Destructive database operation"),
    (re.compile(r"\bkill\s+-9\b"), "Forceful process termination"),
]


def check_dangerous_command(command: str) -> str | None:
    """
    Check if a command is dangerous and needs confirmation.
    Returns the reason if dangerous, or None if safe.
    """
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def read_project_config(project_dir: str) -> str | None:
    """
    Read project configuration from a CLAUDE.md file.

    Looks for CLAUDE.md in the project root and returns its contents.
    """
    candidates = ["CLAUDE.md", os.path.join(".claude", "CLAUDE.md")]

    for candidate in candidates:
        file_path = os.path.join(project_dir, candidate)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue

    return None


@dataclass
class GitInfo:
    """Git repository information for context injection."""

    branch: str = ""
    last_commit: str = ""
    status: str = ""
    remote_url: str = ""


def parse_git_info(
    branch: str = "",
    last_commit: str = "",
    status: str = "",
    remote_url: str = "",
) -> GitInfo:
    """Parse git info from command outputs."""
    return GitInfo(
        branch=branch.strip(),
        last_commit=last_commit.strip(),
        status=status.strip(),
        remote_url=remote_url.strip(),
    )


def format_git_context(info: GitInfo) -> str:
    """Format git info for injection into system prompt."""
    lines = ["## Project Context"]

    if info.branch:
        lines.append(f"- Branch: {info.branch}")
    if info.last_commit:
        lines.append(f"- Last commit: {info.last_commit}")
    if info.remote_url:
        lines.append(f"- Remote: {info.remote_url}")
    if info.status:
        lines.append(f"- Status:\n{info.status}")

    return "\n".join(lines) if len(lines) > 1 else ""