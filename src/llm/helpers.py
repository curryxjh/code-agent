from src.llm.types import ContentBlock, TextBlock, ToolUseBlock, ToolResultBlock


def extract_text(content: list[ContentBlock]) -> str:
    """Join text from all TextBlocks items"""
    return "".join(b.text for b in content if isinstance(b, TextBlock))

def extract_tool_uses(content: list[ContentBlock]) -> list[ToolUseBlock]:
    """Extract all ToolUseBlocks from content"""
    return [b for b in content if isinstance(b, ToolUseBlock)]

def create_tool_result(tool_use_id: str, content: str, is_error: bool = False) -> ToolResultBlock:
    """Create a ToolResultBlock"""
    return ToolResultBlock(
        tool_use_id=tool_use_id,
        content=content,
        is_error=is_error
    )
