from src.markdown import (
    render_inline,
    render_code_block,
    render_heading,
    render_list_item,
    render_horizontal_rule,
    render_markdown,
    strip_ansi,
    BOLD,
    ITALIC,
    CYAN,
    GREEN,
    YELLOW,
    MAGENTA,
    RESET,
)


class TestStripAnsi:
    def test_remove_ansi(self):
        assert strip_ansi(f"{BOLD}hello{RESET}") == "hello"

    def test_plain_text(self):
        assert strip_ansi("hello") == "hello"


class TestRenderInline:
    def test_bold(self):
        result = render_inline("this is **bold** text")
        assert BOLD in result
        assert strip_ansi(result) == "this is bold text"

    def test_bold_underscore(self):
        result = render_inline("this is __bold__ text")
        assert strip_ansi(result) == "this is bold text"

    def test_inline_code(self):
        result = render_inline("use `console.log`")
        assert CYAN in result
        assert strip_ansi(result) == "use console.log"

    def test_italic(self):
        result = render_inline("this is *italic* text")
        assert ITALIC in result
        assert strip_ansi(result) == "this is italic text"

    def test_mixed(self):
        result = render_inline("**bold** and `code`")
        plain = strip_ansi(result)
        assert "bold" in plain
        assert "code" in plain

    def test_plain_text(self):
        assert render_inline("no formatting") == "no formatting"


class TestRenderCodeBlock:
    def test_with_language(self):
        result = render_code_block("const x = 1;", "typescript")
        plain = strip_ansi(result)
        assert "typescript" in plain
        assert "const x = 1;" in plain
        assert "┌" in plain
        assert "└" in plain

    def test_multiline(self):
        code = "line1\nline2\nline3"
        plain = strip_ansi(render_code_block(code))
        assert "line1" in plain
        assert "line2" in plain
        assert "line3" in plain

    def test_no_language(self):
        plain = strip_ansi(render_code_block("code"))
        assert "code" in plain


class TestRenderHeading:
    def test_h1(self):
        result = render_heading("Title", 1)
        assert MAGENTA in result
        assert "# Title" in strip_ansi(result)

    def test_h2(self):
        result = render_heading("Section", 2)
        assert GREEN in result
        assert "## Section" in strip_ansi(result)

    def test_h3(self):
        result = render_heading("Sub", 3)
        assert YELLOW in result
        assert "### Sub" in strip_ansi(result)


class TestRenderListItem:
    def test_bullet(self):
        plain = strip_ansi(render_list_item("item text"))
        assert "• item text" in plain

    def test_indent(self):
        plain = strip_ansi(render_list_item("nested", 4))
        assert plain.startswith("    •")

    def test_inline_formatting(self):
        plain = strip_ansi(render_list_item("**bold** item"))
        assert "bold item" in plain


class TestRenderHorizontalRule:
    def test_dashes(self):
        plain = strip_ansi(render_horizontal_rule())
        assert "─" * 48 in plain


class TestRenderMarkdown:
    def test_headings(self):
        plain = strip_ansi(render_markdown("# Title\n\n## Section"))
        assert "# Title" in plain
        assert "## Section" in plain

    def test_code_blocks(self):
        md = "```typescript\nconst x = 1;\n```"
        plain = strip_ansi(render_markdown(md))
        assert "typescript" in plain
        assert "const x = 1;" in plain

    def test_unordered_lists(self):
        md = "- item 1\n- item 2"
        plain = strip_ansi(render_markdown(md))
        assert "• item 1" in plain
        assert "• item 2" in plain

    def test_ordered_lists(self):
        md = "1. first\n2. second"
        plain = strip_ansi(render_markdown(md))
        assert "• first" in plain
        assert "• second" in plain

    def test_horizontal_rules(self):
        plain = strip_ansi(render_markdown("---"))
        assert "─" * 48 in plain

    def test_inline_in_paragraphs(self):
        plain = strip_ansi(render_markdown("This is **bold** and `code`"))
        assert "bold" in plain
        assert "code" in plain

    def test_empty_input(self):
        assert render_markdown("") == ""

    def test_unclosed_code_block(self):
        md = "```python\nprint('hi')"
        plain = strip_ansi(render_markdown(md))
        assert "print('hi')" in plain

    def test_preserve_empty_lines(self):
        md = "line1\n\nline2"
        parts = render_markdown(md).split("\n")
        assert len(parts) == 3