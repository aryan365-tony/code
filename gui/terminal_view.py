import html
from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtCore import Qt

from gui.theme import Theme

class TerminalView(QTextEdit):
    """
    A pure text-based terminal view.
    Accepts HTML appends and acts like a linear log.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrameStyle(0)
        self.setStyleSheet(f"""
            QTextEdit {{
                background: transparent;
                border: none;
                padding: 18px;
                selection-background-color: {Theme.ACCENT_CYAN_STR};
                selection-color: #00141A;
            }}
        """)
        
        font = QFont()
        font.setFamilies(["Cascadia Code", "Fira Code", "Consolas", "Menlo", "monospace"])
        font.setPointSize(11)
        self.setFont(font)
        
        # Word wrap to avoid horizontal scrolling
        self.setLineWrapMode(QTextEdit.WidgetWidth)

    def write_raw_html(self, html_text: str):
        """Append raw HTML via the document cursor (bypasses read-only)."""
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html_text)
        self._scroll_to_bottom()

    def print_user(self, text: str):
        self.write_raw_html("<br><br>")
        safe_text = html.escape(text).replace("\n", "<br>")
        self.write_raw_html(
            f'<span style="color:{Theme.ACCENT_CYAN_STR};font-weight:bold;">You  </span>'
            f'<span style="color:{Theme.TEXT_MAIN};">{safe_text}</span>'
        )

    def print_assistant_label(self):
        self.write_raw_html("<br><br>")
        self.write_raw_html(f'<span style="color:#44DD88;font-weight:bold;">Assistant </span>')

    def print_assistant_token(self, text: str):
        safe = html.escape(text).replace("\n", "<br>")
        self.write_raw_html(f'<span style="color:{Theme.TEXT_MAIN};">{safe}</span>')

    def print_thinking_header(self):
        self.write_raw_html("<br><br>")
        self.write_raw_html(f'<span style="color:#6E89AD;">╭─ 💭 Thinking ─────────────────────────────────────────</span><br>')

    def print_thinking_token(self, text: str):
        safe = html.escape(text).replace("\n", "<br>")
        self.write_raw_html(f'<span style="color:#6E89AD;font-style:italic;">{safe}</span>')

    def print_thinking_footer(self, chars: int):
        self.write_raw_html(f'<br><span style="color:#6E89AD;">╰──────────────────────── {chars:,} chars ─╯</span><br>')

    def print_tool_call(self, tool_name: str, args_json: str):
        self.write_raw_html("<br><br>")
        header = f"╭─ 🔧 Tool Call ─╮<br>│ {tool_name:<14} │<br>╰────────────────╯"
        safe_json = html.escape(args_json).replace("\n", "<br>")
        self.write_raw_html(
            f'<span style="color:{Theme.ACCENT_CYAN_STR};font-weight:bold;">{header}</span><br>'
            f'<span style="color:{Theme.TEXT_MAIN};">{safe_json}</span><br>'
        )

    def print_tool_result(self, tool_name: str, result: str, elapsed: float | None):
        self.write_raw_html("<br>")
        title = f"📄 Result • {tool_name}"
        if elapsed is not None:
            title += f" ({elapsed:.2f}s)"
            
        safe_result = html.escape(result).replace("\n", "<br>")
        self.write_raw_html(
            f'<span style="color:{Theme.TEXT_MUTED};font-weight:bold;">╭─ {title} ─╮</span><br>'
            f'<span style="color:{Theme.TEXT_MUTED};">{safe_result}</span><br>'
        )

    def print_system(self, text: str, error: bool = False):
        color = Theme.WARNING_RED_STR if error else "#6E89AD"
        prefix = "✗ " if error else ""
        safe = html.escape(text).replace("\n", "<br>")
        self.write_raw_html(f'<br><span style="color:{color};font-weight:bold;">{prefix}{safe}</span>')

    def clear(self):
        super().clear()

    def _scroll_to_bottom(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
