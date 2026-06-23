from PySide6.QtGui import QColor

class Theme:
    BG_COLOR = QColor(8, 12, 20, 245)
    PANEL_BG = "rgba(18, 26, 40, 170)"
    PANEL_BG_HOVER = "rgba(28, 40, 58, 200)"
    HEADER_BG = "rgba(14, 20, 32, 200)"
    DOCK_BG = "rgba(14, 20, 32, 200)"

    ACCENT_CYAN = QColor(60, 220, 255)
    ACCENT_CYAN_STR = "#3CDCFF"
    ACCENT_BLUE_VIOLET = QColor(120, 90, 255)
    ACCENT_BLUE_VIOLET_STR = "#785AFF"

    TEXT_MAIN = "#E8F4FF"
    TEXT_MUTED = "#6E89AD"

    WARNING_RED = QColor(255, 80, 80)
    WARNING_RED_STR = "#FF5050"

    BORDER_RADIUS = "14px"
    BORDER_COLOR = "rgba(60, 220, 255, 45)"
    BORDER_COLOR_SOFT = "rgba(60, 220, 255, 18)"

    @staticmethod
    def get_stylesheet():
        return f"""
        QWidget {{
            color: {Theme.TEXT_MAIN};
            font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
            font-size: 13px;
        }}
        QMainWindow {{
            background: transparent;
        }}
        QToolTip {{
            background: {Theme.HEADER_BG};
            color: {Theme.ACCENT_CYAN_STR};
            border: 1px solid {Theme.BORDER_COLOR};
            padding: 4px;
        }}
        QScrollBar:vertical {{
            border: none;
            background: transparent;
            width: 6px;
            margin: 4px 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {Theme.PANEL_BG_HOVER};
            min-height: 24px;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {Theme.ACCENT_CYAN_STR};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0px;
        }}
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background: {Theme.PANEL_BG};
            border: 1px solid {Theme.BORDER_COLOR_SOFT};
            border-radius: 10px;
            padding: 10px 12px;
            color: {Theme.TEXT_MAIN};
            selection-background-color: {Theme.ACCENT_CYAN_STR};
            selection-color: #00141A;
        }}
        QTextEdit:focus, QLineEdit:focus {{
            border: 1px solid {Theme.ACCENT_CYAN_STR};
        }}
        QPushButton {{
            background: {Theme.PANEL_BG};
            border: 1px solid {Theme.BORDER_COLOR_SOFT};
            border-radius: 8px;
            padding: 6px 14px;
            color: {Theme.ACCENT_CYAN_STR};
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        QPushButton:hover {{
            background: {Theme.PANEL_BG_HOVER};
            border: 1px solid {Theme.ACCENT_CYAN_STR};
        }}
        QPushButton:pressed {{
            background: rgba(60, 220, 255, 35);
        }}
        QPushButton:disabled {{
            color: {Theme.TEXT_MUTED};
            border: 1px solid {Theme.BORDER_COLOR_SOFT};
        }}
        """
