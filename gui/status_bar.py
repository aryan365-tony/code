from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
from events.event_types import AgentState
from gui.theme import Theme

class StatusBar(QWidget):
    clear_clicked = Signal()
    stop_clicked = Signal()
    settings_clicked = Signal()

    def __init__(self, model_name: str = "", parent=None):
        super().__init__(parent)
        self.setFixedHeight(54)
        self.setObjectName("StatusBar")
        self.setStyleSheet(f"""
            #StatusBar {{
                background: {Theme.HEADER_BG};
                border-bottom: 1px solid {Theme.BORDER_COLOR_SOFT};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 12, 0)
        layout.setSpacing(10)

        self.name_label = QLabel("J A R V I S")
        self.name_label.setStyleSheet(f"color: {Theme.ACCENT_CYAN_STR}; font-weight: 700; font-size: 15px; letter-spacing: 3px;")
        layout.addWidget(self.name_label)

        self.model_label = QLabel(model_name.upper())
        self.model_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(self.model_label)

        self.state_badge = QLabel("IDLE")
        self.state_badge.setAlignment(Qt.AlignCenter)
        self.state_badge.setFixedSize(110, 24)
        layout.addWidget(self.state_badge)

        layout.addStretch()

        self.stop_btn = self._tool_button("STOP")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.stop_btn)

        self.clear_btn = self._tool_button("CLEAR")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        layout.addWidget(self.clear_btn)

        self.settings_btn = self._icon_button("⚙")
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)

        self.minimize_btn = self._icon_button("—")
        self.minimize_btn.clicked.connect(lambda: self.window().showMinimized())
        layout.addWidget(self.minimize_btn)

        self.close_btn = self._icon_button("✕")
        self.close_btn.setStyleSheet(self.close_btn.styleSheet() + f"""
            QPushButton:hover {{ color: {Theme.WARNING_RED_STR}; border: 1px solid {Theme.WARNING_RED_STR}; }}
        """)
        self.close_btn.clicked.connect(lambda: self.window().close())
        layout.addWidget(self.close_btn)

        self._set_badge_style(Theme.TEXT_MUTED, Theme.BORDER_COLOR_SOFT)

    def _tool_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(28)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _icon_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(28, 28)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 0px;
                color: {Theme.TEXT_MUTED};
            }}
            QPushButton:hover {{
                color: {Theme.ACCENT_CYAN_STR};
                border: 1px solid {Theme.BORDER_COLOR_SOFT};
            }}
        """)
        return btn

    def _set_badge_style(self, color: str, border: str):
        self.state_badge.setStyleSheet(f"""
            background: {Theme.PANEL_BG};
            border: 1px solid {border};
            border-radius: 12px;
            color: {color};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
        """)

    def set_state(self, state: AgentState):
        self.state_badge.setText(state.value)
        busy = state in (AgentState.THINKING, AgentState.TOOL_RUNNING, AgentState.RESPONDING)
        self.stop_btn.setEnabled(busy)
        if state == AgentState.IDLE:
            self._set_badge_style(Theme.TEXT_MUTED, Theme.BORDER_COLOR_SOFT)
        elif state == AgentState.ERROR:
            self._set_badge_style(Theme.WARNING_RED_STR, Theme.WARNING_RED_STR)
        else:
            self._set_badge_style(Theme.ACCENT_CYAN_STR, Theme.ACCENT_CYAN_STR)
