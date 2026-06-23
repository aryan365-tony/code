import asyncio
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QTextEdit, QPushButton, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QColor, QKeySequence, QShortcut

from gui.theme import Theme
from gui.status_bar import StatusBar
from gui.terminal_view import TerminalView
from llm import get_active_model_name

class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._active_task = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(900, 700)
        self.setMinimumSize(700, 500)

        self.central_widget = QWidget()
        self.central_widget.setObjectName("CentralWidget")
        self.central_widget.setStyleSheet(f"""
            #CentralWidget {{
                background-color: {Theme.BG_COLOR.name(QColor.HexArgb)};
                border-radius: 16px;
                border: 1px solid {Theme.BORDER_COLOR_SOFT};
            }}
        """)
        self.setCentralWidget(self.central_widget)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(60, 220, 255, 60))
        shadow.setOffset(0, 0)
        self.central_widget.setGraphicsEffect(shadow)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.status_bar = StatusBar(model_name=get_active_model_name())
        self.status_bar.clear_clicked.connect(self.clear_chat)
        self.status_bar.stop_clicked.connect(self.stop_generation)
        self.main_layout.addWidget(self.status_bar)

        self.terminal_view = self.controller.terminal_view
        self.main_layout.addWidget(self.terminal_view, stretch=1)

        self.bottom_dock = QWidget()
        self.bottom_dock.setObjectName("BottomDock")
        self.bottom_dock.setStyleSheet(f"""
            #BottomDock {{
                background: {Theme.DOCK_BG};
                border-top: 1px solid {Theme.BORDER_COLOR_SOFT};
            }}
        """)
        self.bottom_layout = QHBoxLayout(self.bottom_dock)
        self.bottom_layout.setContentsMargins(16, 12, 16, 16)
        self.bottom_layout.setSpacing(10)

        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(56)
        self.input_box.setPlaceholderText("Message Assistant... (Enter to send, Shift+Enter for new line)")
        self.input_box.installEventFilter(self)

        self.send_btn = QPushButton("SEND")
        self.send_btn.setFixedSize(72, 44)
        self.send_btn.clicked.connect(self.send_message)

        self.bottom_layout.addWidget(self.input_box)
        self.bottom_layout.addWidget(self.send_btn)
        self.main_layout.addWidget(self.bottom_dock)

        self.drag_pos = None
        self._setup_shortcuts()

        # Connect controller state signal
        self.controller.state_changed.connect(self.handle_state_changed)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Esc"), self, self.stop_generation)
        QShortcut(QKeySequence("Ctrl+L"), self, self.clear_chat)
        QShortcut(QKeySequence("Ctrl+K"), self, lambda: self.input_box.setFocus())

    def eventFilter(self, obj, event):
        if obj == self.input_box and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    def send_message(self):
        if self._active_task and not self._active_task.done():
            return
            
        text = self.input_box.toPlainText()
        if not text.strip():
            return

        self.input_box.clear()

        loop = asyncio.get_event_loop()
        self._active_task = loop.create_task(self.controller.process_input(text))

    def stop_generation(self):
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()

    def clear_chat(self):
        loop = asyncio.get_event_loop()
        self._active_task = loop.create_task(self.controller.process_input("/clear"))

    def handle_state_changed(self, state: str):
        # Fake an AgentState enum object for the status bar
        class FakeState:
            value = state
        self.status_bar.set_state(FakeState())
        
        is_idle = (state == "IDLE" or state == "ERROR")
        self.input_box.setEnabled(is_idle)
        self.send_btn.setEnabled(is_idle)
        
        if is_idle:
            self._active_task = None
            self.input_box.setFocus()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.status_bar.geometry().contains(event.position().toPoint()):
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_pos = None
