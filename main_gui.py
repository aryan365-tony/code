import sys
import asyncio
import qasync
import signal
from pathlib import Path
from PySide6.QtWidgets import QApplication

from kernel import RuntimeKernel
from memory import MemoryManager
from integrations.gui_controller import GuiController
from gui.terminal_view import TerminalView
from gui.main_window import MainWindow
from gui.theme import Theme

# Load environment
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except ImportError:
        pass

def main():
    # Allow Ctrl+C to kill the app from the terminal instantly
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(Theme.get_stylesheet())
    
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    async def setup_backend():
        memory = await MemoryManager.create()
        kernel = await RuntimeKernel.create(memory)
        session = kernel.sessions.create()
        
        terminal_view = TerminalView()
        controller = GuiController(kernel, memory, session, terminal_view)
        
        window = MainWindow(controller)
        window.show()
        
        # Prevent garbage collection
        app._window = window
        app._kernel = kernel
        app._session = session
        app._controller = controller
        
    loop.create_task(setup_backend())
    
    try:
        with loop:
            loop.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        pass

if __name__ == "__main__":
    main()
