from __future__ import annotations
import os
from pathlib import Path
from kernel.event_bus import EventBus,EventKind,KernelEvent
from kernel.session import SessionManager
from tools.telemetry.metrics import ToolMetrics
_instance:"RuntimeKernel|None"=None
class RuntimeKernel:
    def __init__(self,memory,storage_dir:Path)->None:
        self.event_bus=EventBus()
        self.sessions=SessionManager(storage_dir)
        self.metrics=ToolMetrics()
        self.memory=memory
        self._bound_llm=None
    @classmethod
    async def create(cls,memory)->"RuntimeKernel":
        global _instance
        storage_dir=Path(os.getenv("STORAGE_DIR","./storage"))
        storage_dir.mkdir(parents=True,exist_ok=True)
        _instance=cls(memory,storage_dir)
        return _instance
    @classmethod
    def get(cls)->"RuntimeKernel":
        if _instance is None:
            raise RuntimeError("RuntimeKernel not initialized. Call RuntimeKernel.create() first.")
        return _instance
    async def shutdown(self,state=None)->None:
        if state is not None:
            self.sessions.save(state)
        await self.event_bus.emit(KernelEvent(kind=EventKind.SHUTDOWN))
        self.memory.close()
