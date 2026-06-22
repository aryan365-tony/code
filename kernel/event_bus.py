from __future__ import annotations
import asyncio
from dataclasses import dataclass,field
from enum import auto,Enum
from typing import Any,Callable
class EventKind(Enum):
    TURN_START=auto()
    TURN_END=auto()
    TOOL_START=auto()
    TOOL_END=auto()
    SESSION_SAVE=auto()
    SHUTDOWN=auto()
@dataclass
class KernelEvent:
    kind:EventKind
    payload:dict[str,Any]=field(default_factory=dict)
class EventBus:
    def __init__(self)->None:
        self._subs:dict[EventKind,list[Callable]]={}
    def subscribe(self,kind:EventKind,cb:Callable)->None:
        self._subs.setdefault(kind,[]).append(cb)
    async def emit(self,event:KernelEvent)->None:
        for cb in self._subs.get(event.kind,[]):
            r=cb(event)
            if asyncio.iscoroutine(r):
                await r
