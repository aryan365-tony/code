from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional

class EventType(Enum):
    USER_MESSAGE = auto()
    ASSISTANT_TOKEN = auto()
    ASSISTANT_THINK_TOKEN = auto()
    ASSISTANT_MESSAGE_END = auto()
    TOOL_START = auto()
    TOOL_END = auto()
    TOOL_ERROR = auto()
    STATUS_CHANGED = auto()
    VOICE_START = auto()
    VOICE_STOP = auto()
    MEMORY_UPDATED = auto()
    ERROR = auto()

class AgentState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    TOOL_RUNNING = "TOOL RUNNING"
    RESPONDING = "RESPONDING"
    ERROR = "ERROR"

@dataclass
class Event:
    type: EventType
    data: Any = None
