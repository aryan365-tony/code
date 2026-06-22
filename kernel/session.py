from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass,field
from pathlib import Path
from langchain_core.messages import BaseMessage,messages_from_dict,messages_to_dict
MAX_SAVED_MESSAGES=40
@dataclass
class SessionState:
    session_id:str=field(default_factory=lambda:uuid.uuid4().hex[:8])
    history:list[BaseMessage]=field(default_factory=list)
    turn_count:int=0
    tool_calls_total:int=0
    cancelled:bool=False
    show_tools:bool=True
    start_time:float=field(default_factory=time.monotonic)
class SessionManager:
    def __init__(self,storage_dir:Path)->None:
        self._path=storage_dir/"last_session.json"
        self._state:SessionState|None=None
    def create(self)->SessionState:
        self._state=SessionState()
        return self._state
    def load(self)->SessionState|None:
        if not self._path.exists():
            return None
        try:
            data=json.loads(self._path.read_text(encoding="utf-8"))
            s=SessionState(
                session_id=data.get("session_id",uuid.uuid4().hex[:8]),
                turn_count=data.get("turn_count",0),
                tool_calls_total=data.get("tool_calls_total",0),
            )
            raw=data.get("history",[])
            s.history=messages_from_dict(raw) if raw else []
            self._state=s
            return s
        except Exception:
            return None
    def save(self,state:SessionState)->None:
        msgs=state.history[-MAX_SAVED_MESSAGES:]
        try:
            self._path.write_text(json.dumps({
                "session_id":state.session_id,
                "turn_count":state.turn_count,
                "tool_calls_total":state.tool_calls_total,
                "history":messages_to_dict(msgs),
            },ensure_ascii=False,indent=2),encoding="utf-8")
        except Exception:
            pass
    @property
    def state(self)->SessionState|None:
        return self._state
