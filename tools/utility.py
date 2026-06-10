from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata


@tool
def get_current_time() -> str:
    """Return the current date and time in ISO-8601 format with timezone."""
    return datetime.now(timezone.utc).astimezone().isoformat()


register_lc_tool(get_current_time, metadata=ToolMetadata(risk_level="safe"))
