from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.tools import BaseTool

RiskLevel = Literal["safe", "moderate", "destructive"]


@dataclass(frozen=True)
class ToolMetadata:
    risk_level: RiskLevel = "safe"
    supports_parallel: bool = True
    timeout_s: float = 30.0
    retry_limit: int = 0
    retry_backoff_base: float = 1.5


@dataclass
class ToolEntry:
    name: str
    tool: BaseTool
    description: str
    metadata: ToolMetadata = field(default_factory=ToolMetadata)


_registry: dict[str, ToolEntry] = {}
_lc_tools: list[BaseTool] = []


def register_lc_tool(tool: BaseTool, *, metadata: ToolMetadata | None = None) -> BaseTool:
    """
    Register a LangChain BaseTool so the executor can find it by name.
    """
    entry = ToolEntry(
        name=tool.name,
        tool=tool,
        description=getattr(tool, "description", "") or "",
        metadata=metadata or ToolMetadata(),
    )
    _registry[entry.name] = entry
    _lc_tools.append(tool)
    return tool


def get_tool(name: str) -> BaseTool | None:
    entry = _registry.get(name)
    return entry.tool if entry else None


def get_tool_entry(name: str) -> ToolEntry | None:
    return _registry.get(name)


def get_lc_tools() -> list[BaseTool]:
    return list(_lc_tools)


def get_registered_tool_names() -> list[str]:
    return sorted(_registry.keys())


def list_tools_text() -> str:
    if not _registry:
        return "No tools registered."

    lines: list[str] = []
    for name in sorted(_registry):
        entry = _registry[name]
        lines.append(
            f"- {name}: {entry.description or 'No description'} "
            f"[risk={entry.metadata.risk_level}, parallel={entry.metadata.supports_parallel}]"
        )
    return "\n".join(lines)
