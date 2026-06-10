from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")).expanduser().resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_DIR / candidate
    candidate = candidate.resolve()

    if candidate != WORKSPACE_DIR and WORKSPACE_DIR not in candidate.parents:
        raise ValueError(
            f"Path outside workspace is not allowed: {path}. "
            f"Workspace root: {WORKSPACE_DIR}"
        )
    return candidate


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace sandbox."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: file not found: {target}"
    if not target.is_file():
        return f"Error: not a file: {target}"
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text: {target}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 text file inside the workspace sandbox."""
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {target}"


register_lc_tool(read_file, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(write_file, metadata=ToolMetadata(risk_level="moderate", supports_parallel=False))
