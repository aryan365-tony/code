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


@tool
def list_dir(path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories under path, optionally recursive and filtered by glob pattern."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: directory not found: {target}"
    if not target.is_dir():
        return f"Error: not a directory: {target}"
    
    try:
        iterator = target.rglob(pattern) if recursive else target.glob(pattern)
        results = []
        for p in iterator:
            rel = p.relative_to(WORKSPACE_DIR)
            type_str = "dir " if p.is_dir() else "file"
            results.append(f"{type_str} {rel}")
        return "\n".join(results) if results else "No matches found."
    except Exception as e:
        return f"Error: {e}"


@tool
def search_files(query: str, path: str = ".", glob_pattern: str = "*", max_results: int = 30) -> str:
    """Search for case-insensitive substring query inside files matching glob_pattern."""
    target = _safe_path(path)
    if not target.exists() or not target.is_dir():
        return f"Error: directory not found: {target}"
    
    query_lower = query.lower()
    results = []
    
    try:
        for p in target.rglob(glob_pattern):
            if not p.is_file():
                continue
            try:
                content = p.read_text(encoding="utf-8")
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if query_lower in line.lower():
                        rel = p.relative_to(WORKSPACE_DIR)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            return "\n".join(results)
            except UnicodeDecodeError:
                pass  # Skip binary files
    except Exception as e:
        return f"Error: {e}"
        
    return "\n".join(results) if results else "No matches found."


@tool
def move_file(src: str, dst: str) -> str:
    """Move or rename a file or directory."""
    src_path = _safe_path(src)
    dst_path = _safe_path(dst)
    
    if not src_path.exists():
        return f"Error: source not found: {src_path}"
    if dst_path.exists():
        return f"Error: destination already exists: {dst_path}"
        
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.rename(dst_path)
        return f"Moved {src_path.relative_to(WORKSPACE_DIR)} to {dst_path.relative_to(WORKSPACE_DIR)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def delete_file(path: str) -> str:
    """Delete a file. Refuses to delete directories."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: not found: {target}"
    if target.is_dir():
        return f"Error: cannot delete directory: {target}"
        
    try:
        target.unlink()
        return f"Deleted file: {target.relative_to(WORKSPACE_DIR)}"
    except Exception as e:
        return f"Error: {e}"


register_lc_tool(list_dir, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(search_files, metadata=ToolMetadata(risk_level="safe", timeout_s=20))
register_lc_tool(move_file, metadata=ToolMetadata(risk_level="moderate"))
register_lc_tool(delete_file, metadata=ToolMetadata(risk_level="destructive"))
