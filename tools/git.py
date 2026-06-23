from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata
from tools.filesystem import WORKSPACE_DIR

def _run_git(*args: str) -> str:
    check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=WORKSPACE_DIR, capture_output=True, text=True
    )
    if check.returncode != 0:
        return f"Error: not a git repository: {WORKSPACE_DIR}"
        
    result = subprocess.run(
        ["git", *args], cwd=WORKSPACE_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"Error: git command failed:\n{result.stderr}"
    return result.stdout or "Success"

@tool
def git_status() -> str:
    """Get the short git status."""
    return _run_git("status", "--short")

@tool
def git_diff(path: str | None = None) -> str:
    """Get the git diff, optionally for a specific path."""
    args = ["diff"]
    if path:
        args.extend(["--", path])
    out = _run_git(*args)
    if out.startswith("Error:"):
        return out
    if len(out) > 4000:
        return out[:4000] + "\n... [truncated]"
    return out

@tool
def git_log(max_count: int = 10) -> str:
    """Get the recent git commit log."""
    return _run_git("log", "--oneline", "-n", str(max_count))

@tool
def git_commit(message: str) -> str:
    """Stage all changes and commit with the given message."""
    add_out = _run_git("add", "-A")
    if add_out.startswith("Error:"):
        return add_out
    return _run_git("commit", "-m", message)

register_lc_tool(git_status, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(git_diff, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(git_log, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(git_commit, metadata=ToolMetadata(risk_level="moderate"))
