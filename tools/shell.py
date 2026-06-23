from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata
from tools.filesystem import WORKSPACE_DIR

@tool
def run_shell_command(command: str) -> str:
    """Run a shell command in the workspace directory."""
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKSPACE_DIR, capture_output=True, text=True
        )
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        output.append(f"Return Code: {result.returncode}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Error: {e}"

register_lc_tool(run_shell_command, metadata=ToolMetadata(risk_level="destructive", timeout_s=20.0, retry_limit=0))
