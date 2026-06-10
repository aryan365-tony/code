from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")).expanduser().resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


@tool
def run_python_code(code: str) -> str:
    """
    Run a Python snippet in a subprocess and return stdout/stderr.
    The code runs outside the current process; it is not exec()'d in-process.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=WORKSPACE_DIR, delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        temp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
        )
        parts: list[str] = [
            f"Return code: {proc.returncode}",
        ]
        if proc.stdout:
            parts.append("STDOUT:\n" + proc.stdout.rstrip())
        if proc.stderr:
            parts.append("STDERR:\n" + proc.stderr.rstrip())
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Error: Python execution timed out after 30 seconds."
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


register_lc_tool(run_python_code, metadata=ToolMetadata(risk_level="moderate", supports_parallel=False))
