from __future__ import annotations

from markitdown import MarkItDown

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata
from tools.filesystem import _safe_path

@tool
def convert_to_markdown(path: str) -> str:
    """Convert a document (PDF, DOCX, PPTX, XLSX) to markdown."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: file not found: {target}"
        
    try:
        result = MarkItDown().convert(str(target)).text_content
        if len(result) > 10000:
            return result[:10000] + "\n… [truncated]"
        return result
    except Exception as e:
        return f"Error: could not convert {target}: {e}"

register_lc_tool(convert_to_markdown, metadata=ToolMetadata(risk_level="safe", timeout_s=30.0))
