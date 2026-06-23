from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata

@tool
def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch URL, strip scripts/styles, and return text content."""
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style"]):
            element.extract()
            
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        
        if len(text) > max_chars:
            return text[:max_chars] + "… [truncated]"
        return text
    except Exception as e:
        return f"Error: {e}"

@tool
def http_request(method: str, url: str, headers: dict | None = None, json_body: dict | None = None, max_chars: int = 4000) -> str:
    """Make an HTTP request and return status and body."""
    if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        return f"Error: invalid method: {method}"
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
        
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.request(
                method.upper(), url, headers=headers, json=json_body
            )
            
        body = response.text
        if len(body) > max_chars:
            body = body[:max_chars] + "… [truncated]"
            
        return f"Status: {response.status_code}\n\n{body}"
    except Exception as e:
        return f"Error: {e}"

register_lc_tool(fetch_url, metadata=ToolMetadata(risk_level="safe", timeout_s=20.0))
register_lc_tool(http_request, metadata=ToolMetadata(risk_level="moderate", timeout_s=20.0))
