from __future__ import annotations

from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata


def _format_results(rows: list[tuple[str, str, str]]) -> str:
    if not rows:
        return "No results found."

    lines: list[str] = []
    for i, (title, link, snippet) in enumerate(rows, start=1):
        lines.append(f"{i}. {title}\n   {link}\n   {snippet}".rstrip())
    return "\n\n".join(lines)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo HTML results and return a compact summary.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }

    with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[tuple[str, str, str]] = []

    for item in soup.select(".result"):
        title_el = item.select_one(".result__title")
        snippet_el = item.select_one(".result__snippet")
        link_el = item.select_one(".result__url")
        anchor = item.select_one("a.result__a")

        title = anchor.get_text(" ", strip=True) if anchor else (title_el.get_text(" ", strip=True) if title_el else "")
        link = anchor.get("href", "") if anchor else (link_el.get_text(" ", strip=True) if link_el else "")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        if title and link:
            results.append((title, link, snippet))
        if len(results) >= max_results:
            break

    return _format_results(results)


register_lc_tool(web_search, metadata=ToolMetadata(risk_level="safe"))
