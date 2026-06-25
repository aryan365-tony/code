from __future__ import annotations

import concurrent.futures
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata


def _scrape_site(url: str, title: str, max_chars: int = 4000) -> str:
    """Scrape a single site and extract text."""
    try:
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
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.extract()
            
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        
        if len(text) > max_chars:
            text = text[:max_chars] + "… [truncated]"
            
        return f"--- SITE: {title} ---\nURL: {url}\nCONTENT:\n{text}\n"
    except Exception as e:
        return f"--- SITE: {title} ---\nURL: {url}\nERROR: {e}\n"


@tool
def web_search(query: str, max_results: int = 10) -> str:
    """
    Search the web using DuckDuckGo HTML results and perform deep RPA scraping 
    on the top results to extract comprehensive website contents.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception as e:
        return f"Search Error: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    search_results: list[tuple[str, str]] = []

    for item in soup.select(".result"):
        title_el = item.select_one(".result__title")
        link_el = item.select_one(".result__url")
        anchor = item.select_one("a.result__a")

        title = anchor.get_text(" ", strip=True) if anchor else (title_el.get_text(" ", strip=True) if title_el else "")
        link = anchor.get("href", "") if anchor else (link_el.get_text(" ", strip=True) if link_el else "")

        if title and link:
            search_results.append((title, link))
        if len(search_results) >= max_results:
            break

    if not search_results:
        return "No results found."

    # Concurrent deep scraping of the top results
    scraped_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_results) as executor:
        futures = {
            executor.submit(_scrape_site, link, title): (title, link)
            for title, link in search_results
        }
        for future in concurrent.futures.as_completed(futures):
            scraped_data.append(future.result())

    return "\n\n".join(scraped_data)


register_lc_tool(web_search, metadata=ToolMetadata(risk_level="safe", timeout_s=60.0))
