"""
Web Search Tools

Provides DuckDuckGo web search functionality for real-time information retrieval.
Used as fallback when local document search has low confidence or insufficient data.
"""

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


def web_search(query: str, max_results: int = 3) -> str:
    """Fetch real-time web search results using DuckDuckGo."""
    if DDGS is None:
        return "Web search unavailable: duckduckgo_search is not installed"
    results = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            for r in search_results:
                results.append(f"Source: {r['href']}\nContent: {r['body']}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"
