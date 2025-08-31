from ddgs import DDGS
from agents import function_tool

@function_tool
def web_search(query: str) -> str:
    """Fetch latest info from DuckDuckGo search."""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results)
    except Exception as e:
        return f"❌ Web search failed: {e}"