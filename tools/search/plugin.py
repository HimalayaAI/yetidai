import asyncio
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from typing import Dict, Any

from core.tool_contracts import (
    ToolCategory,
    ToolContext,
    ToolParam,
    ToolResult,
    ToolSpec,
)
from core.tool_registry import get_registry

SEARCH_SPEC = ToolSpec(
    tool_id="search.internet",
    name="internet_search",
    description=(
        "FALLBACK tool — general-purpose DuckDuckGo web search. Use this ONLY "
        "when the question is clearly NOT about Nepal, OR when "
        "get_nepal_live_context already ran and returned no useful data.\n\n"
        "USE FOR:\n"
        "  • Non-Nepal people (foreign heads of state, global CEOs, "
        "international celebrities, historical figures).\n"
        "  • World facts & trivia (capitals, sports results, scientific "
        "constants, book/movie info).\n"
        "  • Recent non-Nepal events or products.\n\n"
        "DO NOT USE FOR (use get_nepal_live_context instead):\n"
        "  • Nepal ministers, PM, officials — NepalOSINT tracks them live.\n"
        "  • Nepal macro data (inflation, remittance, reserves, trade).\n"
        "  • NEPSE, stocks, IPOs, dividends.\n"
        "  • Public debt, parliament bills, government decisions.\n"
        "  • Nepal news (current or historical).\n\n"
        "OUTPUT: up to 5 English snippets with source URLs. You MUST "
        "summarize / translate these into Nepali before replying to the user. "
        "Never paste raw English into the final answer."
    ),
    category=ToolCategory.UTILITY,
    parameters=[
        ToolParam(
            name="query",
            type="string",
            description=(
                "Concise English search query (DDG works best in English). "
                "Include the current year when asking about current people or "
                "events (e.g. '… 2026'). Quote multi-word names for precision."
            ),
            required=True,
            examples=[
                "UEFA Champions League 2025 winner",
                "current president of India 2026",
                "Mount Everest height meters",
                "OpenAI GPT release 2026",
            ],
        )
    ],
)

def _run_search_sync(query: str) -> list[dict]:
    """Run search synchronously via DDG HTML."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result'):
            title_node = result.find('a', class_='result__url')
            snippet_node = result.find('a', class_='result__snippet')
            if title_node and snippet_node:
                results.append({
                    'title': title_node.text.strip(),
                    'body': snippet_node.text.strip(),
                    'href': title_node.get('href', '')
                })
            if len(results) >= 5:
                break
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

async def handle_search(ctx: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    """Execute search and return results."""
    query = arguments.get("query")
    if not query:
        return ToolResult(tool_id=SEARCH_SPEC.tool_id, success=False, error="Missing query")

    try:
        raw_results = await asyncio.to_thread(_run_search_sync, query)
        
        if not raw_results:
             return ToolResult(
                tool_id=SEARCH_SPEC.tool_id,
                success=True,
                content="No search results were found for this query."
            )

        formatted_results = []
        for r in raw_results:
            formatted_results.append(f"Snippet: {r.get('body')}\nSource: {r.get('href')}")
            
        content = "Internet Search Results:\n\n" + "\n\n".join(formatted_results)
        
        return ToolResult(
            tool_id=SEARCH_SPEC.tool_id,
            success=True,
            content=content
        )
    except Exception as e:
        return ToolResult(tool_id=SEARCH_SPEC.tool_id, success=False, error=str(e))

def register() -> None:
    """Register the search tool with the global registry."""
    get_registry().register(SEARCH_SPEC, handle_search)
