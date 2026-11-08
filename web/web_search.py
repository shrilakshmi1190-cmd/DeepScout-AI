import os

from tavily import TavilyClient
from dotenv import load_dotenv


_client = None


def _get_client():
    """Create the Tavily client only when a web search is actually needed."""

    global _client

    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set in the .env file."
        )

    if _client is None:
        _client = TavilyClient(api_key=api_key)

    return _client
    
def _choose_include_domains(query_lower):
    """
    Hint Tavily toward authoritative sources for technical queries.
    Avoid locking on weak or overly broad sources.
    """

    if (
        "python" in query_lower
        and (
            "latest version" in query_lower
            or "current version" in query_lower
            or "latest release" in query_lower
            or "current release" in query_lower
            or "newest version" in query_lower
            or "newest release" in query_lower
        )
    ):
        return [
            "python.org",
            "docs.python.org"
        ]

    if "python" in query_lower:
        return [
            "python.org",
            "docs.python.org"
        ]

    if "openai" in query_lower:
        return [
            "openai.com",
            "platform.openai.com"
        ]

    if "google" in query_lower or "gemini" in query_lower:
        return [
            "google.com",
            "ai.google"
        ]

    if "machine learning" in query_lower:
        return [
            "arxiv.org",
            "research.google",
            "ai.google"
        ]

    if "research paper" in query_lower:
        return [
            "arxiv.org"
        ]

    return None


def _process_results(response, max_results):

    results = []

    for result in response.get("results", []):

        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        score = result.get("score", 0)

        if score < 0.5:
            continue

        if not str(content).strip():
            continue

        results.append({
            "title": title,
            "url": url,
            "content": content,
            "score": float(score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:max_results]


def search_web(query, max_results=5):

    """
    Search the web using Tavily.
    Prefer authoritative sources for technical queries.
    If a domain filter yields nothing, retry without it.
    """

    client = _get_client()

    query_lower = query.lower()
    include_domains = _choose_include_domains(query_lower)

    search_query = query

    if (
        "python" in query_lower
        and (
            "latest version" in query_lower
            or "current version" in query_lower
            or "latest release" in query_lower
            or "current release" in query_lower
            or "newest version" in query_lower
            or "newest release" in query_lower
        )
    ):
        search_query = (
            f"{query} latest stable release "
            "official Python.org"
        )

    search_kwargs = {
        "query": search_query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False
    }

    if include_domains:
        search_kwargs["include_domains"] = include_domains

    response = client.search(**search_kwargs)
    results = _process_results(response, max_results)

    if not results and include_domains:

        fallback_kwargs = {
            "query": search_query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False
        }

        response = client.search(**fallback_kwargs)
        results = _process_results(response, max_results)

    return results
