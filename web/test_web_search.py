from web.web_search import search_web


query = "latest AI research"


results = search_web(
    query,
    max_results=3
)


for i, result in enumerate(results):

    print(f"\nResult {i + 1}")

    print(
        "Title:",
        result["title"]
    )

    print(
        "URL:",
        result["url"]
    )

    print(
        "Score:",
        result["score"]
    )

    print(
        "Content:",
        result["content"][:500]
    )