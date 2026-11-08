from router import classify_query


def check(question, expected, has_document=True, recent_messages=None):

    route = classify_query(
        question,
        has_document=has_document,
        recent_messages=recent_messages
    )

    print(f"\nQuestion: {question}")
    print(f"Expected: {expected}")
    print(f"Route: {route}")

    if route != expected:
        raise AssertionError(
            f"Expected {expected!r} for {question!r}, got {route!r}"
        )


document_history = [
    {
        "role": "user",
        "content": "What architecture does this paper propose?",
        "route": "document"
    },
    {
        "role": "assistant",
        "content": "The paper proposes a transformer variant."
    }
]


tests = [
    (
        "What architecture does this paper propose?",
        "document",
        True,
        None
    ),
    (
        "What dataset did they use?",
        "document",
        True,
        None
    ),
    (
        "Summarize the methodology",
        "document",
        True,
        None
    ),
    (
        "What is RAG?",
        "general",
        True,
        None
    ),
    (
        "What is RAG?",
        "general",
        False,
        None
    ),
    (
        "What is the latest AI research?",
        "web",
        True,
        None
    ),
    (
        "What is the current Python version?",
        "web",
        True,
        None
    ),
    (
        "Compare this paper with the latest Transformer research.",
        "mixed",
        True,
        None
    ),
    (
        "How does this paper's approach compare with the latest Transformer research?",
        "mixed",
        True,
        None
    ),
    (
        "What about the limitations?",
        "document",
        True,
        document_history
    ),
    (
        "What architecture does this paper propose?",
        "general",
        False,
        None
    ),
]


if __name__ == "__main__":

    for question, expected, has_document, history in tests:

        check(
            question,
            expected,
            has_document=has_document,
            recent_messages=history
        )

    print("\nAll router tests passed.")
