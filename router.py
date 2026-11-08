import re


LIVE_PATTERNS = [
    r"\btoday\b",
    r"\btoday's\b",
    r"\blatest\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\btrending\b",
    r"\bweather\b",
    r"\bstock prices?\b",
    r"\bcurrent version\b",
    r"\blatest version\b",
    r"\bcurrent python\b",
    r"\brecent research\b",
    r"\blatest research\b",
    r"\bcurrent news\b",
    r"\bin the news\b",
    r"\bwhat is the current\b",
    r"\bwhat's the current\b",
    r"\bcurrent research\b",
]

EXTERNAL_COMPARISON_PATTERNS = [
    r"\bmodern\b",
    r"\bcontemporary\b",
    r"\bnewer\b",
    r"\bstate of the art\b",
    r"\bsota\b",
    r"\brecent\b",
    r"\bup-to-date\b",
    r"\bup to date\b",
    r"\bpresent-day\b",
    r"\bpresent day\b",
    r"\bexternal\b",
    r"\boutside the paper\b",
    r"\bbeyond this paper\b",
    r"\boutside this paper\b",
    r"\bbeyond the paper\b",
]

DEIXIS_PATTERNS = [
    r"\bthis paper\b",
    r"\bthis document\b",
    r"\bthis pdf\b",
    r"\bthe paper\b",
    r"\bthe document\b",
    r"\bthe pdf\b",
    r"\baccording to the paper\b",
    r"\baccording to the document\b",
    r"\bthe authors\b",
    r"\bwhat does the paper\b",
    r"\bwhat do the authors\b",
]

COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bcompared\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bvs\.\b",
    r"\bdiffer\b",
    r"\bdifference\b",
    r"\bcontrast\b",
    r"\bstate of the art\b",
    r"\bsota\b",
]

EXPLICIT_GENERAL_PHRASES = [
    "in general",
    "generally speaking",
    "not in the paper",
    "not in the document",
    "not in the pdf",
    "outside the paper",
    "outside the document",
    "outside the pdf",
]

DOCUMENT_TOPIC_PATTERNS = [
    r"\bthe paper\b",
    r"\bthe document\b",
    r"\bthe pdf\b",
    r"\bthe authors\b",
    r"\bthe study\b",
    r"\bthe dataset\b",
    r"\bthe methodology\b",
    r"\bthe method\b",
    r"\bthe approach\b",
    r"\bthe results?\b",
    r"\bthe limitations?\b",
    r"\bthe architecture\b",
    r"\bthe experiment\b",
    r"\bthe evaluation\b",
    r"\bthe abstract\b",
    r"\bthe conclusion\b",
    r"\bthe model\b",
]

FOLLOW_UP_PATTERNS = [
    r"^what about\b",
    r"^how about\b",
    r"^and the\b",
    r"^and what\b",
    r"^also\b",
    r"^what else\b",
    r"^why\??$",
    r"^and\??$",
]


def _matches_any(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def _has_live_web_intent(question_lower):
    return _matches_any(question_lower, LIVE_PATTERNS)


def _has_external_comparison_signal(question_lower):
    return _matches_any(question_lower, EXTERNAL_COMPARISON_PATTERNS)


def _has_document_deixis(question_lower):
    return _matches_any(question_lower, DEIXIS_PATTERNS)


def _has_comparison_intent(question_lower):
    if _matches_any(question_lower, COMPARISON_PATTERNS):
        return True

    # "this paper ... with the latest ..."
    if (
        _has_document_deixis(question_lower)
        and _has_live_web_intent(question_lower)
        and re.search(r"\b(with|against|vs|versus)\b", question_lower)
    ):
        return True

    return False


def _is_explicit_general(question_lower):
    return any(
        phrase in question_lower
        for phrase in EXPLICIT_GENERAL_PHRASES
    )


def _is_definitional_general(question_lower, has_document):
    """
    Detect genuinely general/definition questions.

    When a PDF is loaded, broad questions such as
    "What are the major components of the Transformer architecture?"
    should go to the document unless the user is clearly asking
    about a general concept.
    """

    question = question_lower.strip()

    # Explicit references to the document are never general.
    if _has_document_deixis(question):
        return False

    # Questions referring to document-specific topics should use RAG.
    if _matches_any(question, DOCUMENT_TOPIC_PATTERNS):
        return False

    # Pronouns usually refer to the uploaded document when one exists.
    if has_document and re.search(r"\b(they|their|them)\b", question):
        return False

    # Explicit general definitions.
    if _is_explicit_general(question):
        return True

    # Clear acronym/term-definition questions.
    if re.match(
        r"^(what does \w+ (mean|stand for)|"
        r"what is an acronym for|"
        r"define)\b",
        question
    ):
        return True

    # Common short conceptual questions should remain general
    # even when a PDF is loaded.
    GENERAL_CONCEPTS = [
        "rag",
        "llm",
        "nlp",
        "api",
        "ml",
        "ai",
        "cnn",
        "rnn",
        "gpu",
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "natural language processing",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    ]

    if has_document:
        for concept in GENERAL_CONCEPTS:
            if re.match(
                rf"^(what is|what are|what's|whats|explain|define)\s+{re.escape(concept)}\b",
                question
            ):
                return True

        # With a PDF loaded, generic "what is/are X?" questions
        # default to the document unless X is clearly a general concept.
        return False

    # No PDF is loaded: normal definition questions are general.
    return bool(re.match(
        r"^(what is|what are|what's|whats|explain|define)\b",
        question
    ))
     


def _is_follow_up(question_lower):
    return _matches_any(question_lower.strip(), FOLLOW_UP_PATTERNS)


def _last_user_route(recent_messages):
    if not recent_messages:
        return None

    for message in reversed(recent_messages):
        if (
            message.get("role") == "user"
            and message.get("route")
        ):
            return message["route"]

    return None


def classify_query(
    question,
    has_document=False,
    recent_messages=None
):
    """
    Route a question to general, document, web, or mixed.

    When a PDF is loaded, document is the default unless the question
    is clearly general, live-web, or mixed.
    """

    question_lower = (question or "").lower()

    has_live_web = _has_live_web_intent(question_lower)
    has_deixis = _has_document_deixis(question_lower)
    has_comparison = _has_comparison_intent(question_lower)
    last_route = _last_user_route(recent_messages)

    needs_mixed = (
        has_document
        and has_comparison
        and (has_live_web or _has_external_comparison_signal(question_lower))
    )

    if needs_mixed:
        return "mixed"

    # Follow-ups inherit the previous user route when they are not
    # clearly a new general/web/mixed question.
    if (
        has_document
        and _is_follow_up(question_lower)
        and last_route in ("document", "mixed")
        and not has_live_web
        and not _is_explicit_general(question_lower)
        and not _is_definitional_general(question_lower, has_document)
    ):
        return "document"

    if (
        _is_follow_up(question_lower)
        and last_route == "general"
        and not has_live_web
        and not has_deixis
    ):
        return "general"

    if (
        _is_follow_up(question_lower)
        and last_route == "web"
        and not has_deixis
        and not has_document
    ):
        return "web"

    if has_live_web and not has_deixis:
        return "web"

    if (
        _is_explicit_general(question_lower)
        or _is_definitional_general(question_lower, has_document)
    ):
        return "general"

    if has_document:
        return "document"

    return "general"
