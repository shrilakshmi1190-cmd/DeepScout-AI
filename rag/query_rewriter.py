"""
Query rewriter for conversational follow-up questions in RAG pipeline.
Rewrites ambiguous follow-up questions into self-contained queries for better retrieval/search.
"""

import re
from typing import List, Dict, Optional

# Import from router for follow-up detection
from router import _is_follow_up


def _build_rewriting_history(history: Optional[List[Dict]]) -> str:
    """
    Format recent conversation history for query rewriting.
    
    Only includes the last 4 messages to keep context manageable.
    """
    if not history or len(history) < 2:
        return ""
    
    # Take last 4 messages (or fewer if history is shorter)
    recent_messages = history[-4:]
    
    lines = ["CONVERSATION HISTORY FOR CONTEXT:"]
    
    for message in recent_messages:
        role = message.get("role", "").upper()
        content = message.get("content", "").strip()
        route = message.get("route", "")
        
        if content:
            route_marker = f" [{route}]" if route else ""
            lines.append(f"{role}{route_marker}: {content}")
    
    return "\n".join(lines)


def _create_rewrite_prompt(
    question: str, 
    history_context: str, 
    route: str
) -> str:
    """
    Create a route-specific prompt for query rewriting.
    
    Args:
        question: Current user question
        history_context: Formatted conversation history
        route: Current route ("document", "web", "mixed", "general")
    
    Returns:
        Prompt for the rewriting LLM
    """
    base_prompt = f"""Rewrite this follow-up question to be a self-contained query for {route} retrieval/search.

CONVERSATION HISTORY:
{history_context}

FOLLOW-UP QUESTION:
{question}

REWRITE RULES:
"""
    
    if route == "document":
        return base_prompt + """
- Resolve all pronouns (it, they, this, that, these, those, its, their, etc.) to explicit document references
- Keep the question focused on the content of the uploaded PDF/document
- Do not add web or external information
- Maintain technical accuracy from the conversation history
- Never invent facts not supported by the conversation
- Output only the rewritten question, no explanations"""
    
    elif route == "web":
        return base_prompt + """
- Make the question suitable for web search (clear, concise, keyword-rich)
- Remove any document-specific references
- Add current date/time context if relevant to the question (e.g., "current status", "latest version")
- Use common search terms
- Never invent facts not supported by the conversation
- Output only the rewritten query, no explanations"""
    
    elif route == "mixed":
        return base_prompt + """
- Rewrite to be self-contained for both document retrieval and web search
- Resolve pronouns to explicit references from the conversation
- Maintain the comparative nature of the question
- Keep both document and web aspects clear
- Never invent facts not supported by the conversation
- Output only the rewritten question, no explanations"""
    
    else:  # general route - though we shouldn't be calling rewrite for general
        return base_prompt + """
- Resolve pronouns to explicit references from the conversation
- Keep the question clear and self-contained
- Never invent facts not supported by the conversation
- Output only the rewritten question, no explanations"""


def _is_valid_rewrite(rewritten: str, original: str) -> bool:
    """
    Validate that a rewritten query is reasonable.
    
    Returns False if:
    - Empty or only whitespace
    - Too short (less than 3 characters)
    - Contains placeholder markers like "[...]" indicating unresolved references
    - Contains refusal phrases like "I cannot", "I don't know"
    """
    if not rewritten or not rewritten.strip():
        return False
    
    rewritten = rewritten.strip()
    
    if len(rewritten) < 3:
        return False
    
    # Check for unresolved placeholders
    if re.search(r'\[.*?\]', rewritten):
        return False
    
    # Check for refusal phrases
    refusal_patterns = [
        r"i (?:cannot|can't|could not)",
        r"i don[']?t know",
        r"i am unable",
        r"not enough information",
        r"no context",
        r"ambiguous",
        r"cannot determine",
    ]
    
    lower_rewritten = rewritten.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, lower_rewritten):
            return False
    
    return True


def rewrite_query_for_retrieval(
    question: str,
    history: Optional[List[Dict]] = None,
    route: str = "document"
) -> str:
    """
    Rewrite conversational follow-up questions for better retrieval/search.
    
    Args:
        question: Current user question
        history: List of recent conversation messages (each with "role", "content", optional "route")
        route: Current route classification ("document", "web", "mixed", "general")
    
    Returns:
        Rewritten question if it's a follow-up and rewriting succeeds,
        original question otherwise.
    
    Important: For "general" route, returns original question unchanged
    as specified in requirements.
    """
    # For general route, don't rewrite (per requirements)
    if route == "general":
        return question
    
    # Check if this is a follow-up question
    if not history or len(history) < 2:
        return question
    
    question_lower = (question or "").lower().strip()
    if not _is_follow_up(question_lower):
        return question
    
    try:
        # Build history context
        history_context = _build_rewriting_history(history)
        if not history_context:
            return question
        
        # Create route-specific prompt
        rewrite_prompt = _create_rewrite_prompt(question, history_context, route)
        
        # Import here to avoid circular imports
        from llm.llm_service import generate_answer
        
        # Call LLM for rewriting
        rewritten = generate_answer(rewrite_prompt)
        
        # Validate the rewrite
        if not _is_valid_rewrite(rewritten, question):
            return question
        
        return rewritten.strip()
        
    except Exception as e:
        # Log the error but don't crash (in production, you'd want logging here)
        # print(f"Query rewriting failed: {e}")
        return question