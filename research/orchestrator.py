import re
from dataclasses import dataclass, field

from rag.context_builder import (
    build_context,
    build_general_prompt,
    build_mixed_prompt,
    build_web_prompt,
)
from rag.query_rewriter import rewrite_query_for_retrieval
from llm.llm_service import generate_answer
from web.web_search import search_web

# Evidence critic integration
from research.evidence_critic import criticize_evidence_safely


@dataclass
class ResearchResult:
    answer: str
    pdf_sources: list = field(default_factory=list)
    web_sources: list = field(default_factory=list)
    route: str = "general"
    warnings: list = field(default_factory=list)


DOCUMENT_DEIXIS = [
    "this paper's",
    "this paper",
    "this document",
    "this pdf",
    "the paper",
    "the document",
    "the pdf",
    "the authors",
    "according to the paper",
    "according to the document",
]


def _web_focused_query(question):
    """Strip document-only phrasing so Tavily searches the open web."""
    query = question or ""

    for phrase in DOCUMENT_DEIXIS:
        query = re.sub(
            re.escape(phrase),
            " ",
            query,
            flags=re.IGNORECASE
        )

    query = re.sub(r"\s+", " ", query).strip()

    return query or question


def _build_history_context(history):
    """Format recent conversation history for the LLM."""
    if not history:
        return ""

    lines = ["RECENT CONVERSATION:"]

    for message in history:
        role = message.get("role", "").upper()
        content = message.get("content", "").strip()

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ============================================================================
# Evidence Sufficiency Functions
# ============================================================================

def _check_document_sufficiency(query, pdf_results):
    """Simplified document sufficiency check."""
    if not pdf_results:
        return False, "No PDF evidence retrieved", {"retry_with_more": True}
    return True, "PDF evidence retrieved", {}


def _check_web_sufficiency(query, web_results):
    """Simplified web sufficiency check."""
    if not web_results:
        return False, "No web evidence retrieved", {"targeted_aspects": ["any evidence"]}
    return True, "Web evidence retrieved", {}


def _check_mixed_sufficiency(query, pdf_results, web_results):
    """Mixed sufficiency check focusing on missing sides."""
    query_lower = query.lower()
    
    # Check for explicit comparison terms
    explicit_comparison = any(term in query_lower for term in 
                            ["compare", "vs", "versus", "contrast", "difference between"])
    
    # Check for modern/current terms  
    modern_terms = ["modern", "current", "recent", "latest", "state-of-the-art", "sota"]
    has_modern = any(term in query_lower for term in modern_terms)
    
    has_pdf = bool(pdf_results)
    has_web = bool(web_results)
    
    if not has_pdf and not has_web:
        return False, "No evidence from PDF or web", {"missing_side": "both"}
    
    if explicit_comparison:
        if not has_pdf:
            return False, "Comparison requires PDF evidence", {"missing_side": "pdf"}
        if not has_web:
            return False, "Comparison requires web evidence", {"missing_side": "web"}
    
    if has_modern and not has_web:
        return False, "Query about modern/current topic requires web evidence", {"missing_side": "web", "targeted_aspects": ["modern/current"]}
    
    return True, "Mixed evidence retrieved", {}


def _refine_web_query_for_gap(query, gap_analysis):
    """Create targeted web query refinement based on gap analysis."""
    refined = query
    
    missing_aspects = gap_analysis.get("targeted_aspects", [])
    
    # Add missing aspects if query doesn't already contain them
    if "modern/current" in missing_aspects:
        # Only add recency emphasis if not already present
        if not any(term in query.lower() for term in ["current", "recent", "latest", "modern"]):
            refined = f"{query} current"
    
    # For completely missing evidence, try broader phrasing
    if "any evidence" in missing_aspects and len(query.split()) < 5:
        # If query is short, add a clarifying term
        refined = f"{query} information"
    
    return refined.strip()


def _retrieve_with_sufficiency(query, route, retriever=None, max_iterations=2, history=None):
    """
    Main controller for iterative retrieval with sufficiency checks.
    
    Args:
        query: Question to retrieve for
        route: "document", "web", or "mixed"
        retriever: PDF retriever instance (for document/mixed)
        max_iterations: Maximum retrieval attempts (default: 2)
        history: Conversation history (for query rewriting if needed)
    
    Returns:
        tuple: (results: dict/list, sufficient: bool, warnings: list)
        
        results format depends on route:
          - document: list of PDF chunks
          - web: list of web results  
          - mixed: {"pdf": list, "web": list}
    """
    warnings = []
    iteration = 1
    sufficient = False
    
    # Initialize result storage
    pdf_results = []
    web_results = []
    refined_web_query = None
    
    current_query = query
    
    while iteration <= max_iterations and not sufficient:
        if route == "document":
            # Document route: retrieve PDF only
            top_k = 3 if iteration == 1 else 6  # Double on second attempt
            pdf_results = retriever.search(current_query, top_k=top_k)
            
            sufficient, gap, refinement = _check_document_sufficiency(current_query, pdf_results)
            
            if not sufficient and iteration < max_iterations:
                warnings.append(f"Iteration {iteration}: {gap}")
                # For document, keep same query, just retry with more results
                # top_k is handled in the next iteration
                pass
                
        elif route == "web":
            # Web route: retrieve web only
            web_results = search_web(current_query, max_results=5)
            
            sufficient, gap, refinement = _check_web_sufficiency(current_query, web_results)
            
            if not sufficient and iteration < max_iterations:
                warnings.append(f"Iteration {iteration}: {gap}")
                current_query = _refine_web_query_for_gap(current_query, refinement)
                
        elif route == "mixed":
            # Mixed route: may retrieve PDF, web, or both
            missing_side = None
            
            if iteration == 1:
                # First iteration: retrieve both sides
                pdf_results = retriever.search(current_query, top_k=3) if retriever else []
                web_query = _web_focused_query(current_query)
                web_results = search_web(web_query, max_results=5)
                refined_web_query = web_query
                
                sufficient, gap, refinement = _check_mixed_sufficiency(
                    current_query, pdf_results, web_results
                )
                
                if not sufficient:
                    missing_side = refinement.get("missing_side")
                    warnings.append(f"Iteration {iteration}: {gap}")
            else:
                # Second iteration: retry only missing side
                missing_side = refinement.get("missing_side")
                
                if missing_side == "pdf":
                    # Retry PDF only with more results
                    pdf_results = retriever.search(current_query, top_k=6) if retriever else []
                    # Keep original web_results
                    
                    # Check sufficiency again
                    sufficient, gap, refinement = _check_mixed_sufficiency(
                        current_query, pdf_results, web_results
                    )
                    
                elif missing_side == "web":
                    # Retry web only with refined query
                    if refined_web_query:
                        refined_web_query = _refine_web_query_for_gap(refined_web_query, refinement)
                        web_results = search_web(refined_web_query, max_results=5)
                    else:
                        web_query = _web_focused_query(current_query)
                        refined_web_query = _refine_web_query_for_gap(web_query, refinement)
                        web_results = search_web(refined_web_query, max_results=5)
                    
                    # Keep original pdf_results
                    
                    # Check sufficiency again
                    sufficient, gap, refinement = _check_mixed_sufficiency(
                        current_query, pdf_results, web_results
                    )
                    
                elif missing_side == "both":
                    # Retry both sides
                    pdf_results = retriever.search(current_query, top_k=6) if retriever else []
                    
                    if refined_web_query:
                        refined_web_query = _refine_web_query_for_gap(refined_web_query, refinement)
                    else:
                        web_query = _web_focused_query(current_query)
                        refined_web_query = _refine_web_query_for_gap(web_query, refinement)
                    
                    web_results = search_web(refined_web_query, max_results=5)
                    
                    # Check sufficiency again
                    sufficient, gap, refinement = _check_mixed_sufficiency(
                        current_query, pdf_results, web_results
                    )
        
        iteration += 1
    
    # Add final warning if still insufficient after max iterations
    if not sufficient and warnings:
        warnings.append(f"Evidence may be limited after {max_iterations} retrieval attempts")
    
    # Return results based on route
    if route == "document":
        return pdf_results, sufficient, warnings
    elif route == "web":
        return web_results, sufficient, warnings
    elif route == "mixed":
        return {"pdf": pdf_results, "web": web_results}, sufficient, warnings


# ============================================================================
# Route Execution Functions (Modified)
# ============================================================================

def _generate_insufficient_answer(question, route, missing_info=None):
    """Generate a fallback answer when evidence is insufficient."""
    if route == "document":
        base = "I could not find enough information in the provided document to answer that question."
    elif route == "web":
        base = "I could not find enough relevant information on the web to answer that question."
    elif route == "mixed":
        base = "I could not find enough evidence in the uploaded document or on the web to answer this comparison."
    else:
        base = "I could not find enough information to answer that question."
    
    if missing_info:
        details = f" Specifically, information about {missing_info} was not found."
        return base + details
    return base


def _run_document(question, retriever, warnings, history=None):
    if retriever is None:
        return ResearchResult(
            answer=(
                "Please upload a PDF first so I can "
                "answer questions using its content."
            ),
            route="document",
            warnings=warnings + [
                "No PDF is loaded for a document question."
            ]
        )

    # Apply query rewriting for conversational follow-ups
    retrieval_question = rewrite_query_for_retrieval(
        question, history, "document"
    )
    
    # Use the rewritten question for retrieval, original question for answer
    pdf_sources, basic_sufficient, retrieval_warnings = _retrieve_with_sufficiency(
        retrieval_question, "document", retriever=retriever, history=history
    )
    
    warnings.extend(retrieval_warnings)

    if not pdf_sources:
        return ResearchResult(
            answer=_generate_insufficient_answer(question, "document"),
            pdf_sources=[],
            route="document",
            warnings=warnings + [
                "PDF retrieval returned no passages."
            ]
        )

    # ============================================================================
    # Evidence Critic Integration with Proper Flow
    # ============================================================================
    
    max_attempts = 2
    attempts = 1
    current_evidence = pdf_sources
    current_sufficient = basic_sufficient
    critic_result = None
    
    # Main critic evaluation loop
    while attempts <= max_attempts:
        # Run critic on current evidence
        if current_evidence:
            critic_result = criticize_evidence_safely(
                question=question,
                evidence=current_evidence,
                route="document",
                evidence_type="pdf"
            )
        
        if critic_result is None:
            # Critic failed - use basic assessment
            warnings.append("Evidence critic unavailable, using basic assessment")
            current_sufficient = basic_sufficient if attempts == 1 else current_sufficient
            break
        else:
            current_sufficient = critic_result["sufficient"]
            
            # If critic says sufficient, we're done
            if current_sufficient:
                break
            
            # If insufficient and we have another attempt
            if attempts < max_attempts and critic_result.get("refined_query"):
                attempts += 1
                # Targeted retrieval with refined query
                refined_pdf = retriever.search(critic_result["refined_query"], top_k=6)
                if refined_pdf:
                    current_evidence = refined_pdf
                    warnings.append(f"Targeted retrieval completed (attempt {attempts})")
                    # Continue loop to re-evaluate with new evidence
                    continue
                else:
                    # Retrieval failed, keep existing evidence
                    warnings.append(f"Targeted retrieval attempt {attempts} returned no results")
                    break
            else:
                # No more attempts or no refined query
                break
    
    # Final decision based on critic evaluation
    if not current_sufficient:
        # Evidence is insufficient after all attempts
        missing_info = critic_result.get("missing", []) if critic_result else None
        missing_text = ", ".join(missing_info[:2]) if missing_info else None
        return ResearchResult(
            answer=_generate_insufficient_answer(question, "document", missing_text),
            pdf_sources=current_evidence,
            route="document",
            warnings=warnings + ["Evidence critic marked retrieved information as insufficient"]
        )
    
    # ============================================================================
    # Answer Generation (only if evidence is sufficient)
    # ============================================================================
    
    prompt = build_context(question, current_evidence)

    history_context = _build_history_context(history)

    if history_context:
        prompt += f"""

{history_context}

Use the recent conversation to understand references such as
"it", "they", "this", "that", or "what about".
"""

    answer = generate_answer(prompt)

    return ResearchResult(
        answer=answer,
        pdf_sources=current_evidence,
        route="document",
        warnings=warnings
    )


def _run_general(question, warnings, history=None):
    prompt = build_general_prompt(question)

    history_context = _build_history_context(history)

    if history_context:
        prompt += f"""

{history_context}

Use the recent conversation when the user's question is a follow-up.
"""

    answer = generate_answer(prompt)

    return ResearchResult(
        answer=answer,
        route="general",
        warnings=warnings
    )


def _run_web(question, warnings, history=None):
    # Apply query rewriting for conversational follow-ups
    search_query = rewrite_query_for_retrieval(
        question, history, "web"
    )
    
    try:
        web_sources, basic_sufficient, retrieval_warnings = _retrieve_with_sufficiency(
            search_query, "web", retriever=None, history=history
        )
        
        warnings.extend(retrieval_warnings)

    except ValueError as error:
        return ResearchResult(
            answer=(
                "Web search is unavailable because a Tavily API key "
                "is not configured. You can still ask document or "
                "general questions."
            ),
            route="web",
            warnings=warnings + [str(error)]
        )

    if not web_sources:
        return ResearchResult(
            answer=_generate_insufficient_answer(question, "web"),
            route="web",
            warnings=warnings + [
                "Web search returned no usable results."
            ]
        )

    # ============================================================================
    # Evidence Critic Integration with Proper Flow
    # ============================================================================
    
    max_attempts = 2
    attempts = 1
    current_evidence = web_sources
    current_sufficient = basic_sufficient
    critic_result = None
    
    # Main critic evaluation loop
    while attempts <= max_attempts:
        # Run critic on current evidence
        if current_evidence:
            critic_result = criticize_evidence_safely(
                question=question,
                evidence=current_evidence,
                route="web",
                evidence_type="web"
            )
        
        if critic_result is None:
            # Critic failed - use basic assessment
            warnings.append("Evidence critic unavailable, using basic assessment")
            current_sufficient = basic_sufficient if attempts == 1 else current_sufficient
            break
        else:
            current_sufficient = critic_result["sufficient"]
            
            # If critic says sufficient, we're done
            if current_sufficient:
                break
            
            # If insufficient and we have another attempt
            if attempts < max_attempts and critic_result.get("refined_query"):
                attempts += 1
                # Targeted retrieval with refined query
                refined_web = search_web(critic_result["refined_query"], max_results=5)
                if refined_web:
                    current_evidence = refined_web
                    warnings.append(f"Targeted web search completed (attempt {attempts})")
                    # Continue loop to re-evaluate with new evidence
                    continue
                else:
                    # Retrieval failed, keep existing evidence
                    warnings.append(f"Targeted web search attempt {attempts} returned no results")
                    break
            else:
                # No more attempts or no refined query
                break
    
    # Final decision based on critic evaluation
    if not current_sufficient:
        # Evidence is insufficient after all attempts
        missing_info = critic_result.get("missing", []) if critic_result else None
        missing_text = ", ".join(missing_info[:2]) if missing_info else None
        return ResearchResult(
            answer=_generate_insufficient_answer(question, "web", missing_text),
            web_sources=current_evidence,
            route="web",
            warnings=warnings + ["Evidence critic marked retrieved information as insufficient"]
        )
    
    # ============================================================================
    # Answer Generation (only if evidence is sufficient)
    # ============================================================================
    
    prompt = build_web_prompt(question, current_evidence)

    history_context = _build_history_context(history)

    if history_context:
        prompt += f"""

{history_context}

Use the recent conversation to understand follow-up questions.
"""

    answer = generate_answer(prompt)

    return ResearchResult(
        answer=answer,
        web_sources=current_evidence,
        route="web",
        warnings=warnings
    )


def _run_mixed(question, retriever, warnings, history=None):
    if retriever is None:
        warnings = warnings + [
            "No PDF is loaded; mixed research cannot use document evidence."
        ]

        web_result = _run_web(
            question,
            warnings,
            history
        )

        note = (
            "I do not have an uploaded PDF to compare against. "
            "The answer below uses web evidence only.\n\n"
        )

        return ResearchResult(
            answer=note + web_result.answer,
            web_sources=web_result.web_sources,
            route="mixed",
            warnings=web_result.warnings
        )

    # Apply query rewriting for conversational follow-ups
    retrieval_question = rewrite_query_for_retrieval(
        question, history, "mixed"
    )
    
    try:
        results, basic_sufficient, retrieval_warnings = _retrieve_with_sufficiency(
            retrieval_question, "mixed", retriever=retriever, history=history
        )
        
        warnings.extend(retrieval_warnings)
        
        pdf_sources = results.get("pdf", [])
        web_sources = results.get("web", [])

    except ValueError as error:
        pdf_sources = []
        web_sources = []
        warnings = warnings + [str(error)]

    if not pdf_sources:
        warnings = warnings + [
            "PDF retrieval returned no passages."
        ]

    if not web_sources:
        warnings = warnings + [
            "Web search returned no usable results."
        ]

    if not pdf_sources and not web_sources:
        return ResearchResult(
            answer=_generate_insufficient_answer(question, "mixed"),
            route="mixed",
            warnings=warnings
        )

    # ============================================================================
    # Evidence Critic Integration (Mixed Route - Proper Flow)
    # ============================================================================
    
    max_attempts = 2
    attempts = 1
    current_pdf = pdf_sources
    current_web = web_sources
    
    # Track sufficiency for each side
    pdf_sufficient = bool(current_pdf)  # Start with basic assessment
    web_sufficient = bool(current_web)
    combined_sufficient = True  # Optimistic default
    
    # Store critic results for potential retry
    pdf_critic_result = None
    web_critic_result = None
    combined_critic_result = None
    
    # Main evaluation loop
    while attempts <= max_attempts:
        # Run critics on current evidence
        if current_pdf:
            pdf_critic_result = criticize_evidence_safely(
                question=question,
                evidence=current_pdf,
                route="mixed",
                evidence_type="pdf"
            )
            if pdf_critic_result is not None:
                pdf_sufficient = pdf_critic_result["sufficient"]
        
        if current_web:
            web_critic_result = criticize_evidence_safely(
                question=question,
                evidence=current_web,
                route="mixed",
                evidence_type="web"
            )
            if web_critic_result is not None:
                web_sufficient = web_critic_result["sufficient"]
        
        # Evaluate combined evidence if we have both sides
        if current_pdf or current_web:
            combined_evidence = {"pdf": current_pdf, "web": current_web}
            combined_critic_result = criticize_evidence_safely(
                question=question,
                evidence=combined_evidence,
                route="mixed",
                evidence_type="pdf"
            )
            if combined_critic_result is not None:
                combined_sufficient = combined_critic_result["sufficient"]
        
        # Determine overall sufficiency
        query_lower = question.lower()
        explicit_comparison = any(term in query_lower for term in 
                                ["compare", "vs", "versus", "contrast", "difference between"])
        
        if explicit_comparison:
            # For comparison questions, need both sides if available
            effective_sufficient = (
                (pdf_sufficient or not current_pdf) and  # PDF sufficient if we have it
                (web_sufficient or not current_web) and  # Web sufficient if we have it
                combined_sufficient
            )
        else:
            # For non-comparison, sufficient if at least one side is sufficient
            effective_sufficient = (
                (pdf_sufficient if current_pdf else True) and
                (web_sufficient if current_web else True) and
                combined_sufficient
            )
        
        # If sufficient or no more attempts, we're done
        if effective_sufficient or attempts >= max_attempts:
            break
        
        # Determine which sides need retry
        attempts += 1
        need_pdf_retry = current_pdf and not pdf_sufficient and pdf_critic_result and pdf_critic_result.get("refined_query")
        need_web_retry = current_web and not web_sufficient and web_critic_result and web_critic_result.get("refined_query")
        
        # Perform targeted retrievals
        if need_pdf_retry:
            refined_pdf = retriever.search(pdf_critic_result["refined_query"], top_k=6)
            if refined_pdf:
                current_pdf = refined_pdf
                warnings.append(f"Targeted PDF retrieval completed (attempt {attempts})")
        
        if need_web_retry:
            refined_web = search_web(web_critic_result["refined_query"], max_results=5)
            if refined_web:
                current_web = refined_web
                warnings.append(f"Targeted web search completed (attempt {attempts})")
        
        # If no retries were performed, break
        if not need_pdf_retry and not need_web_retry:
            break
    
    # Final sufficiency check after all attempts
    query_lower = question.lower()
    explicit_comparison = any(term in query_lower for term in 
                            ["compare", "vs", "versus", "contrast", "difference between"])
    
    if explicit_comparison:
        final_sufficient = (
            (pdf_sufficient or not current_pdf) and
            (web_sufficient or not current_web) and
            combined_sufficient
        )
    else:
        final_sufficient = (
            (pdf_sufficient if current_pdf else True) and
            (web_sufficient if current_web else True) and
            combined_sufficient
        )
    
    # If evidence is insufficient, return fallback answer
    if not final_sufficient:
        missing_info = []
        if pdf_critic_result and not pdf_sufficient:
            missing_info.extend(pdf_critic_result.get("missing", []))
        if web_critic_result and not web_sufficient:
            missing_info.extend(web_critic_result.get("missing", []))
        if combined_critic_result and not combined_sufficient:
            missing_info.extend(combined_critic_result.get("missing", []))
        
        missing_text = ", ".join(set(missing_info[:3])) if missing_info else None
        return ResearchResult(
            answer=_generate_insufficient_answer(question, "mixed", missing_text),
            pdf_sources=current_pdf,
            web_sources=current_web,
            route="mixed",
            warnings=warnings + ["Evidence critic marked comparison evidence as insufficient"]
        )
    
    # ============================================================================
    # Answer Generation (only if evidence is sufficient)
    # ============================================================================
    
    prompt = build_mixed_prompt(
        question,
        current_pdf,
        current_web,
        warnings
    )

    history_context = _build_history_context(history)

    if history_context:
        prompt += f"""

{history_context}

Use the recent conversation to understand follow-up references
and maintain continuity between the document and web discussion.
"""

    answer = generate_answer(prompt)

    return ResearchResult(
        answer=answer,
        pdf_sources=current_pdf,
        web_sources=current_web,
        route="mixed",
        warnings=warnings
    )


def run(route, question, retriever=None, history=None):
    """
    Execute a routed research question.

    Recent conversation history is passed to the appropriate
    answering pipeline to support conversational follow-ups.
    """

    warnings = []

    if route == "document":
        return _run_document(
            question,
            retriever,
            warnings,
            history
        )

    if route == "general":
        return _run_general(
            question,
            warnings,
            history
        )

    if route == "web":
        return _run_web(
            question,
            warnings,
            history
        )

    if route == "mixed":
        return _run_mixed(
            question,
            retriever,
            warnings,
            history
        )

    return _run_general(
        question,
        warnings + [
            f"Unknown route '{route}'; used general answering."
        ],
        history
    )