"""
Evidence Critic for DeepScout AI.

Analyzes whether retrieved evidence actually answers the user's question.
Returns structured assessment with missing information and refined queries.
"""
import json
import re
from typing import Dict, List, Any, Optional, Union
from llm.llm_service import generate_answer


def criticize_evidence(
    question: str, 
    evidence: Union[List[Dict], Dict], 
    route: str, 
    evidence_type: str
) -> Optional[Dict[str, Any]]:
    """
    Analyze whether evidence answers the question.
    
    Args:
        question: User's original question
        evidence: Retrieved evidence (PDF chunks or web results)
        route: "document", "web", or "mixed"
        evidence_type: "pdf" or "web"
    
    Returns:
        Dict with keys:
            - sufficient: bool
            - reason: str (optional, explains why insufficient)
            - missing: List[str] (what information is missing)
            - refined_query: str (better query for missing info)
        Returns None if critic fails.
    """
    # Format evidence for the prompt
    formatted_evidence = _format_evidence_for_critic(evidence, evidence_type)
    
    # Create critic prompt
    prompt = _create_critic_prompt(question, formatted_evidence, route, evidence_type)
    
    try:
        # Call LLM for evidence assessment
        llm_response = generate_answer(prompt)
        
        # Parse and validate response
        critic_result = _parse_critic_response(llm_response)
        
        return critic_result
        
    except Exception as e:
        # LLM call failed - return None to indicate critic failure
        return None


def criticize_evidence_safely(
    question: str, 
    evidence: Union[List[Dict], Dict], 
    route: str, 
    evidence_type: str
) -> Optional[Dict[str, Any]]:
    """
    Safe wrapper for evidence critic with explicit error handling.
    
    Returns None if critic fails for any expected reason.
    """
    try:
        result = criticize_evidence(question, evidence, route, evidence_type)
        
        # Validate the result structure
        if result is None:
            return None
            
        if not isinstance(result, dict):
            return None
            
        if "sufficient" not in result:
            return None
            
        # Ensure sufficient is boolean
        if not isinstance(result["sufficient"], bool):
            return None
            
        # Validate optional fields if present
        if "missing" in result and not isinstance(result["missing"], list):
            result["missing"] = []
            
        if "refined_query" in result and not isinstance(result["refined_query"], str):
            result["refined_query"] = None
            
        if "reason" in result and not isinstance(result["reason"], str):
            result["reason"] = ""
            
        return result
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Expected failures: JSON parsing, missing keys, invalid values
        return None
        
    except Exception as e:
        # Unexpected errors - fail gracefully
        return None


def _format_evidence_for_critic(evidence: Union[List[Dict], Dict], evidence_type: str) -> str:
    """Format evidence for the critic prompt."""
    if isinstance(evidence, dict) and "pdf" in evidence and "web" in evidence:
        # Mixed evidence - format both sides
        pdf_text = _format_pdf_evidence(evidence.get("pdf", []))
        web_text = _format_web_evidence(evidence.get("web", []))
        return f"PDF Evidence:\n{pdf_text}\n\nWeb Evidence:\n{web_text}"
    
    if evidence_type == "pdf":
        return _format_pdf_evidence(evidence)
    else:  # web
        return _format_web_evidence(evidence)


def _format_pdf_evidence(pdf_results: List[Dict]) -> str:
    """Format PDF chunks for critic."""
    if not pdf_results:
        return "(No PDF evidence)"
    
    formatted = []
    for i, result in enumerate(pdf_results):
        chunk = result.get("chunk", "")
        page = result.get("page", "Unknown")
        source = result.get("source", "Unknown")
        
        formatted.append(f"[PDF Source {i+1}, Page {page}, Source: {source}]\n{chunk}")
    
    return "\n\n".join(formatted)


def _format_web_evidence(web_results: List[Dict]) -> str:
    """Format web results for critic."""
    if not web_results:
        return "(No web evidence)"
    
    formatted = []
    for i, result in enumerate(web_results):
        title = result.get("title", "Untitled")
        content = result.get("content", "")
        url = result.get("url", "")
        
        formatted.append(f"[Web Source {i+1}, Title: {title}, URL: {url}]\n{content}")
    
    return "\n\n".join(formatted)


def _create_critic_prompt(question: str, formatted_evidence: str, route: str, evidence_type: str) -> str:
    """Create prompt for evidence critic LLM."""
    
    evidence_type_label = "PDF document" if evidence_type == "pdf" else "web"
    
    # Detect if this is a comparison question
    question_lower = question.lower()
    is_comparison = any(term in question_lower for term in 
                       ["compare", "vs", "versus", "contrast", "difference between", "how does", "relation"])
    
    # Special guidance based on route and evidence type
    route_specific_guidance = ""
    
    if route == "mixed" and is_comparison:
        # For mixed route comparison questions
        if evidence_type == "pdf":
            route_specific_guidance = """
MIXED-ROUTE COMPARISON (PDF EVIDENCE):
- You are evaluating PDF evidence for a MIXED comparison question.
- Your task is ONLY to determine if the PDF evidence adequately describes the CONTENT OF THE UPLOADED PAPER.
- DO NOT require the PDF to discuss modern/current topics - that's the web evidence's responsibility.
- The PDF should provide sufficient information about what the paper proposes/describes.
- If the PDF describes the paper's architecture, methods, or contributions, mark as sufficient.
- Only mark insufficient if the PDF lacks basic information about the paper's content.
- Missing "modern architecture innovations" is NOT a reason to mark PDF insufficient.
"""
        elif evidence_type == "web":
            route_specific_guidance = """
MIXED-ROUTE COMPARISON (WEB EVIDENCE):
- You are evaluating web evidence for a MIXED comparison question.
- Your task is ONLY to determine if the web evidence adequately describes MODERN/CURRENT EXTERNAL INFORMATION.
- DO NOT require web evidence to compare against the uploaded paper - that's the synthesis step.
- The web evidence should provide sufficient information about modern developments to support comparison.
- If web evidence discusses modern Transformer architectures/variants/improvements, mark as sufficient.
- Do NOT require specific technical innovations (RoPE, RMSNorm, GQA) - general discussion of modern developments is sufficient.
- Only mark insufficient if web evidence lacks meaningful information about modern developments.
"""
    
    # General comparison guidance for non-mixed routes
    elif is_comparison:
        if "modern" in question_lower or "current" in question_lower or "recent" in question_lower:
            route_specific_guidance = """
COMPARISON QUESTION GUIDELINES:
- This question asks about MODERN/CURRENT vs PAPER/CLASSIC.
- For PDF evidence: The original 2017 paper cannot be considered evidence about modern architectures unless it explicitly discusses later developments.
- For web evidence: Generic titles or broad discussions are insufficient. Need concrete, specific information about modern variants.
- Both sides must be adequately represented to support a meaningful comparison.
- If evidence only discusses one side, mark as insufficient.
"""
        else:
            route_specific_guidance = """
COMPARISON QUESTION GUIDELINES:
- Verify evidence covers BOTH sides being compared.
- Generic mentions of topics are insufficient for detailed comparison.
- Both sides must have concrete, specific information to support comparison.
"""
    
    # Special guidance for Transformer architecture questions
    transformer_guidance = ""
    if "transformer" in question_lower and "architecture" in question_lower:
        transformer_guidance = """
TRANSFORMER ARCHITECTURE SPECIFICS:
- Modern Transformer architectures include specific innovations: RoPE (Rotary Position Embeddings), RMSNorm, GQA/MQA (Grouped/Multi-Query Attention), pre-LN (pre-layer normalization), FlashAttention, etc.
- The original 2017 paper does NOT contain these modern concepts unless explicitly mentioned.
- Web evidence should discuss concrete modern Transformer variants for meaningful comparison.
"""
    
    return f"""
You are an Evidence Critic for a research assistant. Your task is to determine whether the retrieved evidence actually answers the user's question.

CRITICAL RULES:
1. DO NOT generate or invent the factual answer.
2. Only evaluate whether the QUESTION can be answered from the PROVIDED EVIDENCE.
3. DO NOT use your general knowledge to fill gaps. Judge ONLY what's in the evidence.
4. If evidence is insufficient, identify EXACTLY what specific information is missing.
5. Create a better, targeted query to retrieve the missing information.
6. Return valid JSON only.

QUESTION:
{question}

RETRIEVED {evidence_type_label.upper()} EVIDENCE:
{formatted_evidence}

ANALYSIS GUIDELINES:
- Consider whether ALL aspects of the question are addressed by the evidence.
- Check for semantic mismatches (e.g., English-to-French vs English-to-German).
- Verify numerical values, dates, names, and technical terms match the question.
- If evidence contains RELATED but not SPECIFIC information, mark as insufficient.
- Be strict: evidence must directly support answering the question, not just be topically related.
{route_specific_guidance}{transformer_guidance}
OUTPUT FORMAT (JSON only):
{{
  "sufficient": true/false,
  "reason": "brief explanation (required if insufficient)",
  "missing": ["list of missing information"],
  "refined_query": "better query for retrieving missing info"
}}

EXAMPLES:
1. Question: "What BLEU score on WMT 2014 English-to-German?"
   Evidence: Contains "41.0 BLEU on English-to-French"
   Output: {{"sufficient": false, "reason": "Evidence shows English-to-French, not English-to-German", "missing": ["English-to-German BLEU score"], "refined_query": "Transformer WMT 2014 English German BLEU score"}}

2. Question: "What architecture does this paper propose?"
   Evidence: Contains "The paper proposes a Transformer architecture"
   Output: {{"sufficient": true, "reason": "", "missing": [], "refined_query": ""}}

3. Question: "Compare this paper with modern architectures."
   Evidence: Only discusses original paper, no modern concepts
   Output: {{"sufficient": false, "reason": "Evidence only covers original paper, missing modern architecture details", "missing": ["Modern architecture innovations"], "refined_query": "modern Transformer architecture innovations RoPE RMSNorm GQA"}}

Now analyze the evidence above and return JSON:
"""


def _parse_critic_response(llm_response: str) -> Optional[Dict[str, Any]]:
    """Parse and validate LLM critic response."""
    try:
        # Try to extract JSON from response (LLM might wrap it in text)
        response_text = llm_response.strip()
        
        # Look for JSON object
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate required structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
                
            if "sufficient" not in result:
                raise KeyError("Missing 'sufficient' field")
                
            # Ensure sufficient is boolean (allow string "true"/"false")
            sufficient = result["sufficient"]
            if isinstance(sufficient, str):
                if sufficient.lower() == "true":
                    result["sufficient"] = True
                elif sufficient.lower() == "false":
                    result["sufficient"] = False
                else:
                    raise ValueError(f"Invalid boolean string: {sufficient}")
            elif not isinstance(sufficient, bool):
                raise ValueError(f"Invalid boolean type: {type(sufficient)}")
            
            # Set defaults for optional fields
            result.setdefault("reason", "")
            result.setdefault("missing", [])
            result.setdefault("refined_query", "")
            
            # Ensure types are correct
            if not isinstance(result["reason"], str):
                result["reason"] = str(result["reason"])
                
            if not isinstance(result["missing"], list):
                result["missing"] = []
                
            if not isinstance(result["refined_query"], str):
                result["refined_query"] = str(result["refined_query"])
            
            return result
            
        else:
            # No JSON found in response
            raise ValueError("No JSON object found in response")
            
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Re-raise for the safe wrapper to catch
        raise
    except Exception as e:
        # Any other parsing error
        raise ValueError(f"Failed to parse critic response: {e}")