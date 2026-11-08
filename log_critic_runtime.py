"""
Logging wrapper to see actual Evidence Critic runtime behavior.
"""
import sys
import os
import json
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import and wrap the critic functions
from research.evidence_critic import (
    criticize_evidence as original_criticize_evidence,
    criticize_evidence_safely as original_criticize_evidence_safely,
    _create_critic_prompt,
    _format_evidence_for_critic
)

def logged_criticize_evidence(question, evidence, route, evidence_type):
    """Logged version of criticize_evidence."""
    logger.info("="*60)
    logger.info("EVIDENCE CRITIC CALLED")
    logger.info(f"Question: {question}")
    logger.info(f"Route: {route}, Evidence Type: {evidence_type}")
    
    # Log evidence being sent
    if isinstance(evidence, dict) and "pdf" in evidence and "web" in evidence:
        logger.info("Mixed evidence (PDF + Web)")
        if evidence.get("pdf"):
            logger.info(f"PDF evidence: {len(evidence['pdf'])} chunks")
            for i, chunk in enumerate(evidence['pdf'][:2]):  # First 2 chunks
                content_preview = chunk.get('chunk', '')[:100]
                logger.info(f"  PDF chunk {i}: {content_preview}...")
        if evidence.get("web"):
            logger.info(f"Web evidence: {len(evidence['web'])} results")
            for i, result in enumerate(evidence['web'][:2]):  # First 2 results
                title = result.get('title', 'No title')
                logger.info(f"  Web result {i}: {title}")
    elif evidence_type == "pdf":
        logger.info(f"PDF evidence: {len(evidence) if isinstance(evidence, list) else 'unknown'} chunks")
        if evidence and isinstance(evidence, list):
            for i, chunk in enumerate(evidence[:2]):  # First 2 chunks
                content_preview = chunk.get('chunk', '')[:100]
                logger.info(f"  Chunk {i}: {content_preview}...")
    else:  # web
        logger.info(f"Web evidence: {len(evidence) if isinstance(evidence, list) else 'unknown'} results")
        if evidence and isinstance(evidence, list):
            for i, result in enumerate(evidence[:2]):  # First 2 results
                title = result.get('title', 'No title')
                logger.info(f"  Result {i}: {title}")
    
    # Get formatted evidence for prompt preview
    formatted = _format_evidence_for_critic(evidence, evidence_type)
    formatted_preview = formatted[:500] + "..." if len(formatted) > 500 else formatted
    logger.info(f"Formatted evidence preview: {formatted_preview}")
    
    # Call original function
    logger.info("Calling original critic...")
    try:
        result = original_criticize_evidence(question, evidence, route, evidence_type)
        logger.info(f"Critic returned: {result}")
        return result
    except Exception as e:
        logger.error(f"Critic failed with error: {e}")
        raise

def logged_criticize_evidence_safely(question, evidence, route, evidence_type):
    """Logged version of criticize_evidence_safely."""
    logger.info("="*60)
    logger.info("CRITIC SAFE WRAPPER CALLED")
    logger.info(f"Question: {question[:100]}...")
    
    try:
        result = original_criticize_evidence_safely(question, evidence, route, evidence_type)
        if result is None:
            logger.warning("Critic returned None (failed or invalid)")
        else:
            logger.info(f"Critic result - Sufficient: {result.get('sufficient')}")
            if not result.get('sufficient'):
                logger.info(f"  Reason: {result.get('reason', 'No reason')}")
                logger.info(f"  Missing: {result.get('missing', [])}")
                logger.info(f"  Refined query: {result.get('refined_query', 'No query')}")
        return result
    except Exception as e:
        logger.error(f"Safe wrapper error: {e}")
        return None

# Patch the critic module
import research.evidence_critic
research.evidence_critic.criticize_evidence = logged_criticize_evidence
research.evidence_critic.criticize_evidence_safely = logged_criticize_evidence_safely

# Also patch in orchestrator
import research.orchestrator
research.orchestrator.criticize_evidence_safely = logged_criticize_evidence_safely

# Test with mock retriever
def test_with_mock():
    """Test the logged critic with mock data."""
    from unittest.mock import Mock
    
    # Create mock retriever
    mock_retriever = Mock()
    mock_retriever.search.return_value = [
        {
            "chunk": "The Transformer model achieved 41.0 BLEU on the WMT 2014 English-to-French translation task.",
            "page": 3,
            "source": "sample.pdf",
            "score": 2.5
        }
    ]
    
    # Mock LLM response for critic
    from unittest.mock import patch
    
    with patch('research.evidence_critic.generate_answer') as mock_generate:
        # Mock critic response for BLEU case
        mock_response = json.dumps({
            "sufficient": False,
            "reason": "Evidence shows English-to-French, not English-to-German",
            "missing": ["English-to-German BLEU score"],
            "refined_query": "Transformer WMT 2014 English German BLEU score"
        })
        mock_generate.return_value = mock_response
        
        # Test BLEU case
        logger.info("\n" + "="*60)
        logger.info("TESTING BLEU ENGLISH-GERMAN CASE")
        logger.info("="*60)
        
        from research.orchestrator import _run_document
        
        result = _run_document(
            question="What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation task?",
            retriever=mock_retriever,
            warnings=[],
            history=None
        )
        
        logger.info(f"Final answer length: {len(result.answer)} chars")
        logger.info(f"Answer preview: {result.answer[:200]}...")

if __name__ == "__main__":
    print("Logging Evidence Critic runtime behavior")
    print("="*60)
    print("\nThis script patches the critic to add logging.")
    print("Run your Streamlit app with this script imported to see logs.")
    print("\nTo use: import this module before running your app.")
    print("\nExample test:")
    test_with_mock()