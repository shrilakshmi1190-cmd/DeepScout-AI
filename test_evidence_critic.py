"""
Tests for evidence critic functionality.
"""
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.evidence_critic import (
    criticize_evidence_safely,
    _format_pdf_evidence,
    _format_web_evidence,
    _parse_critic_response
)


class TestEvidenceCritic(unittest.TestCase):
    
    def test_format_pdf_evidence_empty(self):
        """Test formatting empty PDF evidence."""
        result = _format_pdf_evidence([])
        self.assertEqual(result, "(No PDF evidence)")
    
    def test_format_pdf_evidence_with_chunks(self):
        """Test formatting PDF evidence with chunks."""
        pdf_results = [
            {
                "chunk": "The Transformer achieved 41.0 BLEU on WMT 2014 English-to-German.",
                "page": 5,
                "source": "sample.pdf"
            },
            {
                "chunk": "It uses multi-head attention mechanism.",
                "page": 7,
                "source": "sample.pdf"
            }
        ]
        
        result = _format_pdf_evidence(pdf_results)
        
        self.assertIn("Transformer achieved 41.0 BLEU", result)
        self.assertIn("Page 5", result)
        self.assertIn("PDF Source 1", result)
    
    def test_format_web_evidence_empty(self):
        """Test formatting empty web evidence."""
        result = _format_web_evidence([])
        self.assertEqual(result, "(No web evidence)")
    
    def test_format_web_evidence_with_results(self):
        """Test formatting web evidence with results."""
        web_results = [
            {
                "title": "Transformer Architecture",
                "content": "The Transformer model was introduced in 2017.",
                "url": "https://example.com/transformer"
            }
        ]
        
        result = _format_web_evidence(web_results)
        
        self.assertIn("Transformer Architecture", result)
        self.assertIn("introduced in 2017", result)
        self.assertIn("Web Source 1", result)
    
    def test_parse_critic_response_valid_json(self):
        """Test parsing valid JSON critic response."""
        llm_response = '{"sufficient": false, "reason": "Missing English-to-German", "missing": ["English-to-German BLEU score"], "refined_query": "Transformer English German BLEU"}'
        
        result = _parse_critic_response(llm_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["sufficient"], False)
        self.assertEqual(result["reason"], "Missing English-to-German")
        self.assertEqual(result["missing"], ["English-to-German BLEU score"])
        self.assertEqual(result["refined_query"], "Transformer English German BLEU")
    
    def test_parse_critic_response_with_text_wrapper(self):
        """Test parsing critic response wrapped in text."""
        llm_response = """
        The evidence shows English-to-French but not English-to-German.
        
        {
          "sufficient": false,
          "reason": "Evidence shows English-to-French, not English-to-German",
          "missing": ["English-to-German BLEU score"],
          "refined_query": "Transformer WMT 2014 English German BLEU score"
        }
        
        Please use this for targeted retrieval.
        """
        
        result = _parse_critic_response(llm_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["sufficient"], False)
        self.assertIn("English-to-French", result["reason"])
        self.assertEqual(result["missing"], ["English-to-German BLEU score"])
    
    def test_parse_critic_response_string_boolean(self):
        """Test parsing critic response with string boolean."""
        llm_response = '{"sufficient": "true", "reason": "", "missing": [], "refined_query": ""}'
        
        result = _parse_critic_response(llm_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["sufficient"], True)
    
    def test_parse_critic_response_invalid_json(self):
        """Test parsing invalid JSON critic response."""
        llm_response = "This is not JSON at all"
        
        with self.assertRaises(ValueError):
            _parse_critic_response(llm_response)
    
    def test_parse_critic_response_missing_sufficient(self):
        """Test parsing critic response missing sufficient field."""
        llm_response = '{"reason": "test", "missing": [], "refined_query": ""}'
        
        with self.assertRaises(KeyError):
            _parse_critic_response(llm_response)
    
    @patch('research.evidence_critic.criticize_evidence')
    def test_criticize_evidence_safely_success(self, mock_criticize):
        """Test safe critic wrapper with successful result."""
        mock_criticize.return_value = {
            "sufficient": False,
            "reason": "Semantic mismatch",
            "missing": ["specific detail"],
            "refined_query": "better query"
        }
        
        result = criticize_evidence_safely(
            question="test question",
            evidence=[{"chunk": "test evidence"}],
            route="document",
            evidence_type="pdf"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["sufficient"], False)
        self.assertEqual(result["reason"], "Semantic mismatch")
    
    @patch('research.evidence_critic.criticize_evidence')
    def test_criticize_evidence_safely_critic_returns_none(self, mock_criticize):
        """Test safe critic wrapper when critic returns None."""
        mock_criticize.return_value = None
        
        result = criticize_evidence_safely(
            question="test question",
            evidence=[{"chunk": "test evidence"}],
            route="document",
            evidence_type="pdf"
        )
        
        self.assertIsNone(result)
    
    @patch('research.evidence_critic.criticize_evidence')
    def test_criticize_evidence_safely_invalid_structure(self, mock_criticize):
        """Test safe critic wrapper with invalid result structure."""
        mock_criticize.return_value = {"not_sufficient": True}
        
        result = criticize_evidence_safely(
            question="test question",
            evidence=[{"chunk": "test evidence"}],
            route="document",
            evidence_type="pdf"
        )
        
        self.assertIsNone(result)
    
    @patch('research.evidence_critic.criticize_evidence')
    def test_criticize_evidence_safely_string_instead_of_boolean(self, mock_criticize):
        """Test safe critic wrapper with string instead of boolean."""
        mock_criticize.return_value = {"sufficient": "maybe"}
        
        result = criticize_evidence_safely(
            question="test question",
            evidence=[{"chunk": "test evidence"}],
            route="document",
            evidence_type="pdf"
        )
        
        self.assertIsNone(result)
    
    @patch('research.evidence_critic.generate_answer')
    def test_bleu_regression_case(self, mock_generate):
        """Regression test for BLEU English-German vs English-French mismatch."""
        question = "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation task?"
        
        # Mock evidence containing only English-to-French result
        pdf_results = [
            {
                "chunk": "The Transformer model achieved 41.0 BLEU on the WMT 2014 English-to-French translation task.",
                "page": 3,
                "source": "transformer_paper.pdf"
            }
        ]
        
        # Mock LLM response correctly identifying the mismatch
        mock_generate.return_value = """
        {
          "sufficient": false,
          "reason": "Evidence shows English-to-French BLEU score, but question asks for English-to-German",
          "missing": ["English-to-German BLEU score"],
          "refined_query": "Transformer WMT 2014 English German BLEU score"
        }
        """
        
        # Import the actual criticize_evidence function
        from research.evidence_critic import criticize_evidence
        
        result = criticize_evidence(
            question=question,
            evidence=pdf_results,
            route="document",
            evidence_type="pdf"
        )
        
        # Critic should catch the semantic mismatch
        self.assertIsNotNone(result)
        self.assertEqual(result["sufficient"], False)
        self.assertIn("English-to-German", "".join(result.get("missing", [])) or result.get("reason", ""))
    
    @patch('research.evidence_critic.generate_answer')
    def test_critic_sufficient_case(self, mock_generate):
        """Test critic with sufficient evidence."""
        question = "What architecture does this paper propose?"
        
        pdf_results = [
            {
                "chunk": "The paper proposes a Transformer architecture with multi-head attention.",
                "page": 2,
                "source": "paper.pdf"
            }
        ]
        
        mock_generate.return_value = """
        {
          "sufficient": true,
          "reason": "",
          "missing": [],
          "refined_query": ""
        }
        """
        
        from research.evidence_critic import criticize_evidence
        
        result = criticize_evidence(
            question=question,
            evidence=pdf_results,
            route="document",
            evidence_type="pdf"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["sufficient"], True)
    
    @patch('research.evidence_critic.generate_answer')
    def test_critic_mixed_evidence(self, mock_generate):
        """Test critic with mixed evidence (both PDF and web)."""
        question = "Compare Transformer with modern architectures."
        
        mixed_evidence = {
            "pdf": [
                {
                    "chunk": "Transformer uses self-attention mechanism.",
                    "page": 4,
                    "source": "transformer.pdf"
                }
            ],
            "web": [
                {
                    "title": "Modern Architectures",
                    "content": "Modern architectures include Efficient Transformers.",
                    "url": "https://example.com/modern"
                }
            ]
        }
        
        mock_generate.return_value = """
        {
          "sufficient": true,
          "reason": "",
          "missing": [],
          "refined_query": ""
        }
        """
        
        from research.evidence_critic import criticize_evidence
        
        result = criticize_evidence(
            question=question,
            evidence=mixed_evidence,
            route="mixed",
            evidence_type="pdf"  # Type doesn't matter, will be handled by formatter
        )
        
        self.assertIsNotNone(result)
    
    def test_criticize_evidence_safely_json_error(self):
        """Test safe critic wrapper handling JSON parsing error."""
        with patch('research.evidence_critic.criticize_evidence') as mock_criticize:
            mock_criticize.side_effect = ValueError("JSON parsing error")
            
            result = criticize_evidence_safely(
                question="test question",
                evidence=[{"chunk": "test"}],
                route="document",
                evidence_type="pdf"
            )
            
            self.assertIsNone(result)
    
    def test_criticize_evidence_safely_key_error(self):
        """Test safe critic wrapper handling KeyError."""
        with patch('research.evidence_critic.criticize_evidence') as mock_criticize:
            mock_criticize.side_effect = KeyError("Missing key")
            
            result = criticize_evidence_safely(
                question="test question",
                evidence=[{"chunk": "test"}],
                route="document",
                evidence_type="pdf"
            )
            
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()