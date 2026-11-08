"""
Tests for evidence sufficiency checking in orchestrator.py
"""
import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.orchestrator import (
    _check_document_sufficiency,
    _check_web_sufficiency,
    _check_mixed_sufficiency,
    _refine_web_query_for_gap
)


class TestEvidenceSufficiency(unittest.TestCase):
    
    def test_document_sufficiency_no_evidence(self):
        """Test document sufficiency with no results."""
        query = "What architecture does this paper propose?"
        results = []
        
        sufficient, gap, refinement = _check_document_sufficiency(query, results)
        
        self.assertFalse(sufficient)
        self.assertIn("No PDF evidence", gap)
        self.assertIn("retry_with_more", refinement)
    
    def test_document_sufficiency_with_evidence(self):
        """Test document sufficiency with results."""
        query = "What architecture does this paper propose?"
        results = [
            {"chunk": "The paper proposes a Transformer architecture.", "score": 2.5},
            {"chunk": "It uses multi-head attention.", "score": 1.8}
        ]
        
        sufficient, gap, refinement = _check_document_sufficiency(query, results)
        
        self.assertTrue(sufficient)
        self.assertIn("PDF evidence retrieved", gap)
        self.assertEqual(refinement, {})
    
    def test_web_sufficiency_no_evidence(self):
        """Test web sufficiency with no results."""
        query = "What is the latest version of Python?"
        results = []
        
        sufficient, gap, refinement = _check_web_sufficiency(query, results)
        
        self.assertFalse(sufficient)
        self.assertIn("No web evidence", gap)
        self.assertIn("any evidence", refinement.get("targeted_aspects", []))
    
    def test_web_sufficiency_with_evidence(self):
        """Test web sufficiency with results."""
        query = "What is the latest version of Python?"
        results = [
            {"title": "Python 3.12 Release", "content": "Python 3.12 is the latest..."},
            {"title": "Python Downloads", "content": "Download Python 3.12..."}
        ]
        
        sufficient, gap, refinement = _check_web_sufficiency(query, results)
        
        self.assertTrue(sufficient)
        self.assertIn("Web evidence retrieved", gap)
        self.assertEqual(refinement, {})
    
    def test_mixed_sufficiency_no_evidence(self):
        """Test mixed sufficiency with no results from either side."""
        query = "How does this compare with modern approaches?"
        pdf_results = []
        web_results = []
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        self.assertFalse(sufficient)
        self.assertIn("No evidence from PDF or web", gap)
        self.assertEqual(refinement.get("missing_side"), "both")
    
    def test_mixed_sufficiency_comparison_missing_pdf(self):
        """Test mixed sufficiency for comparison missing PDF evidence."""
        query = "Compare this with modern approaches."
        pdf_results = []
        web_results = [
            {"title": "Modern Approaches", "content": "Modern approaches use..."}
        ]
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        self.assertFalse(sufficient)
        self.assertIn("Comparison requires PDF evidence", gap)
        self.assertEqual(refinement.get("missing_side"), "pdf")
    
    def test_mixed_sufficiency_comparison_missing_web(self):
        """Test mixed sufficiency for comparison missing web evidence."""
        query = "Compare this with modern approaches."
        pdf_results = [
            {"chunk": "The paper proposes method X.", "score": 2.5}
        ]
        web_results = []
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        self.assertFalse(sufficient)
        self.assertIn("Comparison requires web evidence", gap)
        self.assertEqual(refinement.get("missing_side"), "web")
    
    def test_mixed_sufficiency_modern_topic_missing_web(self):
        """Test mixed sufficiency for modern topic missing web evidence."""
        query = "What are the modern Transformer architectures?"
        pdf_results = [
            {"chunk": "The paper discusses attention.", "score": 2.0}
        ]
        web_results = []
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        self.assertFalse(sufficient)
        self.assertIn("modern/current topic requires web evidence", gap)
        self.assertEqual(refinement.get("missing_side"), "web")
        self.assertIn("modern/current", refinement.get("targeted_aspects", []))
    
    def test_mixed_sufficiency_sufficient_both_sides(self):
        """Test mixed sufficiency with both sides present."""
        query = "How does this compare with modern approaches?"
        pdf_results = [
            {"chunk": "The paper proposes method X.", "score": 2.5}
        ]
        web_results = [
            {"title": "Modern Approaches", "content": "Modern approaches use..."}
        ]
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        self.assertTrue(sufficient)
        self.assertIn("Mixed evidence retrieved", gap)
    
    def test_refine_web_query_modern_gap(self):
        """Test web query refinement for modern/current gap."""
        query = "Transformer architectures"
        gap_analysis = {"targeted_aspects": ["modern/current"]}
        
        refined = _refine_web_query_for_gap(query, gap_analysis)
        
        self.assertIn("current", refined.lower())
        self.assertIn("Transformer", refined)
    
    def test_refine_web_query_already_has_current(self):
        """Test web query refinement when query already has current term."""
        query = "current Transformer architectures"
        gap_analysis = {"targeted_aspects": ["modern/current"]}
        
        refined = _refine_web_query_for_gap(query, gap_analysis)
        
        # Should not duplicate "current"
        self.assertEqual(refined.lower().count("current"), 1)
    
    def test_refine_web_query_any_evidence_gap_short(self):
        """Test web query refinement for any evidence gap with short query."""
        query = "Python"
        gap_analysis = {"targeted_aspects": ["any evidence"]}
        
        refined = _refine_web_query_for_gap(query, gap_analysis)
        
        self.assertIn("information", refined.lower())
    
    def test_refine_web_query_any_evidence_gap_long(self):
        """Test web query refinement for any evidence gap with long query."""
        query = "latest version of Python programming language"
        gap_analysis = {"targeted_aspects": ["any evidence"]}
        
        refined = _refine_web_query_for_gap(query, gap_analysis)
        
        # Long query shouldn't get "information" added
        self.assertEqual(refined, query)
    
    def test_mixed_sufficiency_non_comparison_only_pdf(self):
        """Test mixed sufficiency for non-comparison query with only PDF."""
        query = "What method does this paper use?"
        pdf_results = [
            {"chunk": "The paper uses method X.", "score": 2.5}
        ]
        web_results = []
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        # Non-comparison with only PDF should be sufficient
        self.assertTrue(sufficient)
    
    def test_mixed_sufficiency_non_comparison_only_web(self):
        """Test mixed sufficiency for non-comparison query with only web."""
        query = "What are common machine learning methods?"
        pdf_results = []
        web_results = [
            {"title": "ML Methods", "content": "Common methods include..."}
        ]
        
        sufficient, gap, refinement = _check_mixed_sufficiency(query, pdf_results, web_results)
        
        # Non-comparison with only web should be sufficient
        self.assertTrue(sufficient)


if __name__ == '__main__':
    unittest.main()