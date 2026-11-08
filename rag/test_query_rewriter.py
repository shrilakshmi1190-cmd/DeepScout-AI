"""
Tests for query_rewriter.py
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_rewriter import (
    _build_rewriting_history,
    _create_rewrite_prompt,
    _is_valid_rewrite,
    rewrite_query_for_retrieval
)


class TestQueryRewriter(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.sample_history = [
            {"role": "user", "content": "What architecture does this paper propose?", "route": "document"},
            {"role": "assistant", "content": "The paper proposes a Transformer-based architecture for NLP tasks."},
            {"role": "user", "content": "What about its limitations?", "route": "document"},
        ]
        
        self.sample_web_history = [
            {"role": "user", "content": "What's the latest version of Python?", "route": "web"},
            {"role": "assistant", "content": "Python 3.12 is the latest stable version as of 2024."},
            {"role": "user", "content": "And what about its performance improvements?", "route": "web"},
        ]
    
    def test_build_rewriting_history_no_history(self):
        """Test building history with no history."""
        result = _build_rewriting_history(None)
        self.assertEqual(result, "")
        
        result = _build_rewriting_history([])
        self.assertEqual(result, "")
    
    def test_build_rewriting_history_with_history(self):
        """Test building history with conversation."""
        result = _build_rewriting_history(self.sample_history)
        self.assertIn("CONVERSATION HISTORY FOR CONTEXT:", result)
        self.assertIn("USER [document]: What architecture does this paper propose?", result)
        self.assertIn("ASSISTANT: The paper proposes a Transformer-based architecture for NLP tasks.", result)
        self.assertIn("USER [document]: What about its limitations?", result)
    
    def test_build_rewriting_history_limits_messages(self):
        """Test that history is limited to recent messages."""
        long_history = [
            {"role": "user", "content": f"Message {i}", "route": "document"}
            for i in range(10)
        ]
        result = _build_rewriting_history(long_history)
        # Should only include last 4 messages
        self.assertIn("Message 6", result)
        self.assertIn("Message 7", result)
        self.assertIn("Message 8", result)
        self.assertIn("Message 9", result)
        self.assertNotIn("Message 5", result)
    
    def test_create_rewrite_prompt_document(self):
        """Test document route prompt creation."""
        prompt = _create_rewrite_prompt(
            "What about its limitations?",
            "CONVERSATION HISTORY FOR CONTEXT:\nUSER [document]: What architecture does this paper propose?",
            "document"
        )
        self.assertIn("Rewrite this follow-up question to be a self-contained query for document retrieval/search.", prompt)
        self.assertIn("Resolve all pronouns", prompt)
        self.assertIn("Keep the question focused on the content of the uploaded PDF/document", prompt)
    
    def test_create_rewrite_prompt_web(self):
        """Test web route prompt creation."""
        prompt = _create_rewrite_prompt(
            "And what about its performance improvements?",
            "CONVERSATION HISTORY FOR CONTEXT:\nUSER [web]: What's the latest version of Python?",
            "web"
        )
        self.assertIn("Rewrite this follow-up question to be a self-contained query for web retrieval/search.", prompt)
        self.assertIn("Make the question suitable for web search", prompt)
        self.assertIn("Add current date/time context if relevant", prompt)
    
    def test_create_rewrite_prompt_mixed(self):
        """Test mixed route prompt creation."""
        prompt = _create_rewrite_prompt(
            "How does it compare to current approaches?",
            "CONVERSATION HISTORY FOR CONTEXT:\nUSER [mixed]: What method does the paper describe?",
            "mixed"
        )
        self.assertIn("Rewrite this follow-up question to be a self-contained query for mixed retrieval/search.", prompt)
        self.assertIn("Rewrite to be self-contained for both document retrieval and web search", prompt)
        self.assertIn("Maintain the comparative nature of the question", prompt)
    
    def test_is_valid_rewrite_valid(self):
        """Test valid rewrite validation."""
        self.assertTrue(_is_valid_rewrite("What are the limitations of the Transformer architecture?", "What about its limitations?"))
        self.assertTrue(_is_valid_rewrite("How did they evaluate the model?", "How did they evaluate it?"))
    
    def test_is_valid_rewrite_invalid(self):
        """Test invalid rewrite validation."""
        # Empty or whitespace
        self.assertFalse(_is_valid_rewrite("", "original"))
        self.assertFalse(_is_valid_rewrite("   ", "original"))
        
        # Too short
        self.assertFalse(_is_valid_rewrite("a", "original"))
        self.assertFalse(_is_valid_rewrite("it", "original"))
        
        # Contains placeholders
        self.assertFalse(_is_valid_rewrite("What are the limitations of [the architecture]?", "original"))
        
        # Contains refusal phrases
        self.assertFalse(_is_valid_rewrite("I cannot determine what you're referring to.", "original"))
        self.assertFalse(_is_valid_rewrite("I don't know how to rewrite this.", "original"))
    
    @patch('rag.query_rewriter._is_follow_up')
    def test_rewrite_query_for_retrieval_general_route(self, mock_is_follow_up):
        """Test that general route returns original question unchanged."""
        mock_is_follow_up.return_value = True
        
        result = rewrite_query_for_retrieval(
            "What about this?",
            self.sample_history,
            "general"
        )
        
        self.assertEqual(result, "What about this?")
        mock_is_follow_up.assert_not_called()  # Should not even check if follow-up for general
    
    @patch('rag.query_rewriter._is_follow_up')
    def test_rewrite_query_for_retrieval_not_follow_up(self, mock_is_follow_up):
        """Test that non-follow-up returns original question."""
        mock_is_follow_up.return_value = False
        
        result = rewrite_query_for_retrieval(
            "What architecture does this paper propose?",
            self.sample_history,
            "document"
        )
        
        self.assertEqual(result, "What architecture does this paper propose?")
        mock_is_follow_up.assert_called_once()
    
    @patch('rag.query_rewriter._is_follow_up')
    @patch('llm.llm_service.generate_answer')
    def test_rewrite_query_for_retrieval_successful(self, mock_generate_answer, mock_is_follow_up):
        """Test successful query rewriting."""
        mock_is_follow_up.return_value = True
        
        # Mock LLM to return a valid rewritten query
        mock_generate_answer.return_value = "What are the limitations of the Transformer-based architecture proposed in the paper?"
        
        result = rewrite_query_for_retrieval(
            "What about its limitations?",
            self.sample_history,
            "document"
        )
        
        # Should return the rewritten query
        self.assertEqual(result, "What are the limitations of the Transformer-based architecture proposed in the paper?")
        mock_is_follow_up.assert_called_once()
        mock_generate_answer.assert_called_once()
    
    @patch('rag.query_rewriter._is_follow_up')
    @patch('llm.llm_service.generate_answer')
    def test_rewrite_query_for_retrieval_llm_failure(self, mock_generate_answer, mock_is_follow_up):
        """Test that LLM failure returns original question."""
        mock_is_follow_up.return_value = True
        
        # Mock LLM to raise an exception
        mock_generate_answer.side_effect = Exception("LLM service unavailable")
        
        result = rewrite_query_for_retrieval(
            "What about its limitations?",
            self.sample_history,
            "document"
        )
        
        # Should return original question when LLM fails
        self.assertEqual(result, "What about its limitations?")
        mock_is_follow_up.assert_called_once()
        mock_generate_answer.assert_called_once()
    
    @patch('rag.query_rewriter._is_follow_up')
    @patch('llm.llm_service.generate_answer')
    def test_rewrite_query_for_retrieval_invalid_rewrite(self, mock_generate_answer, mock_is_follow_up):
        """Test that invalid rewrite returns original question."""
        mock_is_follow_up.return_value = True
        
        # Mock LLM to return invalid output (empty string) that fails _is_valid_rewrite
        mock_generate_answer.return_value = ""
        
        result = rewrite_query_for_retrieval(
            "What about its limitations?",
            self.sample_history,
            "document"
        )
        
        # Should return original question when rewrite is invalid
        self.assertEqual(result, "What about its limitations?")
        mock_is_follow_up.assert_called_once()
        mock_generate_answer.assert_called_once()
    
    def test_rewrite_query_for_retrieval_no_history(self):
        """Test rewriting with no history."""
        result = rewrite_query_for_retrieval(
            "What about its limitations?",
            None,
            "document"
        )
        
        self.assertEqual(result, "What about its limitations?")
    
    def test_rewrite_query_for_retrieval_insufficient_history(self):
        """Test rewriting with insufficient history (less than 2 messages)."""
        short_history = [
            {"role": "user", "content": "What architecture does this paper propose?", "route": "document"}
        ]
        
        result = rewrite_query_for_retrieval(
            "What about its limitations?",
            short_history,
            "document"
        )
        
        self.assertEqual(result, "What about its limitations?")


if __name__ == '__main__':
    unittest.main()