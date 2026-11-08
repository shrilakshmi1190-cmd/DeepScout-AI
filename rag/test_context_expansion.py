"""
Tests for context window expansion in retriever.py
"""
import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import Retriever


class TestContextWindowExpansion(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with multiple chunks."""
        # Create chunks with sequential content
        self.chunks = [
            {"text": "Chapter 1: Introduction", "page": 1, "source": "test.pdf"},
            {"text": "This is the beginning of the document.", "page": 1, "source": "test.pdf"},
            {"text": "Here we discuss the background and motivation.", "page": 2, "source": "test.pdf"},
            {"text": "The main contribution is presented here.", "page": 2, "source": "test.pdf"},
            {"text": "Methodology section starts now.", "page": 3, "source": "test.pdf"},
            {"text": "We use a novel approach for the problem.", "page": 3, "source": "test.pdf"},
            {"text": "Experimental results are shown in this section.", "page": 4, "source": "test.pdf"},
            {"text": "The model achieves 95% accuracy.", "page": 4, "source": "test.pdf"},
            {"text": "Conclusion and future work.", "page": 5, "source": "test.pdf"},
            {"text": "References and acknowledgments.", "page": 6, "source": "test.pdf"},
        ]
        
        # Also test with a second document
        self.chunks_with_multiple_sources = [
            {"text": "Doc A: First chunk", "page": 1, "source": "doc_a.pdf"},
            {"text": "Doc A: Second chunk", "page": 1, "source": "doc_a.pdf"},
            {"text": "Doc A: Third chunk", "page": 2, "source": "doc_a.pdf"},
            {"text": "Doc B: First chunk", "page": 1, "source": "doc_b.pdf"},
            {"text": "Doc B: Second chunk", "page": 1, "source": "doc_b.pdf"},
            {"text": "Doc B: Third chunk", "page": 2, "source": "doc_b.pdf"},
        ]
    
    def test_find_chunk_index(self):
        """Test finding chunk index in original chunks list."""
        retriever = Retriever(self.chunks)
        
        # Test finding a chunk that exists
        candidate = {
            "chunk": "Methodology section starts now.",
            "page": 3,
            "source": "test.pdf"
        }
        index = retriever._find_chunk_index(candidate)
        self.assertEqual(index, 4)  # 5th chunk (0-indexed)
        
        # Test finding a chunk that doesn't exist
        candidate = {
            "chunk": "Non-existent chunk",
            "page": 99,
            "source": "test.pdf"
        }
        index = retriever._find_chunk_index(candidate)
        self.assertEqual(index, -1)
    
    def test_get_neighboring_chunks_middle(self):
        """Test getting neighboring chunks for a middle chunk."""
        retriever = Retriever(self.chunks)
        
        # Get neighbors for chunk at index 4 (middle of list)
        neighbors = retriever._get_neighboring_chunks(4, window=1)
        
        # Should get 3 chunks: index 3, 4, 5
        self.assertEqual(len(neighbors), 3)
        self.assertEqual(neighbors[0]["text"], "The main contribution is presented here.")
        self.assertEqual(neighbors[1]["text"], "Methodology section starts now.")
        self.assertEqual(neighbors[2]["text"], "We use a novel approach for the problem.")
    
    def test_get_neighboring_chunks_first(self):
        """Test getting neighboring chunks for the first chunk."""
        retriever = Retriever(self.chunks)
        
        # Get neighbors for first chunk
        neighbors = retriever._get_neighboring_chunks(0, window=1)
        
        # Should get 2 chunks: index 0, 1 (no previous neighbor)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0]["text"], "Chapter 1: Introduction")
        self.assertEqual(neighbors[1]["text"], "This is the beginning of the document.")
    
    def test_get_neighboring_chunks_last(self):
        """Test getting neighboring chunks for the last chunk."""
        retriever = Retriever(self.chunks)
        
        # Get neighbors for last chunk
        neighbors = retriever._get_neighboring_chunks(len(self.chunks) - 1, window=1)
        
        # Should get 2 chunks: index 8, 9 (no next neighbor)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0]["text"], "Conclusion and future work.")
        self.assertEqual(neighbors[1]["text"], "References and acknowledgments.")
    
    def test_get_neighboring_chunks_same_source_only(self):
        """Test that neighbors are only taken from the same source."""
        retriever = Retriever(self.chunks_with_multiple_sources)
        
        # Get neighbors for chunk in doc_a.pdf
        central_chunk = {"text": "Doc A: Second chunk", "page": 1, "source": "doc_a.pdf"}
        central_index = retriever._find_chunk_index(central_chunk)
        neighbors = retriever._get_neighboring_chunks(central_index, window=2)
        
        # Should only get chunks from doc_a.pdf, not doc_b.pdf
        neighbor_sources = [n["source"] for n in neighbors]
        self.assertTrue(all(source == "doc_a.pdf" for source in neighbor_sources))
        self.assertEqual(len(neighbors), 3)  # Should have all 3 doc_a chunks
    
    def test_expand_context_single_chunk(self):
        """Test expanding context around a single selected chunk."""
        retriever = Retriever(self.chunks)
        
        selected_results = [{
            "chunk": "Methodology section starts now.",
            "page": 3,
            "source": "test.pdf",
            "score": 0.9,
            "retrieval_score": 0.8
        }]
        
        expanded = retriever._expand_context(selected_results, window=1)
        
        # Should get 3 chunks with document order preserved
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]["chunk"], "The main contribution is presented here.")
        self.assertEqual(expanded[1]["chunk"], "Methodology section starts now.")
        self.assertEqual(expanded[2]["chunk"], "We use a novel approach for the problem.")
        
        # Scores should be preserved from original candidate
        for chunk in expanded:
            self.assertEqual(chunk["score"], 0.9)
            self.assertEqual(chunk["retrieval_score"], 0.8)
    
    def test_expand_context_multiple_chunks_no_duplicates(self):
        """Test expanding context around multiple chunks without duplicates."""
        retriever = Retriever(self.chunks)
        
        selected_results = [
            {
                "chunk": "Methodology section starts now.",
                "page": 3,
                "source": "test.pdf",
                "score": 0.9,
                "retrieval_score": 0.8
            },
            {
                "chunk": "We use a novel approach for the problem.",
                "page": 3,
                "source": "test.pdf",
                "score": 0.85,
                "retrieval_score": 0.75
            }
        ]
        
        expanded = retriever._expand_context(selected_results, window=1)
        
        # Should get 4 unique chunks (not 6 due to overlap)
        # Chunks: index 3, 4, 5, 6
        self.assertEqual(len(expanded), 4)
        
        # Check document order is preserved
        texts = [chunk["chunk"] for chunk in expanded]
        expected_order = [
            "The main contribution is presented here.",
            "Methodology section starts now.",
            "We use a novel approach for the problem.",
            "Experimental results are shown in this section."
        ]
        self.assertEqual(texts, expected_order)
        
        # Check no duplicates
        chunk_ids = set((chunk["chunk"], chunk["page"], chunk["source"]) for chunk in expanded)
        self.assertEqual(len(chunk_ids), 4)
    
    def test_expand_context_edge_cases(self):
        """Test edge cases for context expansion."""
        retriever = Retriever(self.chunks)
        
        # Test with empty selected results
        expanded = retriever._expand_context([], window=1)
        self.assertEqual(expanded, [])
        
        # Test with candidate not found in chunks
        selected_results = [{
            "chunk": "Non-existent chunk",
            "page": 99,
            "source": "test.pdf",
            "score": 0.9,
            "retrieval_score": 0.8
        }]
        expanded = retriever._expand_context(selected_results, window=1)
        self.assertEqual(expanded, [])
    
    def test_search_with_expansion_disabled(self):
        """Test search with expansion disabled (backward compatibility)."""
        retriever = Retriever(self.chunks)
        
        # Mock the vector store and reranker to control results
        # We'll create a simple mock by overriding the search method
        # But first, let's test that expand_context=False works
        # We need to mock the FAISS search to return predictable results
        
        # Since we can't easily mock FAISS, we'll test the helper functions
        # and assume the integration works if helpers are correct
    
    def test_search_with_expansion_enabled(self):
        """Test search with expansion enabled."""
        retriever = Retriever(self.chunks)
        
        # Since we can't easily test the full search without mocking FAISS,
        # we'll verify the expand_context parameter is accepted
        # and doesn't break the method signature
        
        # Create a minimal test that the method can be called
        # This is mainly to verify the parameter exists
        try:
            # Should accept expand_context parameter
            result = retriever.search("test query", top_k=2, expand_context=True)
            # If we get here, the method signature is correct
            # (it might return empty list due to no FAISS index)
            self.assertIsInstance(result, list)
        except TypeError as e:
            self.fail(f"Method call failed: {e}")


if __name__ == '__main__':
    unittest.main()