"""
Simple test for retriever context expansion without external dependencies.
"""
import sys
import os

# Mock the sentence_transformers and FAISS dependencies
class MockSentenceTransformer:
    def encode(self, texts, normalize_embeddings=True):
        # Return dummy embeddings
        import numpy as np
        if isinstance(texts, list):
            return np.random.randn(len(texts), 384).astype("float32")
        else:
            return np.random.randn(1, 384).astype("float32")[0]

class MockCrossEncoder:
    def predict(self, pairs):
        # Return dummy scores
        import numpy as np
        return np.random.rand(len(pairs)).astype("float32")

# Mock the imports before importing retriever
import rag.embeddings
rag.embeddings.model = MockSentenceTransformer()

# Mock the imports in retriever.py
import unittest.mock as mock

# Create mocks for the imports in retriever.py
with mock.patch('sentence_transformers.CrossEncoder', MockCrossEncoder):
    with mock.patch('faiss.IndexFlatIP', mock.Mock()):
        with mock.patch('numpy', mock.Mock()):
            # Now import the retriever
            from rag.retriever import Retriever

def test_basic_functionality():
    """Test basic retriever functionality with context expansion."""
    
    # Create test chunks
    chunks = [
        {"text": "First chunk of document.", "page": 1, "source": "test.pdf"},
        {"text": "Second chunk with important info.", "page": 1, "source": "test.pdf"},
        {"text": "Third chunk continues the discussion.", "page": 2, "source": "test.pdf"},
        {"text": "Fourth chunk has limitations section.", "page": 2, "source": "test.pdf"},
        {"text": "Fifth chunk concludes the document.", "page": 3, "source": "test.pdf"},
    ]
    
    # Create retriever
    retriever = Retriever(chunks)
    
    # Mock the vector store to return predictable results
    retriever.vector_store = mock.Mock()
    retriever.vector_store.search.return_value = (
        [0.9, 0.8, 0.7, 0.6],  # scores
        [2, 1, 3, 0]  # indices (selecting chunk 2, 1, 3)
    )
    
    # Test search without expansion
    print("Testing search WITHOUT context expansion:")
    results_no_expand = retriever.search("test query", top_k=2, expand_context=False)
    print(f"Number of results: {len(results_no_expand)}")
    
    # Test search with expansion
    print("\nTesting search WITH context expansion:")
    results_expand = retriever.search("test query", top_k=2, expand_context=True)
    print(f"Number of results: {len(results_expand)}")
    
    # Test helper methods directly
    print("\nTesting helper methods:")
    
    # Test _find_chunk_index
    candidate = {
        "chunk": "Third chunk continues the discussion.",
        "page": 2,
        "source": "test.pdf"
    }
    index = retriever._find_chunk_index(candidate)
    print(f"Found chunk at index: {index}")
    
    # Test _get_neighboring_chunks
    if index != -1:
        neighbors = retriever._get_neighboring_chunks(index, window=1)
        print(f"Number of neighbors: {len(neighbors)}")
        print("Neighbors:")
        for i, neighbor in enumerate(neighbors):
            print(f"  {i+1}. {neighbor['text'][:50]}...")
    
    # Test _expand_context
    test_candidates = [{
        "chunk": "Third chunk continues the discussion.",
        "page": 2,
        "source": "test.pdf",
        "score": 0.9,
        "retrieval_score": 0.8
    }]
    expanded = retriever._expand_context(test_candidates, window=1)
    print(f"\nExpanded context has {len(expanded)} chunks")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_basic_functionality()