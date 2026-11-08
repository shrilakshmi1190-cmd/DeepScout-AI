import sys
sys.path.insert(0, '.')

# Import the test class
from rag.test_context_expansion import TestContextWindowExpansion

# Create test instance
test = TestContextWindowExpansion()
test.setUp()

print("Test data self.chunks_with_multiple_sources:")
for i, chunk in enumerate(test.chunks_with_multiple_sources):
    print(f"  [{i}] {chunk}")

print("\nCreating retriever...")
retriever = test.chunks_with_multiple_sources  # Just for inspection

# Actually we need to create a Retriever but skip the embedding initialization
# Let me check what Retriever.__init__ does
import rag.retriever

# Create a simple mock retriever
class DebugRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        
    def _find_chunk_index(self, candidate):
        print(f'DEBUG _find_chunk_index called with candidate: {candidate}')
        print(f'  candidate.get("chunk"): {repr(candidate.get("chunk"))}')
        print(f'  candidate.get("text"): {repr(candidate.get("text"))}')
        candidate_text = candidate.get('chunk') or candidate.get('text')
        print(f'  candidate_text: {repr(candidate_text)}')
        print(f'  candidate.get("page"): {repr(candidate.get("page"))} (type: {type(candidate.get("page"))})')
        print(f'  candidate.get("source"): {repr(candidate.get("source"))} (type: {type(candidate.get("source"))})')
        
        for i, chunk in enumerate(self.chunks):
            print(f'  Checking chunk {i}: {chunk}')
            if not isinstance(chunk, dict):
                if chunk == candidate_text:
                    print(f'    MATCH at index {i} (non-dict)')
                    return i
                continue
                
            print(f'    chunk.get("text"): {repr(chunk.get("text"))}')
            print(f'    chunk.get("page"): {repr(chunk.get("page"))} (type: {type(chunk.get("page"))})')
            print(f'    chunk.get("source"): {repr(chunk.get("source"))} (type: {type(chunk.get("source"))})')
            
            text_match = chunk.get('text') == candidate_text
            page_match = chunk.get('page') == candidate.get('page')
            source_match = chunk.get('source') == candidate.get('source')
            
            print(f'    text_match: {text_match}')
            print(f'    page_match: {page_match}')
            print(f'    source_match: {source_match}')
            
            if text_match and page_match and source_match:
                print(f'    MATCH at index {i}')
                return i
        
        print('  NO MATCH found, returning -1')
        return -1

print("\nCreating DebugRetriever...")
debug_retriever = DebugRetriever(test.chunks_with_multiple_sources)

print("\nCalling _find_chunk_index with central_chunk...")
central_chunk = {"text": "Doc A: Second chunk", "page": 1, "source": "doc_a.pdf"}
central_index = debug_retriever._find_chunk_index(central_chunk)

print(f"\nReturned index: {central_index}")