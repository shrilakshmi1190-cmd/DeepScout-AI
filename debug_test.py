import sys
sys.path.insert(0, '.')

# First, monkey-patch _find_chunk_index to add debug prints
import rag.retriever

original_find_chunk_index = rag.retriever.Retriever._find_chunk_index

def debug_find_chunk_index(self, candidate):
    print('DEBUG _find_chunk_index called with candidate:', candidate)
    print('  candidate.get("chunk"):', repr(candidate.get('chunk')))
    print('  candidate.get("text"):', repr(candidate.get('text')))
    candidate_text = candidate.get('chunk') or candidate.get('text')
    print('  candidate_text:', repr(candidate_text))
    print('  candidate.get("page"):', repr(candidate.get('page')), '(type:', type(candidate.get('page')), ')')
    print('  candidate.get("source"):', repr(candidate.get('source')), '(type:', type(candidate.get('source')), ')')
    
    for i, chunk in enumerate(self.chunks):
        print('  Checking chunk', i, ':', chunk)
        if not isinstance(chunk, dict):
            if chunk == candidate_text:
                print('    MATCH at index', i, '(non-dict)')
                return i
            continue
            
        print('    chunk.get("text"):', repr(chunk.get('text')))
        print('    chunk.get("page"):', repr(chunk.get('page')), '(type:', type(chunk.get('page')), ')')
        print('    chunk.get("source"):', repr(chunk.get('source')), '(type:', type(chunk.get('source')), ')')
        
        text_match = chunk.get('text') == candidate_text
        page_match = chunk.get('page') == candidate.get('page')
        source_match = chunk.get('source') == candidate.get('source')
        
        print('    text_match:', text_match)
        print('    page_match:', page_match)
        print('    source_match:', source_match)
        
        if text_match and page_match and source_match:
            print('    MATCH at index', i)
            return i
    
    print('  NO MATCH found, returning -1')
    return -1

rag.retriever.Retriever._find_chunk_index = debug_find_chunk_index

# Now run the test
import unittest
import rag.test_context_expansion

loader = unittest.TestLoader()
suite = loader.loadTestsFromName('test_get_neighboring_chunks_same_source_only', module=rag.test_context_expansion)
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
result = runner.run(suite)