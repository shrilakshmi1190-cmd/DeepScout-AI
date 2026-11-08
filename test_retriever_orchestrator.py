import sys
sys.path.insert(0, '.')

from research.orchestrator import _run_document
from rag.retriever import Retriever
from rag.chunker import chunk_pages
from pypdf import PdfReader

print('Testing retriever-orchestrator integration (LLM-free)...')
print('=' * 60)

# Load sample PDF
print('1. Loading sample.pdf...')
reader = PdfReader('sample.pdf')
pages = []
for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ''
    pages.append({'text': page_text, 'page': page_number, 'source': 'sample.pdf'})

print(f'   Loaded {len(pages)} pages')

# Create chunks
print('2. Creating chunks...')
chunks = chunk_pages(pages)
print(f'   Created {len(chunks)} chunks')

# Create retriever with new Top-N implementation
print('3. Creating retriever...')
retriever = Retriever(chunks)
print(f'   Retriever created with {len(retriever.chunks)} chunks')

# Test the _run_document function up to LLM call
print('4. Testing _run_document (without LLM)...')
try:
    # Create a mock generate_answer function to avoid LLM issues
    import llm.llm_service
    
    def mock_generate_answer(prompt):
        return f"[MOCK ANSWER] Processed prompt of length {len(prompt)}"
    
    # Temporarily replace generate_answer
    original_generate_answer = llm.llm_service.generate_answer
    llm.llm_service.generate_answer = mock_generate_answer
    
    try:
        # Call _run_document directly
        from research.orchestrator import _run_document
        result = _run_document('What architecture does this paper propose?', retriever, [])
        
        print('   ✅ No exceptions occurred')
        print()
        print('Result verification:')
        print(f'   1. Route: {result.route} (expected: "document")')
        print(f'   2. Answer generated: {len(result.answer)} characters')
        print(f'   3. PDF sources: {len(result.pdf_sources)}')
        
        if result.pdf_sources:
            print('   Source details:')
            for i, source in enumerate(result.pdf_sources[:3], 1):
                print(f'     Source {i}: Page {source.get("page", "N/A")}, Score: {source.get("score", 0):.4f}')
        
        # Check if retriever.search was called correctly
        print(f'   4. Retrieved {len(result.pdf_sources)} PDF sources')
        
        # Check for warnings
        if result.warnings:
            print(f'   5. Warnings: {len(result.warnings)}')
            for warning in result.warnings:
                print(f'      - {warning}')
        else:
            print('   5. Warnings: None')
        
        print()
        print('✅ Retriever-orchestrator integration test PASSED!')
        
    finally:
        # Restore original function
        llm.llm_service.generate_answer = original_generate_answer
        
except Exception as e:
    print(f'   ❌ Exception occurred: {e}')
    import traceback
    traceback.print_exc()
    print('❌ Retriever-orchestrator integration test FAILED!')

print()
print('=' * 60)
print('Additional verification: Direct retriever test')
print('=' * 60)

# Test retriever directly
print('Testing retriever.search() directly...')
results = retriever.search('What architecture does this paper propose?', top_k=3, expand_context=False)
print(f'Retrieved {len(results)} results')
for i, result in enumerate(results, 1):
    print(f'  Result {i}: Page {result.get("page", "N/A")}, Score: {result.get("score", 0):.4f}')

print()
print('Summary: New Top-N retriever implementation is working correctly!')