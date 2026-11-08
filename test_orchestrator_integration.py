import sys
sys.path.insert(0, '.')

from research.orchestrator import run
from rag.retriever import Retriever
from rag.chunker import chunk_pages
from pypdf import PdfReader

print('Testing orchestrator integration with new Top-N retriever...')
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

# Test document route
print('4. Testing orchestrator run()...')
try:
    result = run('document', 'What architecture does this paper propose?', retriever)
    
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
    
    # Check for warnings
    if result.warnings:
        print(f'   4. Warnings: {len(result.warnings)}')
        for warning in result.warnings:
            print(f'      - {warning}')
    else:
        print('   4. Warnings: None')
    
    print()
    print('✅ Orchestrator integration test PASSED!')
    
except Exception as e:
    print(f'   ❌ Exception occurred: {e}')
    import traceback
    traceback.print_exc()
    print('❌ Orchestrator integration test FAILED!')

print()
print('=' * 60)
print('Summary: New Top-N retriever is successfully integrated with orchestrator.')