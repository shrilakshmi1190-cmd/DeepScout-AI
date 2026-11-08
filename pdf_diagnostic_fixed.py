import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

# First, let's check if sample.pdf exists and read it
import hashlib
from pypdf import PdfReader
from rag.chunker import chunk_pages
from rag.retriever import Retriever
from rag.embeddings import model

print('=' * 80)
print('RETRIEVAL DIAGNOSTIC FOR sample.pdf')
print('=' * 80)
print()

# Check if sample.pdf exists
pdf_path = 'sample.pdf'
if not os.path.exists(pdf_path):
    print(f'ERROR: {pdf_path} not found')
    exit(1)

print(f'Found PDF: {pdf_path}')
print(f'File size: {os.path.getsize(pdf_path)} bytes')
print()

# Read and process the PDF
with open(pdf_path, 'rb') as f:
    file_bytes = f.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

reader = PdfReader(pdf_path)
pages = []

print(f'PDF has {len(reader.pages)} pages')
print()

# Extract text page-by-page
for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ''
    pages.append({
        'text': page_text,
        'page': page_number,
        'source': 'sample.pdf'
    })
    print(f'Page {page_number}: {len(page_text)} characters')

print()

# Create chunks
chunks = chunk_pages(pages)
print(f'Total chunks created: {len(chunks)}')
if chunks:
    print(f'Chunk structure keys: {list(chunks[0].keys())}')
    print(f'First chunk preview: {chunks[0]["text"][:100]}...')
print()

# Create retriever
retriever = Retriever(chunks)
print(f'Retriever created with {len(retriever.chunks)} chunks')
print()

# Define diagnostic queries
queries = [
    'What are the limitations of the architecture?',
    'How was the Transformer evaluated?'
]

# Create a diagnostic subclass
class DiagnosticRetriever(Retriever):
    def diagnostic_search(self, query, top_k=3):
        # Step 1: Semantic Retrieval
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        candidate_k = max(top_k * 3, 8)
        
        print(f'candidate_k used: {candidate_k} (max({top_k}*3, 8))')
        
        scores, indices = self.vector_store.search(query_embedding, candidate_k)
        
        print(f'Initial FAISS candidates (BEFORE reranking): {len(scores)} candidates')
        print('-' * 60)
        
        candidates = []
        for i, (score, index) in enumerate(zip(scores, indices)):
            if index == -1:
                continue
                
            chunk = self.chunks[index]
            candidates.append({
                'index': index,
                'chunk_preview': chunk['text'][:100] + '...' if len(chunk['text']) > 100 else chunk['text'],
                'page': chunk['page'],
                'source': chunk['source'],
                'retrieval_score': float(score)
            })
            
            print(f'  Candidate {i+1}:')
            print(f'    Index: {index}, Page: {chunk["page"]}, Score: {float(score):.6f}')
            print(f'    Text preview: {chunk["text"][:80]}...')
            print()
        
        # Check for relevant evidence
        print('Relevant evidence check:')
        print('-' * 60)
        
        # Look for keywords in retrieved chunks
        if 'limitations' in query.lower():
            keywords = ['limitation', 'drawback', 'weakness', 'problem', 'issue', 'challenge']
            print(f'  Looking for limitations keywords: {keywords}')
        else:
            keywords = ['evaluated', 'evaluation', 'results', 'BLEU', 'accuracy', 'dataset', 'test']
            print(f'  Looking for evaluation keywords: {keywords}')
        
        found_keywords = []
        relevant_chunks = []
        for cand in candidates:
            chunk_text = self.chunks[cand['index']]['text'].lower()
            for keyword in keywords:
                if keyword in chunk_text:
                    found_keywords.append(keyword)
                    relevant_chunks.append({
                        'page': cand['page'],
                        'score': cand['retrieval_score'],
                        'keyword': keyword
                    })
        
        if found_keywords:
            print(f'  ✓ Found keywords: {set(found_keywords)}')
            for rc in relevant_chunks:
                print(f'    - Page {rc["page"]}: score {rc["score"]:.6f} (keyword: {rc["keyword"]})')
        else:
            print(f'  ✗ No relevant keywords found in initial candidates')
        
        print()
        return candidates

# Run diagnostic for each query
diagnostic_retriever = DiagnosticRetriever(chunks)

for query in queries:
    print('=' * 80)
    print(f'QUERY: "{query}"')
    print('=' * 80)
    print()
    
    candidates = diagnostic_retriever.diagnostic_search(query, top_k=3)
    print(f'Total initial candidates: {len(candidates)}')
    print()

print('=' * 80)
print('DIAGNOSTIC SUMMARY')
print('=' * 80)
print()
print('Key metrics:')
print(f'1. Total pages in PDF: {len(pages)}')
print(f'2. Total chunks created: {len(chunks)}')
print(f'3. candidate_k used: max(3*3, 8) = 9')
print()
print('To diagnose retrieval issues:')
print('1. Check if relevant keywords appear in initial FAISS candidates')
print('2. Check retrieval scores for relevant chunks')
print('3. If relevant chunks have low scores → FAISS retrieval issue')
print('4. If relevant chunks have decent scores but filtered later → reranking/filtering issue')