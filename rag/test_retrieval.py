import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import chunk_text
from rag.retriever import Retriever


text = """
The researchers developed a new machine learning model.
The model was evaluated using the CIFAR-10 dataset.
The dataset contains 60,000 color images across 10 classes.
The researchers used 50,000 images for training and 10,000
images for testing.
The proposed model achieved an accuracy of 94.2 percent.
"""


chunks = chunk_text(
    text,
    chunk_size=200,
    overlap=50
)

retriever = Retriever(chunks)

query = "Which dataset did the researchers use?"

print("Testing without context expansion:")
results_no_expansion = retriever.search(query, top_k=3, expand_context=False)

for i, result in enumerate(results_no_expansion):
    print(f"\nResult {i + 1}")
    print("Similarity:", result.get("score", "N/A"))
    print("Chunk:", result["chunk"])

print("\n" + "="*50)
print("Testing WITH context expansion:")
results_with_expansion = retriever.search(query, top_k=3, expand_context=True)

print(f"Total chunks retrieved: {len(results_with_expansion)}")
for i, result in enumerate(results_with_expansion):
    print(f"\nResult {i + 1}")
    print("Similarity:", result.get("score", "N/A"))
    print("Chunk:", result["chunk"])