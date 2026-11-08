from rag.chunker import chunk_text
from rag.retriever import Retriever
from rag.context_builder import build_context
from llm.local_client import generate_answer


text = """
The researchers developed a new machine learning model.
The model was evaluated using the CIFAR-10 dataset.
The dataset contains 60,000 color images across 10 classes.
The researchers used 50,000 images for training and 10,000 images
for testing.
The proposed model achieved an accuracy of 94.2 percent.
"""


# 1. Split document into chunks
chunks = chunk_text(text)

# 2. Create retriever
retriever = Retriever(chunks)

# 3. Ask a question
question = "Which dataset did the researchers use?"

# 4. Retrieve relevant chunks
results = retriever.search(question, top_k=3)

# 5. Build LLM context
prompt = build_context(question, results)

# 6. Generate answer using local LLM
answer = generate_answer(prompt)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nSOURCES USED:")
for i, result in enumerate(results):
    print(f"\nSource {i + 1} (score: {result['score']:.3f})")
    print(result["chunk"])