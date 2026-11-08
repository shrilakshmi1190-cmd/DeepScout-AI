from llm.local_client import generate_answer


prompt = """
What is Retrieval-Augmented Generation (RAG)?

Explain it in simple terms.
"""


answer = generate_answer(prompt)

print("\nANSWER:")
print(answer)