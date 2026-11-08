from rag.context_builder import build_context
from llm.mock_llm import generate_answer


question = "Which dataset did the researchers use?"

retrieved_results = [
    {
        "chunk": "The model was evaluated using the CIFAR-10 dataset.",
        "score": 0.91
    },
    {
        "chunk": "The dataset contains 60,000 color images across 10 classes.",
        "score": 0.84
    }
]


prompt = build_context(
    question,
    retrieved_results
)

answer = generate_answer(prompt)

print("QUESTION:")
print(question)

print("\nGENERATED ANSWER:")
print(answer)