from embeddings import create_embeddings


texts = [
    "RAG retrieves relevant information from external sources.",
    "Retrieval augmented generation uses external knowledge.",
    "The weather is sunny today."
]

embeddings = create_embeddings(texts)

print("Number of texts:", len(texts))
print("Embedding shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])