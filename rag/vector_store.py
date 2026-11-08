import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings):
        embeddings = np.asarray(embeddings).astype("float32")
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=3):
        query_embedding = np.asarray(
            [query_embedding]
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        return scores[0], indices[0]