from .embeddings import model
from .vector_store import VectorStore

from sentence_transformers import CrossEncoder


class Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        # Extract text from chunks
        texts = [
            chunk["text"] if isinstance(chunk, dict)
            else chunk
            for chunk in chunks
        ]

        if not texts:
            self.vector_store = None
            self.reranker = None
            return

        # Create embeddings
        embeddings = model.encode(
            texts,
            normalize_embeddings=True
        )

        # Create vector store
        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(dimension)

        self.vector_store.add_embeddings(
            embeddings
        )

        # Load reranking model
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def _find_chunk_index(self, candidate):
        """
        Find the index of a candidate chunk in the original chunks list.
        
        Args:
            candidate: A candidate dictionary with "chunk" or "text", "page", "source" keys
            
        Returns:
            Index in self.chunks if found, -1 otherwise
        """
        if not isinstance(candidate, dict):
            return -1
            
        # Candidate may have "chunk" or "text" key with the text content
        candidate_text = candidate.get("chunk") or candidate.get("text")
        if candidate_text is None:
            return -1
            
        for i, chunk in enumerate(self.chunks):
            if not isinstance(chunk, dict):
                # If chunk is not a dict, compare text directly
                if chunk == candidate_text:
                    return i
                continue
                
            # Original chunk has "text" key
            if (chunk.get("text") == candidate_text and
                chunk.get("page") == candidate.get("page") and
                chunk.get("source") == candidate.get("source")):
                return i
        return -1

    def _get_neighboring_chunks(self, chunk_index, window=1):
        """
        Get neighboring chunks around a given chunk index.
        
        Args:
            chunk_index: Index of the central chunk
            window: Number of chunks to include on each side
            
        Returns:
            List of chunk dictionaries including the central and neighboring chunks
        """
        if chunk_index < 0 or chunk_index >= len(self.chunks):
            return []
        
        result = []
        central_chunk = self.chunks[chunk_index]
        
        # Only expand within the same document/source
        source = central_chunk.get("source") if isinstance(central_chunk, dict) else None
        
        # Add previous chunks
        for i in range(chunk_index - window, chunk_index):
            if i >= 0:
                neighbor = self.chunks[i]
                # Check if neighbor is from the same source
                if source is None or (isinstance(neighbor, dict) and neighbor.get("source") == source):
                    result.append(neighbor)
        
        # Add the central chunk
        result.append(central_chunk)
        
        # Add next chunks
        for i in range(chunk_index + 1, chunk_index + window + 1):
            if i < len(self.chunks):
                neighbor = self.chunks[i]
                # Check if neighbor is from the same source
                if source is None or (isinstance(neighbor, dict) and neighbor.get("source") == source):
                    result.append(neighbor)
        
        return result

    def _expand_context(self, selected_results, window=1):
        """
        Expand context around selected relevant chunks.
        
        Args:
            selected_results: List of selected candidate dictionaries
            window: Number of neighboring chunks to include on each side
            
        Returns:
            List of expanded chunk dictionaries with duplicates removed
        """
        if not selected_results or not self.chunks:
            return []
        
        expanded_chunks = []
        seen_chunks = set()
        
        for candidate in selected_results:
            # Find the index of this candidate in the original chunks
            chunk_index = self._find_chunk_index(candidate)
            if chunk_index == -1:
                continue
            
            # Get neighboring chunks
            neighbors = self._get_neighboring_chunks(chunk_index, window)
            
            # Add neighbors, avoiding duplicates
            for neighbor in neighbors:
                # Create a unique identifier for this chunk
                if isinstance(neighbor, dict):
                    chunk_id = (neighbor.get("text", ""), 
                               neighbor.get("page"), 
                               neighbor.get("source"))
                else:
                    chunk_id = (neighbor, None, None)
                
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    
                    # Convert neighbor to candidate format if needed
                    if isinstance(neighbor, dict):
                        expanded_chunks.append({
                            "chunk": neighbor.get("text", ""),
                            "page": neighbor.get("page"),
                            "source": neighbor.get("source"),
                            "score": candidate.get("score", 0.0),  # Use original candidate's score
                            "retrieval_score": candidate.get("retrieval_score", 0.0)
                        })
                    else:
                        expanded_chunks.append({
                            "chunk": neighbor,
                            "page": None,
                            "source": None,
                            "score": candidate.get("score", 0.0),
                            "retrieval_score": candidate.get("retrieval_score", 0.0)
                        })
        
        # Preserve document order (based on original chunk order)
        # We'll sort by chunk index in original chunks to maintain document flow
        expanded_with_indices = []
        for exp_chunk in expanded_chunks:
            # Find index for sorting
            idx = -1
            for i, chunk in enumerate(self.chunks):
                if isinstance(chunk, dict) and isinstance(exp_chunk, dict):
                    if (chunk.get("text") == exp_chunk.get("chunk") and
                        chunk.get("page") == exp_chunk.get("page") and
                        chunk.get("source") == exp_chunk.get("source")):
                        idx = i
                        break
                elif not isinstance(chunk, dict) and not isinstance(exp_chunk, dict):
                    if chunk == exp_chunk.get("chunk"):
                        idx = i
                        break
            expanded_with_indices.append((idx, exp_chunk))
        
        # Sort by original position in document
        expanded_with_indices.sort(key=lambda x: x[0])
        
        # Return just the chunks
        return [chunk for _, chunk in expanded_with_indices]

    def search(self, query, top_k=3, expand_context=True):

        if (
            not self.chunks
            or self.vector_store is None
            or self.reranker is None
        ):
            return []

        # --------------------------------
        # Step 1: Semantic Retrieval
        # --------------------------------

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Retrieve more candidates than we need
        candidate_k = max(top_k * 3, 8)

        scores, indices = self.vector_store.search(
            query_embedding,
            candidate_k
        )


        # --------------------------------
        # Step 2: Build Candidates
        # --------------------------------

        candidates = []

        for score, index in zip(scores, indices):

            if index == -1:
                continue

            chunk = self.chunks[index]

            if isinstance(chunk, dict):

                candidates.append({
                    "chunk": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "retrieval_score": float(score)
                })

            else:

                candidates.append({
                    "chunk": chunk,
                    "page": None,
                    "source": None,
                    "retrieval_score": float(score)
                })


        # --------------------------------
        # Step 3: Re-rank Candidates
        # --------------------------------

        if not candidates:
            return []

        pairs = [
            [query, candidate["chunk"]]
            for candidate in candidates
        ]

        rerank_scores = self.reranker.predict(
            pairs
        )


        for candidate, score in zip(
            candidates,
            rerank_scores
        ):

            candidate["score"] = float(score)


        # --------------------------------
        # Step 4: Sort by Relevance
        # --------------------------------

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        # --------------------------------
        # Step 5: Top-N Filtering (replaces adaptive filtering)
        # --------------------------------

        # Keep top candidate_k candidates after reranking
        # This matches the initial FAISS retrieval count
        candidates = candidates[:candidate_k]


        # --------------------------------
        # Step 6: Remove Duplicate Pages
        # --------------------------------

        final_results = []

        seen_pages = set()

        for candidate in candidates:

            page = candidate["page"]
            source = candidate["source"]

            # Create a unique page identifier
            page_key = (source, page)

            # If this page was already selected,
            # skip this chunk.
            if page_key in seen_pages:
                continue

            final_results.append(candidate)

            seen_pages.add(page_key)

            # Stop once we have enough results
            if len(final_results) == top_k:
                break


        # --------------------------------
        # Step 7: Context Window Expansion
        # --------------------------------

        if expand_context and final_results:
            expanded_results = self._expand_context(final_results, window=1)
            # Limit the total number of results to avoid sending too much context
            # We allow up to top_k * 3 expanded chunks at most
            max_expanded = min(top_k * 3, len(expanded_results))
            final_results = expanded_results[:max_expanded]


        # --------------------------------
        # Step 8: Return Final Results
        # --------------------------------

        return final_results