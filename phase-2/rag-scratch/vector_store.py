"""
Simple In-Memory Vector Store

Stores text chunks with their embeddings and supports similarity search.
Includes persistence (save/load to JSON).

This is a learning implementation. For production, use:
- ChromaDB (easy)
- Qdrant (production-grade)
- pgvector (if you have Postgres)
- Pinecone (managed)
"""

import json
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class SearchResult:
    """A single search result with text, metadata, and similarity score."""
    text: str
    metadata: dict
    score: float


class SimpleVectorStore:
    """In-memory vector store with cosine similarity search.

    Stores documents as (text, embedding, metadata) triples.
    Search finds the most similar documents to a query embedding.
    """

    def __init__(self):
        self.texts: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.metadata: list[dict] = []

    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict] = None):
        """Add documents to the store.

        Args:
            texts: list of text strings
            embeddings: numpy array of shape (len(texts), embedding_dim)
            metadata: optional list of metadata dicts (one per text)
        """
        if metadata is None:
            metadata = [{} for _ in texts]

        assert len(texts) == len(embeddings) == len(metadata), \
            f"Length mismatch: {len(texts)} texts, {len(embeddings)} embeddings, {len(metadata)} metadata"

        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Find the top-k most similar documents to the query.

        Uses cosine similarity (dot product of normalized vectors).
        """
        if not self.embeddings:
            return []

        # Stack all embeddings into a matrix for efficient computation
        all_embeddings = np.array(self.embeddings)

        # Compute cosine similarity with all documents at once
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        # Normalize all docs
        doc_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-10
        all_normalized = all_embeddings / doc_norms

        # Dot product = cosine similarity (both are unit vectors)
        similarities = all_normalized @ query_norm

        # Get top-k indices (sorted by similarity, descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self.texts[idx],
                metadata=self.metadata[idx],
                score=float(similarities[idx]),
            ))

        return results

    def __len__(self):
        return len(self.texts)

    def save(self, path: str):
        """Save the vector store to a JSON file."""
        data = {
            "texts": self.texts,
            "embeddings": [e.tolist() for e in self.embeddings],
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved {len(self.texts)} documents to {path}")

    def load(self, path: str):
        """Load the vector store from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.embeddings = [np.array(e) for e in data["embeddings"]]
        self.metadata = data["metadata"]
        print(f"Loaded {len(self.texts)} documents from {path}")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VECTOR STORE DEMO")
    print("=" * 60)

    store = SimpleVectorStore()

    # Add some documents with fake embeddings (in real use, these come from an embedding model)
    np.random.seed(42)
    sample_texts = [
        "The password reset process takes 24 hours to complete.",
        "Company holidays include New Year's Day, Memorial Day, and Thanksgiving.",
        "Database connections should use connection pooling for efficiency.",
        "To reset your password, visit the account settings page.",
        "The annual company retreat is held every September.",
    ]

    # Fake embeddings for demo (dimension 8)
    fake_embeddings = np.random.randn(len(sample_texts), 8)
    # Make password-related docs have similar embeddings
    fake_embeddings[0] = np.array([1.0, 0.5, 0.2, -0.1, 0.3, 0.1, -0.2, 0.4])
    fake_embeddings[3] = np.array([0.9, 0.6, 0.1, -0.2, 0.4, 0.0, -0.1, 0.5])  # similar

    metadata = [{"source": f"doc_{i}.txt", "chunk_id": i} for i in range(len(sample_texts))]
    store.add(sample_texts, fake_embeddings, metadata)

    print(f"\nStored {len(store)} documents")

    # Search for something related to passwords
    query_embedding = np.array([0.95, 0.55, 0.15, -0.15, 0.35, 0.05, -0.15, 0.45])
    results = store.search(query_embedding, top_k=3)

    print("\nQuery: 'how to reset password' (using pre-computed embedding)")
    print("\nTop 3 results:")
    for i, r in enumerate(results):
        print(f"  {i+1}. (score: {r.score:.3f}) {r.text}")
        print(f"     metadata: {r.metadata}")
