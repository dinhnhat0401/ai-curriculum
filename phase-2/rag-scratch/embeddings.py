"""
Embedding Module for RAG

Generates vector embeddings from text and computes similarity.

Supports:
- OpenAI embeddings (requires OPENAI_API_KEY)
- Cosine similarity computation (numpy, no API needed)

Usage:
    python embeddings.py  # demo with cosine similarity
"""

import os
import numpy as np
from dataclasses import dataclass


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    cosine_sim(a, b) = (a . b) / (||a|| * ||b||)

    Returns a value between -1 (opposite) and 1 (identical).
    For normalized embeddings, this is equivalent to dot product.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class OpenAIEmbedder:
    """Generate embeddings using OpenAI's API.

    Uses text-embedding-3-small by default (1536 dimensions, cheap, good quality).
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("pip install openai")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns array of shape (len(texts), embedding_dim)."""
        # OpenAI API handles batching internally (max 2048 inputs)
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns 1D array."""
        return self.embed([text])[0]


class SimpleEmbedder:
    """Fallback embedder using basic TF-IDF-like vectors.

    NOT suitable for production -- use OpenAI or sentence-transformers instead.
    This exists so the RAG pipeline can run without API keys for learning.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab = {}

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Create simple bag-of-words embeddings."""
        # Build vocabulary from all texts
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.vocab:
                    # Assign a random but deterministic vector to each word
                    rng = np.random.RandomState(hash(token) % 2**31)
                    self.vocab[token] = rng.randn(self.dim)

        embeddings = []
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                embeddings.append(np.zeros(self.dim))
                continue
            # Average the word vectors
            vecs = [self.vocab.get(t, np.zeros(self.dim)) for t in tokens]
            avg = np.mean(vecs, axis=0)
            # Normalize to unit length
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            embeddings.append(avg)

        return np.array(embeddings)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder():
    """Get the best available embedder."""
    if os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbedder()
    else:
        print("No OPENAI_API_KEY found. Using simple fallback embedder.")
        print("(Set OPENAI_API_KEY for much better results)")
        return SimpleEmbedder()


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING SIMILARITY DEMO")
    print("=" * 60)

    texts = [
        "How do I reset my password?",
        "I forgot my password and need to change it",
        "What is the company holiday schedule?",
        "When are the office closed days this year?",
        "How to configure the database connection",
    ]

    embedder = get_embedder()
    embeddings = embedder.embed(texts)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"(That's {len(texts)} texts, each as a {embeddings.shape[1]}-dimensional vector)\n")

    # Compute all pairwise similarities
    print("Pairwise cosine similarities:")
    print(f"{'':>45}", end="")
    for i in range(len(texts)):
        print(f"  [{i}]", end="")
    print()

    for i, text_i in enumerate(texts):
        label = text_i[:42] + "..." if len(text_i) > 42 else text_i
        print(f"  [{i}] {label:>40}", end="")
        for j in range(len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f" {sim:.2f}", end="")
        print()

    print("\nExpected: texts about passwords should be similar to each other,")
    print("and different from texts about holidays or databases.")
