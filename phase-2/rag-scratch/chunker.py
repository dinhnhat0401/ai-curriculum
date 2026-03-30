"""
Document Chunking Strategies

Three approaches to splitting documents into chunks for RAG:
1. FixedSizeChunker: split every N characters with overlap
2. SentenceChunker: split on sentence boundaries
3. RecursiveChunker: try paragraphs, then sentences, then characters

Usage:
    python chunker.py  # demonstrates all three on sample text
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A piece of a document with metadata."""
    text: str
    chunk_id: int
    start: int       # character offset in original document
    end: int         # character offset end
    metadata: dict   # additional info (source file, etc.)


class FixedSizeChunker:
    """Split text into fixed-size chunks with optional overlap.

    The simplest strategy. Predictable chunk sizes, but may cut mid-sentence.

    Args:
        chunk_size: target size in characters
        overlap: number of overlapping characters between consecutive chunks
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:  # skip empty chunks
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start=start,
                    end=min(end, len(text)),
                    metadata=metadata,
                ))
                chunk_id += 1

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.overlap

        return chunks


class SentenceChunker:
    """Split text on sentence boundaries, grouping into target-size chunks.

    Respects sentence boundaries so meaning isn't cut mid-thought.
    Chunks may be slightly larger or smaller than target size.

    Args:
        target_size: approximate target chunk size in characters
    """

    def __init__(self, target_size: int = 500):
        self.target_size = target_size

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on period, question mark, exclamation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        chunks = []
        current_sentences = []
        current_length = 0
        chunk_id = 0
        start = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.target_size and current_sentences:
                # Current chunk is full, save it
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start=start,
                    end=start + len(chunk_text),
                    metadata=metadata,
                ))
                chunk_id += 1
                start += len(chunk_text) + 1
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += len(sentence)

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start=start,
                end=start + len(chunk_text),
                metadata=metadata,
            ))

        return chunks


class RecursiveChunker:
    """Recursively split text: try paragraphs first, then sentences, then characters.

    This is the strategy LangChain's RecursiveCharacterTextSplitter uses.
    It produces the most semantically meaningful chunks.

    Args:
        chunk_size: target chunk size in characters
        overlap: overlap between chunks
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Separators in order of preference
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the best available separator."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Last resort: split by characters
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.overlap)]

        splits = text.split(sep)
        result = []
        current = ""

        for piece in splits:
            test = current + sep + piece if current else piece

            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    result.append(current)
                # If single piece is too large, split it recursively
                if len(piece) > self.chunk_size:
                    result.extend(self._split_text(piece, remaining_seps))
                else:
                    current = piece

        if current:
            result.append(current)

        return result

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        pieces = self._split_text(text, self.separators)

        chunks = []
        offset = 0
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(
                    text=piece,
                    chunk_id=i,
                    start=offset,
                    end=offset + len(piece),
                    metadata=metadata,
                ))
            offset += len(piece) + 1

        return chunks


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    sample_text = """
Artificial intelligence has transformed many industries over the past decade. Machine learning models can now process natural language, generate images, and make complex decisions.

One of the most significant developments is the transformer architecture, introduced in the 2017 paper "Attention Is All You Need." This architecture forms the basis of modern language models like GPT and Claude.

Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by providing them with relevant context from external documents. Instead of relying solely on the model's training data, RAG systems retrieve relevant information at query time. This approach reduces hallucination and allows the model to access up-to-date information.

The RAG pipeline consists of several steps: document loading, chunking, embedding, storage in a vector database, retrieval based on query similarity, and generation using the retrieved context. Each step presents trade-offs between quality, speed, and cost.

Chunking strategy is particularly important. Chunks that are too large may contain irrelevant information, diluting the signal. Chunks that are too small may lose important context. Finding the right balance requires experimentation and evaluation on your specific use case.
    """.strip()

    print("=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Document length: {len(sample_text)} characters\n")

    for name, chunker in [
        ("Fixed-Size (size=200, overlap=50)", FixedSizeChunker(200, 50)),
        ("Sentence-Based (target=200)", SentenceChunker(200)),
        ("Recursive (size=200, overlap=50)", RecursiveChunker(200, 50)),
    ]:
        chunks = chunker.chunk(sample_text, {"source": "demo"})
        print(f"\n{'─' * 60}")
        print(f"Strategy: {name}")
        print(f"Number of chunks: {len(chunks)}")
        print(f"{'─' * 60}")
        for chunk in chunks:
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  [{chunk.chunk_id}] ({len(chunk.text):3d} chars) {preview}...")
