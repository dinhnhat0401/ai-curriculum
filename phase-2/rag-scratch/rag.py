"""
Complete RAG Pipeline

Wires together all components: chunker + embedder + vector store + LLM.

Usage:
    python rag.py                       # index sample docs and run demo queries
    python rag.py --query "your question here"

Requires:
    ANTHROPIC_API_KEY or OPENAI_API_KEY for generation
    OPENAI_API_KEY for embeddings (falls back to simple embedder without it)
"""

import os
import argparse
from chunker import RecursiveChunker
from embeddings import get_embedder
from vector_store import SimpleVectorStore


class RAGPipeline:
    """Complete RAG pipeline: index documents, then answer questions.

    Components:
        chunker:      splits documents into searchable chunks
        embedder:     converts text to vector embeddings
        vector_store: stores and searches embeddings
        llm:          generates answers from retrieved context
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, top_k: int = 3):
        self.chunker = RecursiveChunker(chunk_size, chunk_overlap)
        self.embedder = get_embedder()
        self.store = SimpleVectorStore()
        self.top_k = top_k

    def index_directory(self, directory: str):
        """Load and index all .txt files from a directory."""
        all_chunks = []

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self.chunker.chunk(text, metadata={"source": filename})
            all_chunks.extend(chunks)
            print(f"  Indexed {filename}: {len(chunks)} chunks")

        if not all_chunks:
            print("No documents found!")
            return

        # Embed all chunks
        texts = [c.text for c in all_chunks]
        metadata = [{"source": c.metadata.get("source", ""), "chunk_id": c.chunk_id} for c in all_chunks]

        print(f"\n  Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        # Store in vector store
        self.store.add(texts, embeddings, metadata)
        print(f"  Stored {len(self.store)} chunks in vector store")

    def query(self, question: str) -> str:
        """Answer a question using RAG.

        1. Embed the question
        2. Retrieve top-k similar chunks
        3. Generate answer with LLM using retrieved context
        """
        # Step 1: Embed the question
        query_embedding = self.embedder.embed_one(question)

        # Step 2: Retrieve relevant chunks
        results = self.store.search(query_embedding, top_k=self.top_k)

        if not results:
            return "No relevant documents found."

        # Step 3: Build the prompt with retrieved context
        context_parts = []
        for i, result in enumerate(results):
            source = result.metadata.get("source", "unknown")
            context_parts.append(
                f"[Source: {source} | Relevance: {result.score:.2f}]\n{result.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer, say "I don't have enough information to answer that question."
Always cite which source document(s) you used.

Context:
{context}

Question: {question}

Answer:"""

        # Step 4: Generate answer
        answer = self._generate(prompt)

        # Include sources for transparency
        sources = list(set(r.metadata.get("source", "") for r in results))
        return f"{answer}\n\n---\nSources: {', '.join(sources)}"

    def _generate(self, prompt: str) -> str:
        """Call an LLM to generate the answer."""
        # Try Anthropic first, then OpenAI, then return the raw context
        if os.environ.get("ANTHROPIC_API_KEY"):
            return self._generate_anthropic(prompt)
        elif os.environ.get("OPENAI_API_KEY"):
            return self._generate_openai(prompt)
        else:
            return "(No LLM API key found -- showing raw retrieval)\n\n" + prompt

    def _generate_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--query", type=str, help="Ask a question")
    parser.add_argument("--docs", type=str, default=None, help="Path to documents directory")
    args = parser.parse_args()

    # Determine docs directory
    docs_dir = args.docs or os.path.join(os.path.dirname(__file__), "sample_docs")

    print("=" * 60)
    print("RAG PIPELINE")
    print("=" * 60)

    # Initialize and index
    rag = RAGPipeline(chunk_size=500, chunk_overlap=100, top_k=3)
    print(f"\nIndexing documents from: {docs_dir}")
    rag.index_directory(docs_dir)

    if args.query:
        # Single query mode
        print(f"\nQuestion: {args.query}")
        print(f"\n{rag.query(args.query)}")
    else:
        # Demo mode with sample questions
        demo_questions = [
            "What is the company's remote work policy?",
            "How do I install the product?",
            "What are the system requirements?",
            "Who do I contact for HR issues?",
            "What programming languages are supported?",
        ]

        for q in demo_questions:
            print(f"\n{'─' * 60}")
            print(f"Q: {q}")
            print(f"{'─' * 60}")
            answer = rag.query(q)
            print(f"A: {answer}")
