# RAG from Scratch: Retrieval-Augmented Generation

**Time: ~5 hours | The most important build in the entire curriculum.**

---

> "RAG is 70% of enterprise AI projects."

If you only master one thing from Phase 2, make it RAG. This module builds every component from scratch so you understand what happens at each step.

---

## What is RAG?

LLMs have two critical limitations:
1. **Knowledge cutoff** -- They don't know about events after training
2. **Hallucination** -- They make up plausible-sounding but false information
3. **No access to your data** -- They can't read your company's documents

RAG solves all three by **retrieving relevant information from your documents and injecting it into the prompt**.

```
    The RAG Pipeline
    ================

    INDEXING (one-time):
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │Documents │───>│  Chunk   │───>│  Embed   │───>│  Store   │
    │(PDF,TXT) │    │(split)   │    │(vectorize│    │(vector   │
    └──────────┘    └──────────┘    └──────────┘    │ database)│
                                                     └──────────┘

    QUERYING (every request):
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Question │───>│  Embed   │───>│ Retrieve │───>│ Generate │
    │          │    │(vectorize│    │(find     │    │(LLM +    │
    │          │    │ query)   │    │ similar) │    │ context) │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Deep Dive: Each Component

### 1. Document Loading
Load documents and extract clean text. The quality of your input text directly determines RAG quality.

### 2. Chunking (Where Most RAG Systems Fail)

Documents must be split into smaller pieces because:
- Embedding models have token limits
- Smaller chunks are more precisely retrievable
- LLM context windows are finite

```
    Chunking Strategies:
    ┌────────────────────────────────────────────────┐
    │ Fixed-size: split every N characters            │
    │ + Simple, predictable                           │
    │ - Cuts mid-sentence                             │
    ├────────────────────────────────────────────────┤
    │ Sentence-based: split on sentence boundaries    │
    │ + Preserves meaning                             │
    │ - Variable chunk sizes                          │
    ├────────────────────────────────────────────────┤
    │ Recursive: try paragraph, then sentence, then N │
    │ + Best of both worlds                           │
    │ - More complex                                  │
    └────────────────────────────────────────────────┘

    The overlap question:
    Chunk 1: [.....text here.....]
    Chunk 2:              [.....text here.....]
                  ^^^^^^^^
                  overlap region

    Overlap ensures context isn't lost at boundaries.
    Typical: 10-20% overlap (e.g., 500 char chunks with 100 char overlap)
```

### 3. Embeddings

Convert text chunks into dense vectors. Similar texts produce similar vectors.

```
    "How to reset my password"  →  [0.12, -0.45, 0.78, 0.33, ...]
    "Password reset process"    →  [0.11, -0.42, 0.75, 0.35, ...]  ← very similar!
    "Company holiday schedule"  →  [-0.67, 0.23, 0.11, -0.89, ...] ← very different

    Cosine similarity:
    "reset password" vs "password reset" = 0.95 (high -- same topic)
    "reset password" vs "holiday schedule" = 0.12 (low -- different topics)
```

### 4. Vector Store

A database optimized for finding similar vectors quickly using approximate nearest neighbor (ANN) algorithms.

### 5. Retrieval

Given a query, find the most relevant chunks:
1. Embed the query
2. Search the vector store for nearest neighbors
3. Return top-k chunks with similarity scores

### 6. Generation

Feed the retrieved chunks into the LLM as context:

```
    System: You are a helpful assistant. Answer questions based ONLY on the
    provided context. If the context doesn't contain the answer, say
    "I don't have enough information to answer that."

    Context:
    [Chunk 1: "The password reset process requires... "]
    [Chunk 2: "Users can access the reset portal at... "]
    [Chunk 3: "Two-factor authentication must be... "]

    Question: How do I reset my password?
```

---

## Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Wrong chunks retrieved | Chunks too large, embedding mismatch | Smaller chunks, better embedding model |
| Right chunks, wrong answer | LLM ignores context or hallucinates | Better prompt, "cite your sources" |
| Info spread across chunks | Multi-hop reasoning needed | Larger chunks, recursive retrieval |
| Query vocabulary mismatch | User asks differently than doc was written | Query expansion, hybrid search |
| Context overflow | Too many chunks exceed token limit | Better ranking, truncation strategy |

---

## Files in This Module

| File | What It Does |
|------|-------------|
| `chunker.py` | 3 chunking strategies: fixed-size, sentence, recursive |
| `embeddings.py` | Embedding generation with OpenAI + cosine similarity |
| `vector_store.py` | Simple in-memory vector store with persistence |
| `rag.py` | Complete RAG pipeline wiring everything together |
| `evaluate.py` | Test the RAG system with questions and score results |
| `sample_docs/` | 3 test documents (company policy, product FAQ, tech guide) |

---

## Exercises

1. Index the sample docs and test with 10 questions. How many does it answer correctly?
2. Vary chunk size (200, 500, 1000, 2000 chars) and measure retrieval quality.
3. Try different top-k values (1, 3, 5, 10). When does more context help vs hurt?
4. Add a "I don't know" test: ask questions NOT covered in the docs. Does it refuse correctly?
5. Compare OpenAI embeddings vs sentence-transformers. Quality difference?
6. Add metadata filtering: only retrieve from a specific document.
7. Implement hybrid search: combine embedding similarity with keyword matching.
8. Build a simple citation system: include source document + chunk ID in the response.

---

## Key Takeaways

1. **RAG = retrieve relevant context + inject into prompt.** The idea is simple. The execution has many gotchas.
2. **Chunking is the most important design decision.** Wrong chunk size = wrong results.
3. **Embeddings turn text into searchable vectors.** Same concept as makemore embeddings, scaled up.
4. **Evaluation is non-negotiable.** You must test with real questions to know if your RAG works.
5. **Start simple, then optimize.** Fixed-size chunks + basic retrieval first. Fancy techniques only when needed.
