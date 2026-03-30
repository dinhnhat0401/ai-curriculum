# RAG with Frameworks: LangChain vs LlamaIndex

**Time: ~3 hours | Files: `langchain_rag.py`, `llamaindex_rag.py`**

---

## Why Use a Framework?

After building RAG from scratch, you know exactly what frameworks hide. Frameworks give you:
- Pre-built document loaders (PDF, web, CSV, databases, etc.)
- Optimized text splitters
- Managed vector store integrations
- Evaluation tools
- Production features (caching, retry, logging)

The trade-off: you gain speed and features, but lose control and debuggability.

## Framework Comparison

| | LangChain | LlamaIndex |
|---|-----------|-----------|
| **Focus** | General LLM application framework | Data framework for LLM apps (RAG-focused) |
| **Strengths** | Huge ecosystem, many integrations, agents | Excellent indexing, document handling, RAG |
| **Weaknesses** | Over-abstracted, frequent API changes | More opinionated, smaller ecosystem |
| **Best for** | Agents, chains, diverse integrations | Document Q&A, knowledge bases, RAG |
| **Learning curve** | Medium | Medium |

## When NOT to Use a Framework

- Simple single-step LLM calls (just use the API directly)
- When you need full control over chunking and retrieval
- When debugging framework abstractions costs more than building from scratch
- When the framework adds 50 dependencies for a 10-line solution

## Decision Matrix

```
    Just calling an API?           → Raw SDK (anthropic, openai)
    Building RAG?                  → LlamaIndex (purpose-built)
    Building agents?               → LangChain or raw tool use
    Need many integrations?        → LangChain (biggest ecosystem)
    Need maximum control?          → Build from scratch
    Prototyping quickly?           → Framework (either one)
    Going to production?           → Understand what's inside first
```

## Your From-Scratch vs Framework

After building both, fill in this comparison:

| Metric | From Scratch | LangChain | LlamaIndex |
|--------|-------------|-----------|-----------|
| Setup time | ___ | ___ | ___ |
| Lines of code | ___ | ___ | ___ |
| Retrieval quality | ___ | ___ | ___ |
| Customizability | ___ | ___ | ___ |
| Debugging ease | ___ | ___ | ___ |

## Exercises

1. Rebuild your Phase 2 RAG with LangChain. What was easier? What was harder?
2. Rebuild again with LlamaIndex. Compare with LangChain.
3. Try a document type your from-scratch RAG can't handle (PDF, web page). Frameworks make this trivial.
4. Use LangChain's RecursiveCharacterTextSplitter. Compare chunks with your chunker.py.
5. Add conversation memory (multi-turn) using LangChain's ConversationBufferMemory.
6. Switch vector stores (e.g., ChromaDB to FAISS) with one line of code. Try that with your from-scratch version.
