# Phase 2: Production AI Engineering

**Weeks 5-8 | ~29 hours | Build the things clients pay for.**

---

Phase 1 taught you how LLMs work. Phase 2 teaches you how to build with them.

```
    Model Comparison ──> RAG (scratch) ──> RAG (framework) ──> Agents ──> Fine-tuning ──> Evaluation
         │                    │                  │                │            │              │
    Learn the APIs      Build the #1        See what          Build AI     When and how   How to know
    and prompt          enterprise AI       frameworks        that acts,    to customize   if it works
    engineering         application         hide              not just      a model
                        from scratch                          talks
```

## Modules

| Module | What You Build | Why It Matters |
|--------|---------------|----------------|
| **Model Comparison** | Multi-provider LLM benchmark tool | Know which model to use when, and how to call them all |
| **RAG (scratch)** | Full RAG pipeline: chunk, embed, store, retrieve, generate | 70% of enterprise AI is RAG. Build it from scratch first. |
| **RAG (framework)** | Same RAG with LangChain + LlamaIndex | Know what frameworks give you and what they hide |
| **Agent** | AI agent with 4 tools and multi-step reasoning | Agents are the next wave after RAG |
| **Fine-tuning** | LoRA fine-tuning of open-source model | Know when fine-tuning beats prompting or RAG |
| **Evaluation** | Automated eval pipeline with LLM-as-judge | The answer to "how do I know it works?" |

## Prerequisites

- Phase 1 complete
- API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- `pip install anthropic openai chromadb langchain llama-index`
- Optional: Ollama installed for local models

## After This Phase

You can build, evaluate, and explain every major AI application pattern. When a client describes a problem, you know immediately whether it needs RAG, an agent, fine-tuning, or just better prompts -- and you can build any of them.
