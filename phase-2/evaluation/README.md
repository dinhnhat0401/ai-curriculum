# Evaluation Pipeline: How to Know If It Works

**Time: ~4 hours | Files: `eval_pipeline.py`, `metrics.py`**

---

> "The #1 question from enterprise clients: How do I know it works?"
> This module IS the answer.

---

## Why Evaluation is Everything

Without evaluation, you're guessing. With evaluation, you're engineering.

Evaluation drives every decision:
- Which model to use? **Evaluate.**
- Which chunk size for RAG? **Evaluate.**
- Is the new prompt better? **Evaluate.**
- Is the system ready for production? **Evaluate.**

## Types of Evaluation

### Automated Metrics

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **Exact Match** | Does output exactly match expected? | Classification, extraction |
| **F1 (token-level)** | Overlap between output and expected tokens | Q&A, summarization |
| **BLEU** | N-gram overlap (precision-focused) | Translation |
| **ROUGE-L** | Longest common subsequence | Summarization |
| **Cosine Similarity** | Semantic similarity via embeddings | Any text comparison |

### LLM-as-Judge

Use a strong LLM (Claude Opus, GPT-4) to evaluate a weaker LLM's output:

```
    Judge prompt:
    "Rate the following answer on a scale of 1-5 for:
     - Faithfulness: Is it grounded in the provided context?
     - Relevance: Does it answer the actual question?
     - Completeness: Does it cover all important points?

     Question: {question}
     Context: {context}
     Answer: {answer}

     Respond as JSON: {faithfulness: N, relevance: N, completeness: N, reasoning: '...'}"
```

LLM-as-judge is surprisingly effective and much cheaper than human evaluation.

### RAG-Specific Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Precision@K** | Of retrieved chunks, how many are relevant? |
| **Recall@K** | Of all relevant chunks, how many were retrieved? |
| **MRR** | How high is the first relevant result ranked? |
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Correctness** | Is the final answer right? |

### Agent-Specific Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Task Completion Rate** | Does the agent finish the task? |
| **Tool Selection Accuracy** | Does it pick the right tools? |
| **Steps to Completion** | How efficient is the agent? |
| **Error Recovery Rate** | Does it recover from tool failures? |

## Building an Evaluation Dataset

Minimum requirements:
- **50 examples** for meaningful results
- **Stratified** by difficulty (easy, medium, hard) and type
- **Include edge cases** and adversarial examples
- **Golden answers** written by domain experts
- **Version controlled** -- your eval set is as important as your code

## Files in This Module

| File | What It Does |
|------|-------------|
| `metrics.py` | Reusable metric functions (exact match, F1, BLEU, ROUGE, cosine sim) |
| `eval_pipeline.py` | Complete evaluation framework with LLM-as-judge |

## Exercises

1. Run the eval pipeline on a simple Claude API wrapper. What scores do you get?
2. Change the prompt and re-evaluate. Did scores improve?
3. Test your RAG system from `rag-scratch/` using this eval pipeline.
4. Add a new metric: response length (are answers too long or too short?).
5. Create an evaluation dataset for YOUR specific use case.
6. Compare two models on the same eval set. Which wins? Is the difference statistically significant?
