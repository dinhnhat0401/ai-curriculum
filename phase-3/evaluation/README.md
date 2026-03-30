# Evaluation at Scale + Cost Optimization

**Time: ~6 hours | Files: `eval_suite.py`, `cost_optimizer.py`**

---

## Cost Optimization Strategies

| Strategy | Effort | Savings | Quality Impact |
|----------|--------|---------|---------------|
| **Prompt caching** | Low | 50-90% on system prompts | None |
| **Model routing** | Medium | 40-70% on easy queries | Minimal if done well |
| **Batch processing** | Low | 50% on non-real-time | None (just slower) |
| **Response length control** | Low | 10-30% | Minimal |
| **Chunk size optimization** | Medium | 20-40% | Varies |
| **Semantic caching** | High | 30-60% on repeat queries | None for exact matches |

### Model Routing

Route queries to the cheapest model that can handle them:

```
    User query ──> Classifier ──> Easy?   ──> Haiku ($0.25/1M)
                                  Medium? ──> Sonnet ($3/1M)
                                  Hard?   ──> Opus ($15/1M)

    Most queries are easy. This alone can cut costs 50%+.
```

### Semantic Caching

Cache responses for semantically similar queries:

```
    Query: "What is the return policy?"        → Generate + cache
    Query: "How do I return a product?"        → Cache hit! (similar enough)
    Query: "What is the weather in Tokyo?"     → Generate + cache (different topic)
```

## Building a Cost Model

```
    Cost per query = (input_tokens * input_price + output_tokens * output_price) / 1M

    Example (Claude Sonnet):
    - Average query: 500 input tokens + 200 output tokens
    - Cost: (500 * $3 + 200 * $15) / 1M = $0.0045 per query
    - 1,000 queries/day = $4.50/day = $135/month

    At 10x scale: $1,350/month
    At 100x scale: $13,500/month
    → This is where cost optimization becomes critical
```

## Exercises

1. Run `cost_optimizer.py` and observe the cost savings from model routing.
2. Implement semantic caching and measure cache hit rate on your test queries.
3. Calculate your RAG system's cost at 100, 1000, and 10000 queries/day.
4. Compare Anthropic prompt caching vs no caching on a system prompt-heavy workflow.
5. Build a cost monitoring dashboard (even a simple print-based one).
6. Determine the break-even point: when does fine-tuning a smaller model save money vs API calls?
