"""
Cost Optimization Strategies for AI Systems

Demonstrates practical cost reduction techniques:
1. Model routing (cheap model for easy queries, expensive for hard)
2. Semantic caching (skip LLM call for similar queries)
3. Cost tracking (per-request and aggregate)

Usage:
    python cost_optimizer.py
"""

import time
import numpy as np
from dataclasses import dataclass


# ============================================================
# Cost Tracker
# ============================================================

@dataclass
class RequestCost:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


class CostTracker:
    """Track costs across all requests."""

    PRICING = {
        "haiku": {"input": 0.25, "output": 1.25},
        "sonnet": {"input": 3.00, "output": 15.00},
        "opus": {"input": 15.00, "output": 75.00},
    }

    def __init__(self):
        self.requests: list[RequestCost] = []

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = self.PRICING[model]
        return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

    def add(self, model: str, input_tokens: int, output_tokens: int, cached: bool = False):
        cost = 0.0 if cached else self.estimate_cost(model, input_tokens, output_tokens)
        self.requests.append(RequestCost(model, input_tokens, output_tokens, cost, cached))

    def report(self):
        total = sum(r.cost_usd for r in self.requests)
        cached = sum(1 for r in self.requests if r.cached)
        by_model = {}
        for r in self.requests:
            by_model.setdefault(r.model, []).append(r)

        print(f"\n  Total requests:  {len(self.requests)}")
        print(f"  Cache hits:      {cached}")
        print(f"  Total cost:      ${total:.6f}")
        print(f"  Avg cost/req:    ${total/max(len(self.requests),1):.6f}")
        print(f"  Cost per 1000:   ${total/max(len(self.requests),1)*1000:.4f}")

        for model, reqs in sorted(by_model.items()):
            model_cost = sum(r.cost_usd for r in reqs)
            print(f"    {model}: {len(reqs)} requests, ${model_cost:.6f}")


# ============================================================
# Model Router
# ============================================================

class ModelRouter:
    """Route queries to the cheapest adequate model.

    Strategy: classify query complexity, then route:
    - Simple (factual, short answers) → Haiku
    - Medium (analysis, reasoning) → Sonnet
    - Complex (multi-step, code gen) → Opus
    """

    COMPLEXITY_SIGNALS = {
        "simple": ["what is", "who is", "when was", "list", "name", "how many",
                    "true or false", "yes or no", "define"],
        "complex": ["analyze", "compare and contrast", "write code", "design",
                    "explain in detail", "step by step", "optimize",
                    "debug", "refactor", "architect"],
    }

    def route(self, query: str) -> str:
        """Classify query complexity and return model name."""
        query_lower = query.lower()

        # Check for complex signals
        for signal in self.COMPLEXITY_SIGNALS["complex"]:
            if signal in query_lower:
                return "opus"

        # Check for simple signals
        for signal in self.COMPLEXITY_SIGNALS["simple"]:
            if signal in query_lower:
                return "haiku"

        # Default to middle tier
        return "sonnet"


# ============================================================
# Semantic Cache
# ============================================================

class SemanticCache:
    """Cache responses for semantically similar queries.

    Uses simple word overlap for similarity (in production, use embeddings).
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.cache: list[tuple[str, str]] = []  # (query, response) pairs
        self.threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def _similarity(self, q1: str, q2: str) -> float:
        """Simple word-overlap similarity (replace with embedding cosine in production)."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def get(self, query: str) -> str | None:
        """Check if a similar query is in the cache."""
        for cached_query, cached_response in self.cache:
            if self._similarity(query, cached_query) >= self.threshold:
                self.hits += 1
                return cached_response
        self.misses += 1
        return None

    def put(self, query: str, response: str):
        """Add a query-response pair to the cache."""
        self.cache.append((query, response))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{100*self.hits/total:.1f}%" if total > 0 else "N/A",
        }


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COST OPTIMIZATION DEMO")
    print("=" * 60)

    # Sample queries
    queries = [
        "What is machine learning?",
        "Who invented the transformer?",
        "Compare and contrast RAG vs fine-tuning for enterprise use cases",
        "How many layers does GPT-2 have?",
        "Write code to implement a binary search tree in Python",
        "What is the capital of France?",
        "Analyze the trade-offs between model size and inference latency",
        "Define backpropagation",
        "What is machine learning?",  # duplicate (cache hit!)
        "What is ML?",  # similar (potential cache hit)
        "List the top 5 programming languages",
        "Design a production RAG system architecture for a legal firm",
        "True or false: GPT-4 uses a transformer architecture",
        "Step by step, explain how attention works",
        "What is the capital of Japan?",
    ]

    # --- Strategy 1: No optimization (all Sonnet) ---
    print("\n--- Strategy 1: No Optimization (all Sonnet) ---")
    tracker_baseline = CostTracker()
    for q in queries:
        # Simulate token counts
        input_tokens = len(q.split()) * 3 + 100  # rough estimate
        output_tokens = 150
        tracker_baseline.add("sonnet", input_tokens, output_tokens)
    tracker_baseline.report()

    # --- Strategy 2: Model Routing ---
    print("\n--- Strategy 2: Model Routing ---")
    router = ModelRouter()
    tracker_routed = CostTracker()
    for q in queries:
        model = router.route(q)
        input_tokens = len(q.split()) * 3 + 100
        output_tokens = 150
        tracker_routed.add(model, input_tokens, output_tokens)
        print(f"  [{model:6s}] {q[:50]}...")
    tracker_routed.report()

    # --- Strategy 3: Model Routing + Semantic Cache ---
    print("\n--- Strategy 3: Routing + Semantic Cache ---")
    cache = SemanticCache(similarity_threshold=0.7)
    tracker_cached = CostTracker()
    for q in queries:
        cached = cache.get(q)
        if cached:
            tracker_cached.add("haiku", 0, 0, cached=True)
            print(f"  [CACHED] {q[:50]}...")
        else:
            model = router.route(q)
            input_tokens = len(q.split()) * 3 + 100
            output_tokens = 150
            tracker_cached.add(model, input_tokens, output_tokens)
            cache.put(q, f"(response to: {q})")
            print(f"  [{model:6s}] {q[:50]}...")
    tracker_cached.report()
    print(f"  Cache stats: {cache.stats()}")

    # --- Summary ---
    baseline_cost = sum(r.cost_usd for r in tracker_baseline.requests)
    routed_cost = sum(r.cost_usd for r in tracker_routed.requests)
    cached_cost = sum(r.cost_usd for r in tracker_cached.requests)

    print(f"\n{'='*60}")
    print("SAVINGS SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline (all Sonnet):  ${baseline_cost:.6f}")
    print(f"  Model routing:          ${routed_cost:.6f} ({100*(1-routed_cost/baseline_cost):.0f}% savings)")
    print(f"  Routing + caching:      ${cached_cost:.6f} ({100*(1-cached_cost/baseline_cost):.0f}% savings)")
