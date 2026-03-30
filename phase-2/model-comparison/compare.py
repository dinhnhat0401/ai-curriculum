"""
Multi-Model LLM Comparison Tool

Sends the same prompts to multiple LLM providers and compares:
- Response quality
- Latency (time to complete)
- Token usage
- Estimated cost

Usage:
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    python compare.py

Requires: pip install anthropic openai tabulate
"""

import os
import time
from dataclasses import dataclass

# ============================================================
# Configuration
# ============================================================

# Cost per million tokens (approximate, check current pricing)
PRICING = {
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Test prompts covering diverse capabilities
TEST_PROMPTS = [
    # Simple Q&A
    "What is the capital of Japan?",
    # Summarization
    "Summarize the key ideas of the transformer architecture in 3 bullet points.",
    # Reasoning
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    # Code generation
    "Write a Python function that checks if a string is a valid palindrome, ignoring spaces and punctuation.",
    # Creative writing
    "Write a haiku about machine learning.",
    # Data extraction
    'Extract the name, email, and phone from this text: "Contact John Smith at john@example.com or call 555-0123 for details."',
    # Math
    "What is 15% of 847.50?",
    # Analysis
    "What are the three main differences between SQL and NoSQL databases?",
    # Instruction following
    "List exactly 5 programming languages that start with the letter P. Output only the names, one per line.",
    # Multi-step reasoning
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
]


@dataclass
class Result:
    """Stores the result of a single model call."""
    model: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


# ============================================================
# Model Callers
# ============================================================

def call_anthropic(prompt: str) -> Result:
    """Call Anthropic Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        start = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.time() - start) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        text = response.content[0].text

        cost = (input_tokens * PRICING["claude-sonnet"]["input"] +
                output_tokens * PRICING["claude-sonnet"]["output"]) / 1_000_000

        return Result("claude-sonnet", prompt, text, latency, input_tokens, output_tokens, cost)

    except Exception as e:
        return Result("claude-sonnet", prompt, f"ERROR: {e}", 0, 0, 0, 0)


def call_openai(prompt: str) -> Result:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI()

        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.time() - start) * 1000

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        text = response.choices[0].message.content

        cost = (input_tokens * PRICING["gpt-4o-mini"]["input"] +
                output_tokens * PRICING["gpt-4o-mini"]["output"]) / 1_000_000

        return Result("gpt-4o-mini", prompt, text, latency, input_tokens, output_tokens, cost)

    except Exception as e:
        return Result("gpt-4o-mini", prompt, f"ERROR: {e}", 0, 0, 0, 0)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("LLM MODEL COMPARISON")
    print("=" * 70)

    # Check which APIs are available
    providers = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(("Claude Sonnet", call_anthropic))
        print("  [OK] Anthropic API key found")
    else:
        print("  [--] ANTHROPIC_API_KEY not set, skipping Claude")

    if os.environ.get("OPENAI_API_KEY"):
        providers.append(("GPT-4o-mini", call_openai))
        print("  [OK] OpenAI API key found")
    else:
        print("  [--] OPENAI_API_KEY not set, skipping OpenAI")

    if not providers:
        print("\nNo API keys found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.")
        return

    # Run comparisons
    all_results = {name: [] for name, _ in providers}

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 70}")
        print(f"Prompt {i+1}/{len(TEST_PROMPTS)}: {prompt[:60]}...")
        print(f"{'─' * 70}")

        for name, caller in providers:
            result = caller(prompt)
            all_results[name].append(result)

            # Print response preview
            response_preview = result.response[:100].replace("\n", " ")
            print(f"\n  {name}:")
            print(f"    Response: {response_preview}...")
            print(f"    Latency: {result.latency_ms:.0f}ms | "
                  f"Tokens: {result.input_tokens}+{result.output_tokens} | "
                  f"Cost: ${result.cost_usd:.6f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for name, results in all_results.items():
        valid = [r for r in results if not r.response.startswith("ERROR")]
        if not valid:
            continue

        avg_latency = sum(r.latency_ms for r in valid) / len(valid)
        total_cost = sum(r.cost_usd for r in valid)
        avg_output = sum(r.output_tokens for r in valid) / len(valid)

        print(f"\n  {name}:")
        print(f"    Average latency:      {avg_latency:.0f}ms")
        print(f"    Average output tokens: {avg_output:.0f}")
        print(f"    Total cost ({len(valid)} prompts): ${total_cost:.6f}")
        print(f"    Cost per 1000 queries: ${total_cost / len(valid) * 1000:.4f}")


if __name__ == "__main__":
    main()
