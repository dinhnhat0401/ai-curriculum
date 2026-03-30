# Model Comparison: The LLM Landscape

**Time: ~5 hours | Files: `compare.py`, `prompt_engineering.py`**

---

## Why Model Selection Matters

No single model wins at everything. Choosing the right model is one of the most impactful decisions in any AI project.

```
    Cost vs Quality vs Latency -- pick two.

    ┌─────────────────────────────────────┐
    │          HIGH QUALITY               │
    │                                     │
    │    Claude Opus    GPT-4o            │
    │    (expensive,    (expensive,        │
    │     slow)         moderate)          │
    │                                     │
    │    Claude Sonnet  GPT-4.1-mini     │
    │    (balanced)     (balanced)         │
    │                                     │
    │    Haiku          Llama-3 (local)   │
    │    (cheap, fast)  (free, slow)       │
    │                                     │
    │          LOW QUALITY                │
    └─────────────────────────────────────┘
```

## The LLM Landscape (2024-2025)

### Anthropic (Claude)

| Model | Best For | Cost (per 1M tokens) |
|-------|---------|---------------------|
| **Claude Opus** | Complex reasoning, analysis, code generation | ~$15 input / $75 output |
| **Claude Sonnet** | Best balance of quality and cost | ~$3 input / $15 output |
| **Claude Haiku** | Simple tasks, classification, extraction | ~$0.25 input / $1.25 output |

Strengths: instruction following, long context (200K), safety, structured output. Claude excels at careful analysis and nuanced responses.

### OpenAI (GPT)

| Model | Best For | Cost (per 1M tokens) |
|-------|---------|---------------------|
| **GPT-4o** | General purpose, multimodal | ~$2.50 input / $10 output |
| **GPT-4.1-mini** | Fast, affordable general use | ~$0.40 input / $1.60 output |
| **o3** | Complex reasoning, math | Higher (reasoning tokens) |

Strengths: massive ecosystem, function calling, vision capabilities, wide deployment.

### Open Source

| Model | Parameters | Best For |
|-------|-----------|---------|
| **Llama 3.1** (Meta) | 8B / 70B / 405B | General, commercially usable |
| **Mistral** | 7B / 8x22B | Efficient, good for fine-tuning |
| **Qwen 2.5** | 7B-72B | Multilingual, code |
| **DeepSeek** | Various | Code, reasoning |

When to use open source: data privacy requirements, no API costs, fine-tuning needed, offline operation.

---

## API Patterns Every AI Engineer Must Know

### 1. Basic Completion
```python
# Anthropic
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain RAG in one paragraph."}]
)
```

### 2. Streaming (critical for UX)
```python
with client.messages.stream(model="claude-sonnet-4-20250514", ...) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 3. System Prompts
```python
messages.create(
    system="You are a senior data analyst. Respond with specific numbers.",
    messages=[...]
)
```

### 4. Tool Use / Function Calling
```python
tools = [{"name": "get_weather", "description": "...", "input_schema": {...}}]
response = client.messages.create(tools=tools, ...)
```

### 5. Structured Output (JSON)
```python
# Prompt the model to respond in JSON, or use tool use to force structure
system = "Respond with valid JSON only. Schema: {name: string, score: int}"
```

---

## Prompt Engineering Fundamentals

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **Zero-shot** | Simple, well-defined tasks | "Translate this to French: ..." |
| **Few-shot** | Model needs examples of desired format | "Input: X -> Output: Y. Input: A -> Output: ?" |
| **Chain of thought** | Multi-step reasoning | "Let's think step by step..." |
| **Role prompting** | Specialized expertise needed | "You are an expert tax accountant..." |
| **Structured output** | Need parseable responses | "Respond as JSON: {field1: ..., field2: ...}" |

### The prompt engineering hierarchy:
1. Clear instructions (what, not how)
2. Examples (show, don't tell)
3. Structure (input/output format)
4. Constraints (length, style, format)
5. Context (background information)

---

## Exercises

1. Run `compare.py` with 15 test prompts. Document which model wins for which task types.
2. Run `prompt_engineering.py` and observe how different prompting strategies change output quality.
3. Calculate cost per 1000 queries for each model at your expected input/output lengths.
4. Try the same complex reasoning task with temperature 0.0, 0.5, and 1.0. Document the differences.
5. Implement streaming for all three providers and measure time-to-first-token.
6. Design a "model router" that picks the cheapest model that can handle each query type.

---

## Key Takeaways

1. **No single model wins everything.** Match the model to the task.
2. **Cost scales linearly with tokens.** Shorter prompts + shorter responses = lower costs.
3. **Prompt engineering is the highest-leverage optimization.** Before changing models, change your prompt.
4. **Streaming matters for UX.** Users perceive streaming responses as faster even when total time is the same.
5. **Always benchmark on YOUR data.** Published benchmarks don't predict performance on your specific use case.
