# Fine-tuning: When and How to Customize a Model

**Time: ~4 hours | Files: `prepare_dataset.py`, `finetune_lora.py`**

---

## The Decision Framework

Before fine-tuning, ask yourself:

```
    Need specific KNOWLEDGE?      → RAG (retrieve from documents)
    Need specific BEHAVIOR/STYLE? → Fine-tuning
    Need specific OUTPUT FORMAT?  → Prompt engineering
    Tried prompting and it's not enough? → Fine-tuning
    Have fewer than 100 examples? → Don't fine-tune yet
```

Fine-tuning is powerful but expensive (in time and data). Try prompt engineering and RAG first.

## Types of Fine-tuning

| Method | What It Does | Data Needed | Cost | When to Use |
|--------|-------------|-------------|------|-------------|
| **Full fine-tuning** | Updates ALL parameters | 10K+ examples | Very high (multi-GPU) | Maximum quality, unlimited budget |
| **LoRA** | Trains small adapter matrices, freezes base | 500-5K examples | Low (single GPU) | Best balance of quality and cost |
| **QLoRA** | LoRA + quantized base model | 500-5K examples | Very low (consumer GPU) | Limited hardware |
| **API fine-tuning** | Provider handles everything | 50-500 examples | Moderate (per-token) | Fastest path, no infra needed |

## LoRA: How It Works

```
    Standard fine-tuning:
    W_new = W_original + delta_W         (delta_W has millions of params)

    LoRA:
    W_new = W_original + A @ B           (A and B are tiny matrices)

    If W is 4096 x 4096 (16M params),
    A is 4096 x 8 and B is 8 x 4096     (65K params -- 250x smaller!)

    The rank (r=8) controls capacity.
    Higher rank = more capacity but more params.
```

LoRA works because weight updates during fine-tuning are low-rank -- they don't need the full dimensionality.

## Dataset Preparation (The Most Important Step)

**Quality over quantity.** 500 excellent examples beat 5000 mediocre ones.

Format (chat/instruction style):
```json
{"messages": [
    {"role": "system", "content": "You are a SQL expert."},
    {"role": "user", "content": "How many users signed up last month?"},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"}
]}
```

Common dataset mistakes:
- Inconsistent formatting across examples
- Contradictory examples (different answers for similar inputs)
- Too few diverse examples (model memorizes instead of generalizing)
- No validation split (can't detect overfitting)

## Exercises

1. Run `prepare_dataset.py` to create a training dataset for text-to-SQL
2. Examine the dataset: are examples consistent? diverse? correctly labeled?
3. If you have a GPU: run `finetune_lora.py` and compare base vs fine-tuned output
4. Try OpenAI API fine-tuning with 50 examples -- is it enough?
5. Create a dataset for a different task (classification, summarization, translation)
6. Experiment with LoRA rank: r=4, r=8, r=16, r=32. How does quality change?
