# NanoGPT: Build a Transformer from Scratch

**Time: ~6 hours | Difficulty: Advanced | File: `gpt.py` | The Phase 1 capstone.**

---

## What You Will Learn

You will build GPT -- the exact same architecture behind ChatGPT, Claude, and every modern LLM. Character-level, small-scale, but architecturally identical. After this module, the transformer has no mysteries.

```
    "The transformer is just:
        embedding
      + [self-attention + feed-forward] x N
      + output projection

     That's it. Everything else is scale."
```

---

## The Transformer Architecture

### The Full Picture

```
    Input tokens:  "to be or not"
                    ↓
    ┌─────────────────────────────────────┐
    │  Token Embedding + Position Embedding │
    │  "to" → [0.2, -0.1, 0.5, ...]      │
    │  + position 0 → [0.01, 0.03, ...]   │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │         Transformer Block x N        │
    │  ┌─────────────────────────────┐    │
    │  │  Multi-Head Self-Attention   │    │
    │  │  (tokens look at each other) │    │
    │  └──────────────┬──────────────┘    │
    │                 │ + residual         │
    │  ┌──────────────┴──────────────┐    │
    │  │  Layer Normalization         │    │
    │  └──────────────┬──────────────┘    │
    │  ┌──────────────┴──────────────┐    │
    │  │  Feed-Forward Network        │    │
    │  │  (process each token)        │    │
    │  └──────────────┬──────────────┘    │
    │                 │ + residual         │
    │  ┌──────────────┴──────────────┐    │
    │  │  Layer Normalization         │    │
    │  └─────────────────────────────┘    │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │  Linear → Logits → Softmax → Probs  │
    │  P(next token) = [0.01, ..., 0.15]  │
    └─────────────────────────────────────┘
                    ↓
            Next token: "to"
```

### Self-Attention: The Core Innovation

Self-attention lets every token "look at" every other token and decide what's relevant.

**The Intuition:** When predicting what comes after "The cat sat on the ___", the model needs to know about "cat" and "sat" even though they're far from "___". Self-attention lets the blank position attend to all previous tokens.

**The Mechanism -- Query, Key, Value:**

```
    Each token produces three vectors:
    - Query (Q): "What am I looking for?"
    - Key (K):   "What do I contain?"
    - Value (V): "What information do I provide?"

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Step by step:
    1. Q @ K^T           → attention scores (who should I pay attention to?)
    2. / sqrt(d_k)       → scale down (prevents scores from being too large)
    3. mask future        → GPT can only look backward (causal mask)
    4. softmax            → convert scores to weights (sum to 1)
    5. weights @ V        → weighted combination of value vectors
```

### Worked Example: Self-Attention

4 tokens, embedding dimension = 3, head size = 3:

```
    Tokens: ["the", "cat", "sat", "."]

    After embedding, each token is a vector:
    the = [1.0, 0.5, 0.2]
    cat = [0.3, 0.8, 0.1]
    sat = [0.7, 0.2, 0.6]
    .   = [0.1, 0.1, 0.9]

    Q = X @ W_q    K = X @ W_k    V = X @ W_v
    (each is a learned linear projection)

    Attention scores = Q @ K^T:
              the   cat   sat    .
    the  [   0.8   0.3   0.5   0.1  ]
    cat  [   0.4   0.9   0.2   0.3  ]
    sat  [   0.6   0.5   0.7   0.2  ]
    .    [   0.2   0.4   0.3   0.8  ]

    Apply causal mask (future positions → -infinity):
              the   cat   sat    .
    the  [   0.8  -inf  -inf  -inf  ]  ← "the" can only see itself
    cat  [   0.4   0.9  -inf  -inf  ]  ← "cat" sees "the" and itself
    sat  [   0.6   0.5   0.7  -inf  ]  ← "sat" sees all previous
    .    [   0.2   0.4   0.3   0.8  ]  ← "." sees everything

    After softmax (each row sums to 1):
              the   cat   sat    .
    the  [  1.00  0.00  0.00  0.00  ]
    cat  [  0.38  0.62  0.00  0.00  ]
    sat  [  0.33  0.30  0.37  0.00  ]
    .    [  0.17  0.22  0.20  0.41  ]

    Output = attention_weights @ V
    Each token's output is a weighted sum of all Value vectors it can see.
```

### Multi-Head Attention

Instead of one attention mechanism, use multiple "heads" in parallel:

```
    Head 1: learns to attend to syntactic relationships
    Head 2: learns to attend to semantic relationships
    Head 3: learns to attend to positional proximity
    Head 4: learns to attend to something else entirely

    Each head has its own Q, K, V projections (smaller dimension).
    Results are concatenated and projected back to full dimension.
```

**Why multiple heads?** A single attention head can only compute one type of attention pattern. Multiple heads let the model attend to different things simultaneously.

### Feed-Forward Network

After attention, each token passes through a small neural network:

```python
FFN(x) = Linear(ReLU(Linear(x)))
# Typically: d_model → 4*d_model → d_model
```

**The intuition:** Attention gathers information from other tokens. The feed-forward network processes that gathered information for each token independently. Attention = communication. FFN = computation.

### Residual Connections

```python
x = x + attention(x)    # not just attention(x)
x = x + ffn(x)          # not just ffn(x)
```

**Why?** Without residual connections, gradients must flow through every layer, and they often vanish or explode. With residual connections, gradients have a "highway" that bypasses layers, making deep networks trainable.

### Layer Normalization

```python
x = layer_norm(x)  # normalize to mean=0, variance=1 per token
```

Stabilizes training by keeping activations in a reasonable range. Similar to BatchNorm from makemore but normalized per-token instead of per-batch.

---

## Hyperparameter Guide

| Parameter | Symbol | Typical Values | What It Controls |
|-----------|--------|---------------|-----------------|
| Embedding dimension | `n_embd` | 64-12288 | Richness of token representations |
| Number of heads | `n_head` | 4-96 | Parallel attention patterns (must divide n_embd) |
| Number of layers | `n_layer` | 4-96 | Depth of the network (more layers = more capacity) |
| Context length | `block_size` | 64-128K | How far back the model can look |
| Batch size | `batch_size` | 16-2048 | Training throughput vs memory |
| Learning rate | `lr` | 1e-4 to 6e-4 | Step size (transformers use smaller LR than simple nets) |
| Dropout | `dropout` | 0.0-0.3 | Regularization (randomly zeros activations during training) |

**Scaling relationships:**
- GPT-2 Small: n_embd=768, n_head=12, n_layer=12 (117M params)
- GPT-2 Medium: n_embd=1024, n_head=16, n_layer=24 (345M params)
- GPT-3: n_embd=12288, n_head=96, n_layer=96 (175B params)
- Our nanoGPT: n_embd=64, n_head=4, n_layer=4 (~0.2M params)

Same architecture at every scale. Just bigger numbers.

---

## Experiments

| # | Experiment | What to Change | What to Observe |
|---|-----------|---------------|-----------------|
| 1 | Train on Shakespeare | Default dataset | Generated text should be Shakespeare-like |
| 2 | Different corpus | Use Python code, song lyrics, etc. | Model captures the "style" of the training data |
| 3 | Context length | block_size: 32, 64, 128, 256 | Longer context = more coherent output |
| 4 | Number of layers | n_layer: 1, 2, 4, 8 | More layers = better quality (with diminishing returns) |
| 5 | Number of heads | n_head: 1, 2, 4 | Single head vs multi-head quality |
| 6 | Remove positional embedding | Comment out pos_emb | Model can't distinguish token order -- output is incoherent |
| 7 | Remove causal mask | Don't mask future | Model "cheats" during training, can't generate properly |
| 8 | Visualize attention | Extract attention weights | See which tokens attend to which |

---

## From NanoGPT to Real GPT

| | NanoGPT (yours) | GPT-2 | GPT-3 | GPT-4 |
|---|----------------|-------|-------|-------|
| Parameters | ~0.2M | 1.5B | 175B | ~1.8T (est.) |
| Training data | ~1MB text | 40GB WebText | 300B tokens | Unknown (trillions) |
| Context length | 64-256 | 1024 | 2048 | 128K |
| Training time | Minutes (CPU) | Days (8 GPUs) | Weeks (1000s GPUs) | Months (10000s GPUs) |
| Architecture | Same | Same | Same | Same + MoE (est.) |

**The architecture is the same.** The only differences: more parameters, more data, more compute, and training techniques (RLHF, instruction tuning). You built the core.

---

## Re-reading "Attention Is All You Need"

Now that you've built a transformer, re-read the paper. Here's what maps to what:

| Paper Section | Your Code |
|---------------|-----------|
| "Scaled Dot-Product Attention" (Section 3.2.1) | `Head` class: Q@K^T / sqrt(d_k), mask, softmax, @V |
| "Multi-Head Attention" (Section 3.2.2) | `MultiHeadAttention`: multiple Heads, concatenate, project |
| "Position-wise Feed-Forward Networks" (Section 3.3) | `FeedForward`: Linear -> ReLU -> Linear |
| "Embeddings and Softmax" (Section 3.4) | Token embedding + positional embedding |
| "Positional Encoding" (Section 3.5) | Learned positional embeddings (paper uses sinusoidal) |
| "Encoder and Decoder Stacks" (Section 3.1) | `Block` stacked N times (GPT uses decoder-only) |

---

## Key Takeaways

1. **The transformer = embedding + [attention + FFN] x N + output.** That's the entire architecture.
2. **Self-attention lets every token look at every other token.** This is why transformers handle long-range dependencies.
3. **The causal mask makes it autoregressive.** GPT generates left to right by masking future positions.
4. **Multi-head attention captures different types of relationships.** One head might learn syntax, another semantics.
5. **Residual connections and layer norm make deep networks trainable.** Without them, gradients vanish.
6. **Scale is the secret.** Your nanoGPT and GPT-4 are the same architecture. The difference is 10,000x more parameters and data.

---

## PHASE 1 COMPLETE

You now understand:
- How neural networks learn (micrograd)
- How language models work (makemore)
- How transformers work (nanoGPT)
- The training loop that trains all of them (MNIST)

Everything from here is applying this knowledge to real problems.

**Next: Phase 2** -- where you build the things clients pay for.
