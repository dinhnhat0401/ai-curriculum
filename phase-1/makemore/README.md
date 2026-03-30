# Makemore: Character-Level Language Modeling

**Time: ~6 hours | Difficulty: Intermediate | Files: `bigram.py`, `mlp.py`, `names.txt`**

---

## What You Will Learn

You will build language models that generate new words by learning patterns from a dataset of names. Along the way, you'll master **embeddings** -- the single most important concept for everything in Phase 2 (RAG, agents, fine-tuning).

```
    Training data: emma, olivia, ava, sophia, isabella, ...
    Generated:     emmalia, sophira, avelina, oliva, ...

    The model learns the "shape" of names and generates new ones
    that sound plausible but don't exist.
```

---

## Concepts Covered

### What is a Language Model?

A language model assigns probabilities to sequences of tokens. Given some context, it predicts what comes next.

```
    "emm" → P(next='a') = 0.65, P(next='i') = 0.12, P(next='e') = 0.08, ...

    The model has learned that 'a' is the most likely character after "emm"
    because names like "emma" are common in the training data.
```

Every LLM -- GPT-4, Claude, Llama -- is a language model. They just operate on subword tokens instead of characters, and have billions of parameters instead of thousands.

### Embeddings (The Key Concept)

An embedding converts a discrete token (like a character or word) into a continuous vector:

```
    Token       →    Embedding Vector (learned)
    ─────────────────────────────────────────────
    'a'         →    [0.12, -0.45, 0.78, 0.33]
    'b'         →    [-0.67, 0.23, 0.11, -0.89]
    'z'         →    [0.45, 0.67, -0.12, 0.55]

    Similar characters end up with similar vectors:
    'a' and 'e' (both vowels) → vectors are close in space
    'a' and 'z' → vectors are far apart
```

**Why this matters for RAG:** In Phase 2, you'll embed entire documents into vectors and find similar ones using cosine similarity. The concept is identical -- just at a larger scale. If you understand character embeddings here, document embeddings in RAG will be obvious.

### From Counting to Learning

The bigram model shows that counting and neural networks are two sides of the same coin:

```
    Counting approach:
        Count how often each character pair appears
        Normalize to get probabilities
        P('a' after 'e') = count('ea') / count('e*')

    Neural approach:
        One-hot encode the input character
        Multiply by a weight matrix (the embedding)
        Softmax to get probabilities
        Train to minimize negative log likelihood

    Result: they converge to the SAME probabilities
    The neural version is just a differentiable version of counting
```

---

## Part 1: Bigram Model (`bigram.py`)

A bigram model predicts the next character using only the previous character.

### The Algorithm

```
    1. Count all character pairs in the training data:
       ('e','m') appears 42 times
       ('m','m') appears 15 times
       ('m','a') appears 67 times
       ...

    2. Normalize each row to get probabilities:
       After 'm': P('a')=0.35, P('m')=0.08, P('e')=0.12, ...

    3. To generate: start with a special START token,
       sample the next character from the probability distribution,
       repeat until we sample the END token.
```

### The Neural Version

The same model as a neural network:
- Input: one-hot encoded character (27-dim vector: 26 letters + '.')
- Weight matrix: 27x27 (this IS the embedding table)
- Output: softmax of logits -> probability distribution over next character
- Loss: negative log likelihood (cross-entropy)

Training this neural network converges to the same probabilities as counting. But the neural version generalizes: it can be extended to larger contexts, deeper networks, and richer representations.

---

## Part 2: MLP Language Model (`mlp.py`)

Following Bengio et al. (2003), we build an MLP that uses N previous characters to predict the next one.

### Architecture

```
    Context: ['e', 'm', 'm']    (3 previous characters)
              ↓     ↓     ↓
         ┌────────────────────┐
         │  Embedding Lookup  │   Each char → d-dimensional vector
         │  (27 x d matrix)   │   d = embedding dimension (e.g., 10)
         └────────────────────┘
              ↓     ↓     ↓
         [0.3, -0.1, ...]  [0.5, 0.2, ...]  [0.5, 0.2, ...]
              ↓     ↓     ↓
         ┌────────────────────┐
         │    Concatenate     │   3 vectors → one vector of size 3*d
         └────────────────────┘
              ↓
         ┌────────────────────┐
         │  Hidden Layer      │   Linear(3*d, hidden_size) + tanh
         └────────────────────┘
              ↓
         ┌────────────────────┐
         │  Output Layer      │   Linear(hidden_size, 27) + softmax
         └────────────────────┘
              ↓
         P(next char) = [0.02, 0.35, 0.01, ...]
                              'a' is most likely → sample 'a'
```

### Key Ideas

1. **Embedding table**: A learnable matrix. Row `i` is the embedding for character `i`. Looking up a character's embedding is just indexing into this matrix.

2. **Context window**: Using 3 previous characters captures patterns like "emm" → "a". Larger windows capture longer patterns but need more data to learn.

3. **Training**: Same loop as always -- forward, loss, backward, update. The embedding table's gradients tell us how to adjust each character's representation.

---

## Part 3: Batch Normalization

Deep networks are hard to train because activations can become very large or very small as they pass through layers.

### The Problem

```
    Layer 1 output:  [-0.2, 0.5, -0.1, 0.3]     ← reasonable
    Layer 2 output:  [-2.1, 5.3, -1.5, 3.7]      ← getting larger
    Layer 3 output:  [-21, 53, -15, 37]           ← exploding!

    Or worse:
    Layer 1 output:  [-0.002, 0.005, -0.001, 0.003]  ← tiny
    Layer 2 output:  [-0.00002, 0.00005, ...]          ← vanishing!
```

### The Solution: BatchNorm

Before each activation function, normalize the activations across the batch:

```
    1. Compute mean and std across the batch
    2. Normalize: x_norm = (x - mean) / std
    3. Scale and shift: output = gamma * x_norm + beta
       (gamma and beta are learned parameters)
```

This keeps activations in a reasonable range, making training faster and more stable.

---

## Experiments

| # | Experiment | What to Try | What to Learn |
|---|-----------|-------------|---------------|
| 1 | Different datasets | Names from different cultures | Models capture cultural patterns |
| 2 | Context window | 2, 3, 5, 8 characters | Longer context = better quality, needs more data |
| 3 | Embedding dimension | 2, 10, 32 | Larger = more expressive, slower to train |
| 4 | Visualize embeddings | Plot 2D embeddings | See which characters cluster together |
| 5 | Compare bigram vs MLP | Same dataset, same eval | MLP should produce better names |
| 6 | Hidden layer size | 64, 128, 256, 512 | Diminishing returns after a point |
| 7 | Learning rate decay | Start 0.1, decay to 0.01 | Faster convergence, better final loss |
| 8 | Temperature sampling | T=0.5, 1.0, 1.5, 2.0 | Low T = conservative, High T = creative |

### Temperature

When sampling from the model, temperature controls randomness:

```python
logits = model(context)
probs = softmax(logits / temperature)
next_char = sample(probs)

# temperature = 0.5: sharper distribution, safer/more common outputs
# temperature = 1.0: normal distribution (as trained)
# temperature = 2.0: flatter distribution, more diverse/unusual outputs
```

This is the same "temperature" parameter you set when calling GPT or Claude APIs.

---

## Key Takeaways

1. **A language model assigns probabilities to sequences.** Everything from bigrams to GPT-4 does this.
2. **Embeddings convert discrete tokens to continuous vectors.** This is how neural networks handle categorical data.
3. **Counting and neural networks can solve the same problem.** The neural version is just differentiable, so it can be extended.
4. **Context window size matters.** More context = better predictions, but more parameters and data needed.
5. **BatchNorm stabilizes training** by keeping activations in a reasonable range.
6. **Embeddings are the bridge to Phase 2.** When you build RAG, you'll embed documents the same way you embed characters here.

---

## Next Steps

You've built language models that work character by character. But there's a fundamental limitation: the MLP has a fixed context window, and information from distant characters can't flow to the prediction.

**Transformers** solve this with self-attention: every token can "look at" every other token. That's what you'll build in **nanoGPT** -->
