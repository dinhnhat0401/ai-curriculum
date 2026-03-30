# Phase 1: The Engine Room

**Weeks 1-4 | ~22 hours | The foundation everything else is built on.**

---

## What This Phase Covers

You will build four projects, each one building on the last:

```
MNIST ──────> Micrograd ──────> Makemore ──────> NanoGPT
"Hello        "How do          "How do          "How does
 World"        neural nets      language         GPT
               learn?"          models work?"    work?"

Train your    Build an         Build character   Build a
first neural  autograd engine  language models   transformer
network on    from scratch.    with embeddings.  from scratch.
handwritten   Backprop in                        Self-attention,
digits.       100 lines.                         multi-head
                                                 attention,
                                                 the works.
```

## The Learning Arc

| Module | What You Build | Key Concepts | Time |
|--------|---------------|--------------|------|
| **MNIST** | Digit classifier with PyTorch | Tensors, forward pass, loss, training loop | ~5h |
| **Micrograd** | Autograd engine from scratch | Computational graphs, chain rule, backprop | ~5h |
| **Makemore** | Character language model | Embeddings, bigrams, MLPs, batch normalization | ~6h |
| **NanoGPT** | GPT transformer from scratch | Self-attention, multi-head attention, positional encoding | ~6h |

## Prerequisites

- Python (comfortable with classes and basic data structures)
- High school calculus (derivatives, chain rule)
- `pip install torch torchvision matplotlib numpy`

## Key Concepts You Will Master

1. **Tensors** -- Multi-dimensional arrays. The basic data structure of all deep learning.
2. **Forward Pass** -- Data flows through the network, producing predictions.
3. **Loss Functions** -- Measure how wrong the predictions are.
4. **Backpropagation** -- Compute gradients by applying the chain rule backward through the network.
5. **Gradient Descent** -- Update parameters in the direction that reduces loss.
6. **Embeddings** -- Learned dense vector representations of discrete tokens. Critical for RAG.
7. **Self-Attention** -- The mechanism that lets tokens "look at" other tokens. The core of transformers.
8. **The Training Loop** -- forward -> loss -> backward -> update. The same pattern from MNIST to GPT-4.

## After This Phase, You Will Understand

- How neural networks learn (not just "gradient descent" hand-waving -- the actual code)
- How language models assign probabilities to sequences
- How transformers work at every level of abstraction
- Why attention changed everything
- The connection between your 100-line micrograd and PyTorch's autograd
- The connection between your nanoGPT and GPT-4 (it's the same architecture, just scaled up)

## How to Work Through This Phase

1. Watch the recommended videos (Karpathy, 3Blue1Brown)
2. Code along, then close the video and rebuild from memory
3. Read the README in each module for deeper explanations
4. Do the exercises -- they're where the real learning happens
5. When stuck, look at the code, understand it, then close it and try again

Start with `mnist/` -->
