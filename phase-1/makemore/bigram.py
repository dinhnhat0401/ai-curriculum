"""
Bigram Language Model - Two approaches to the same problem.

Part 1: Counting-based bigram (pure statistics)
Part 2: Neural bigram (single linear layer trained with gradient descent)

Both converge to the same result. The neural version is the foundation
for everything that follows.
"""

import torch
import torch.nn.functional as F
import os

# ============================================================
# Load Data
# ============================================================

# Read names from file
data_path = os.path.join(os.path.dirname(__file__), "names.txt")
words = open(data_path, "r").read().splitlines()
print(f"Loaded {len(words)} names")
print(f"Examples: {words[:5]}")

# Build character vocabulary
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}  # char -> index
stoi["."] = 0  # special start/end token at index 0
itos = {i: s for s, i in stoi.items()}  # index -> char
vocab_size = len(stoi)
print(f"Vocabulary size: {vocab_size} ({', '.join(chars[:5])}...)")

# ============================================================
# PART 1: Counting-Based Bigram
# ============================================================
print("\n" + "=" * 60)
print("PART 1: Counting-Based Bigram Model")
print("=" * 60)

# Count all character pairs
# N[i][j] = number of times character j follows character i
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]  # add start/end tokens
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Convert counts to probabilities (add smoothing to avoid log(0))
P = (N + 1).float()  # +1 smoothing (Laplace smoothing)
P = P / P.sum(dim=1, keepdim=True)  # normalize each row to sum to 1

# Generate names by sampling from the bigram distribution
print("\nGenerated names (counting-based):")
for i in range(5):
    out = []
    ix = 0  # start with '.' token
    while True:
        # Sample next character from probability distribution
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        if ix == 0:  # hit end token
            break
        out.append(itos[ix])
    print(f"  {''.join(out)}")

# Evaluate: compute negative log likelihood on training data
log_likelihood = 0.0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += torch.log(prob)
        n += 1

nll = -log_likelihood / n
print(f"\nNegative log likelihood (counting): {nll:.4f}")

# ============================================================
# PART 2: Neural Bigram Model
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Neural Bigram Model")
print("=" * 60)

# Build training dataset: pairs of (current_char, next_char)
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.shape[0]
print(f"Training examples: {num_examples}")

# Initialize weights (this single matrix IS the model)
# It serves as both the embedding table and the output projection
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

# Training loop
print("\nTraining neural bigram...")
for k in range(200):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=vocab_size).float()  # one-hot encode inputs
    logits = xenc @ W  # matrix multiply: [N, 27] @ [27, 27] = [N, 27]
    counts = logits.exp()  # exponentiate to get "counts" (positive values)
    probs = counts / counts.sum(1, keepdim=True)  # normalize to probabilities

    # Loss: negative log likelihood
    # For each example, look up the probability assigned to the correct next char
    loss = -probs[torch.arange(num_examples), ys].log().mean()

    # Add regularization to push weights toward 0 (like smoothing)
    loss += 0.01 * (W ** 2).mean()

    # Backward pass
    W.grad = None  # zero gradients
    loss.backward()

    # Update weights
    W.data += -50 * W.grad  # large learning rate (this is a simple model)

    if k % 50 == 0 or k == 199:
        print(f"  Step {k:3d} | Loss: {loss.item():.4f}")

# Generate names from neural model
print("\nGenerated names (neural bigram):")
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(f"  {''.join(out)}")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Counting model NLL: {nll:.4f}")
print(f"Neural model loss:  {loss.item():.4f}")
print("\nThe neural model's loss should be close to the counting model's NLL.")
print("They solve the same problem -- the neural version just uses gradient descent")
print("instead of direct counting.")
print("\nKey insight: the weight matrix W is essentially a learned version of")
print("the probability table P. One-hot encoding + matrix multiply = table lookup.")
