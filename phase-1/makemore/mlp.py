"""
MLP Character-Level Language Model (following Bengio et al. 2003)

Uses N previous characters (context window) to predict the next character.
Introduces the critical concept of EMBEDDINGS: learned dense vector
representations of discrete tokens.

Architecture:
    context chars → embedding lookup → concatenate → hidden layer → output
"""

import torch
import torch.nn.functional as F
import os

# ============================================================
# Load Data
# ============================================================

data_path = os.path.join(os.path.dirname(__file__), "names.txt")
words = open(data_path, "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# ============================================================
# Hyperparameters
# ============================================================

CONTEXT_SIZE = 3      # how many previous characters to use as context
EMBED_DIM = 10        # dimension of character embeddings
HIDDEN_SIZE = 200     # neurons in hidden layer
BATCH_SIZE = 32       # mini-batch size
MAX_STEPS = 20000     # training iterations
LEARNING_RATE = 0.1   # initial learning rate
LR_DECAY_STEP = 15000 # step at which to reduce learning rate

# ============================================================
# Build Dataset
# ============================================================

def build_dataset(words):
    """Convert words into (context, target) pairs.

    For context_size=3 and word "emma":
        ...  →  e    (context: [0,0,0], target: 5)
        ..e  →  m    (context: [0,0,5], target: 13)
        .em  →  m    (context: [0,5,13], target: 13)
        emm  →  a    (context: [5,13,13], target: 1)
        mma  →  .    (context: [13,13,1], target: 0)
    """
    X, Y = [], []
    for w in words:
        context = [0] * CONTEXT_SIZE  # start with all '.' tokens
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # slide window forward
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# Split data: 80% train, 10% val, 10% test
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])    # training set
Xval, Yval = build_dataset(words[n1:n2]) # validation set
Xte, Yte = build_dataset(words[n2:])     # test set

print(f"Training:   {Xtr.shape[0]:6d} examples")
print(f"Validation: {Xval.shape[0]:6d} examples")
print(f"Test:       {Xte.shape[0]:6d} examples")

# ============================================================
# Initialize Model Parameters
# ============================================================

g = torch.Generator().manual_seed(2147483647)

# The embedding table: vocab_size rows, each row is a EMBED_DIM-dimensional vector
# This is the KEY concept: each character gets a learned dense representation
C = torch.randn((vocab_size, EMBED_DIM), generator=g)

# Hidden layer: takes concatenated embeddings, outputs HIDDEN_SIZE activations
W1 = torch.randn((CONTEXT_SIZE * EMBED_DIM, HIDDEN_SIZE), generator=g) * 0.2  # scaled init
b1 = torch.randn(HIDDEN_SIZE, generator=g) * 0.01

# Output layer: maps hidden activations to logits over vocabulary
W2 = torch.randn((HIDDEN_SIZE, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

parameters = [C, W1, b1, W2, b2]
total_params = sum(p.nelement() for p in parameters)
print(f"\nModel parameters: {total_params}")

for p in parameters:
    p.requires_grad = True

# ============================================================
# Training
# ============================================================

print(f"\nTraining for {MAX_STEPS} steps...")
losses = []

for i in range(MAX_STEPS):
    # Mini-batch: randomly select BATCH_SIZE examples
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))

    # Forward pass
    # 1. Embedding lookup: each character index → its embedding vector
    emb = C[Xtr[ix]]                          # [batch, context_size, embed_dim]

    # 2. Concatenate all context embeddings into one vector
    h_preact = emb.view(-1, CONTEXT_SIZE * EMBED_DIM) @ W1 + b1  # [batch, hidden]

    # 3. Activation function (tanh squashes to [-1, 1])
    h = torch.tanh(h_preact)

    # 4. Output logits
    logits = h @ W2 + b2                       # [batch, vocab_size]

    # 5. Loss (cross-entropy)
    loss = F.cross_entropy(logits, Ytr[ix])

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update: learning rate decay
    lr = LEARNING_RATE if i < LR_DECAY_STEP else LEARNING_RATE * 0.1
    for p in parameters:
        p.data += -lr * p.grad

    # Track loss
    losses.append(loss.item())
    if i % 5000 == 0 or i == MAX_STEPS - 1:
        print(f"  Step {i:5d} | Loss: {loss.item():.4f} | LR: {lr}")

# ============================================================
# Evaluate
# ============================================================

def evaluate(X, Y, name):
    emb = C[X]
    h = torch.tanh(emb.view(-1, CONTEXT_SIZE * EMBED_DIM) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(f"{name} loss: {loss.item():.4f}")
    return loss.item()

print()
evaluate(Xtr, Ytr, "Train")
evaluate(Xval, Yval, "Val  ")
evaluate(Xte, Yte, "Test ")

# ============================================================
# Generate Names
# ============================================================

print("\nGenerated names:")
for _ in range(10):
    out = []
    context = [0] * CONTEXT_SIZE

    while True:
        # Embed the context
        emb = C[torch.tensor([context])]  # [1, context_size, embed_dim]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)

        # Sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]

        if ix == 0:  # end token
            break
        out.append(itos[ix])

    print(f"  {''.join(out)}")

# ============================================================
# Embedding Visualization
# ============================================================

print("\nEmbedding vectors (first 3 dimensions):")
print(f"  {'Char':>4} | {'Dim 0':>7} {'Dim 1':>7} {'Dim 2':>7}")
print(f"  {'-'*4}-+-{'-'*7}-{'-'*7}-{'-'*7}")
for i in range(min(vocab_size, 10)):
    ch = itos[i]
    vec = C[i].tolist()
    print(f"  {repr(ch):>4} | {vec[0]:7.3f} {vec[1]:7.3f} {vec[2]:7.3f}")
print("  ...")
print("  (Similar characters will have similar embedding vectors)")
