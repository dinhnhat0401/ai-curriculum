"""
NanoGPT -- A GPT Language Model from Scratch

A complete implementation of the transformer architecture for
character-level language modeling. Train on any text and generate more.

Based on Karpathy's "Let's build GPT" lecture.

Usage:
    python gpt.py                    # trains on built-in data and generates text
    python gpt.py --data path.txt    # trains on custom text file
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import os

# ============================================================
# Hyperparameters
# ============================================================

# Model architecture
BLOCK_SIZE = 64        # context length (how many tokens the model can see)
N_EMBD = 64            # embedding dimension
N_HEAD = 4             # number of attention heads (must divide N_EMBD)
N_LAYER = 4            # number of transformer blocks
DROPOUT = 0.1          # dropout rate for regularization

# Training
BATCH_SIZE = 32        # number of sequences per training step
MAX_ITERS = 5000       # total training iterations
EVAL_INTERVAL = 500    # evaluate every N steps
EVAL_ITERS = 200       # number of batches to average for evaluation
LEARNING_RATE = 3e-4   # Adam learning rate (standard for transformers)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# Data Loading
# ============================================================

DEFAULT_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city is risen:
why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
Of more strong link asunder than can ever
Appear in your impediment. For the dearth,
The gods, not the patricians, make it, and
Your knees to them, not arms, must help. Alack,
You are transported by calamity
Thither where more attends you, and you slander
The helms o' the state, who care for you like fathers,
When you curse them as enemies.

First Citizen:
Care for us! True, indeed! They ne'er cared for us
yet: suffer us to famish, and their store-houses
crammed with grain; make edicts for usury, to
support usurers; repeal daily any wholesome act
established against the rich, and provide more
piercing statutes daily, to chain up and restrain
the poor. If the wars eat us not up, they will; and
there's all the love they bear us.

MENENIUS:
Either you must
Confess yourselves wondrous malicious,
Or be accused of folly. I shall tell you
A pretty tale: it may be you have heard it;
But, since it serves my purpose, I will venture
To stale 't a little more.

First Citizen:
Well, I'll hear it, sir: yet you must not think to
fob off our disgrace with a tale: but, an 't please
you, deliver.

MENENIUS:
There was a time when all the body's members
Rebell'd against the belly, thus accused it:
That only like a gulf it did remain
I' the midst o' the body, idle and unactive,
Still cupboarding the viand, never bearing
Like labour with the rest, where the other instruments
Did see and hear, devise, instruct, walk, feel,
And, mutually participate, did minister
Unto the appetite and affection common
Of the whole body.
"""


def load_data(filepath=None):
    """Load text data for training."""
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {filepath}")
    else:
        text = DEFAULT_TEXT
        print(f"Using built-in Shakespeare text ({len(text)} chars)")

    # Character-level tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Train/val split (90/10)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocab size: {vocab_size}")
    print(f"Train: {len(train_data)} tokens | Val: {len(val_data)} tokens")

    return train_data, val_data, vocab_size, encode, decode


def get_batch(split, train_data, val_data):
    """Generate a random batch of (input, target) pairs."""
    data = train_data if split == "train" else val_data
    # Random starting positions for each sequence in the batch
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Average loss over multiple batches for stable evaluation."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ============================================================
# Model Components
# ============================================================


class Head(nn.Module):
    """Single head of self-attention.

    This is the core mechanism: each token computes Query, Key, Value vectors,
    then uses dot-product attention to decide which other tokens to attend to.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)    # what do I contain?
        self.query = nn.Linear(N_EMBD, head_size, bias=False)  # what am I looking for?
        self.value = nn.Linear(N_EMBD, head_size, bias=False)  # what info do I provide?
        self.dropout = nn.Dropout(DROPOUT)

        # Causal mask: lower triangular matrix
        # Prevents tokens from attending to future positions
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (embedding dim)

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)

        # Apply causal mask: set future positions to -inf so softmax gives 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax: convert scores to weights (each row sums to 1)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel, then concatenate results.

    Each head can learn to attend to different types of relationships.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)  # projection back to embedding dim
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Run all heads in parallel, concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, N_EMBD)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Applied to each token independently after attention.
    Attention = communication between tokens. FFN = computation per token.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),   # expand
            nn.ReLU(),                         # nonlinearity
            nn.Linear(4 * N_EMBD, N_EMBD),   # project back
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention + feed-forward with residual connections.

    This is the repeating unit. Stack N of these to get a deep transformer.
    """

    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)  # self-attention
        self.ffwd = FeedForward()                         # feed-forward
        self.ln1 = nn.LayerNorm(N_EMBD)                  # layer norm 1
        self.ln2 = nn.LayerNorm(N_EMBD)                  # layer norm 2

    def forward(self, x):
        # Pre-norm architecture (norm before attention/ffn, not after)
        x = x + self.sa(self.ln1(x))    # attention + residual connection
        x = x + self.ffwd(self.ln2(x))  # ffn + residual connection
        return x


class GPTLanguageModel(nn.Module):
    """The full GPT model.

    Combines all components:
    - Token embeddings (what each token "means")
    - Position embeddings (where each token is)
    - N transformer blocks (the processing)
    - Output projection (logits over vocabulary)
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)   # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)  # output projection

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)       # (B, T, N_EMBD)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, N_EMBD)
        x = tok_emb + pos_emb                           # (B, T, N_EMBD)

        # Pass through transformer blocks
        x = self.blocks(x)                               # (B, T, N_EMBD)
        x = self.ln_f(x)                                 # (B, T, N_EMBD)

        # Project to vocabulary size
        logits = self.lm_head(x)                         # (B, T, vocab_size)

        # Compute loss if targets provided (training mode)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate text autoregressively.

        Given a context (idx), predict one token at a time and append it.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (model can't see more than this)
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Focus on the last time step (we only need the next-token prediction)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# ============================================================
# Training
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train nanoGPT")
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    args = parser.parse_args()

    # Check for data file in data/ directory
    data_path = args.data
    if data_path is None:
        default_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
        if os.path.exists(default_path):
            data_path = default_path

    train_data, val_data, vocab_size, encode, decode = load_data(data_path)

    # Create model
    model = GPTLanguageModel(vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {MAX_ITERS} iterations...")
    print("-" * 50)

    for iter in range(MAX_ITERS):
        # Evaluate periodically
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {iter:5d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

        # Get a training batch
        xb, yb = get_batch("train", train_data, val_data)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate text
    print("\n" + "=" * 50)
    print("GENERATED TEXT")
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))
    print("=" * 50)


if __name__ == "__main__":
    main()
