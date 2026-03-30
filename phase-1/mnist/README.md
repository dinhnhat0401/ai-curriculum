# MNIST: Your First Neural Network

**Time: ~5 hours | Difficulty: Beginner | File: `hello.py`**

---

## What You Will Learn

You will train a neural network to recognize handwritten digits with >95% accuracy. More importantly, you will understand the **training loop pattern** -- the same pattern used in every model from this simple classifier to GPT-4.

```
                    THE TRAINING LOOP
                    =================

    ┌──────────────────────────────────────────┐
    │                                          │
    │   for epoch in range(epochs):            │
    │       for images, labels in dataloader:  │
    │                                          │
    │           predictions = model(images)     │  ← Forward pass
    │           loss = criterion(pred, labels)  │  ← Compute loss
    │           optimizer.zero_grad()           │  ← Clear gradients
    │           loss.backward()                 │  ← Backward pass
    │           optimizer.step()                │  ← Update weights
    │                                          │
    └──────────────────────────────────────────┘

    This is the pattern. Memorize it.
    Everything else is details.
```

---

## Concepts Covered

### Tensors

A tensor is a multi-dimensional array. It's the fundamental data structure of deep learning.

```
Scalar (0D tensor):     42
Vector (1D tensor):     [1, 2, 3, 4]
Matrix (2D tensor):     [[1, 2], [3, 4], [5, 6]]
3D tensor:              A batch of matrices (e.g., batch of images)
```

An MNIST image is a 28x28 pixel grid. As a tensor: shape `[1, 28, 28]` (1 channel, 28 height, 28 width). A batch of 64 images: shape `[64, 1, 28, 28]`.

### Neural Network Architecture

```
    Input Layer          Hidden Layer 1       Hidden Layer 2       Output Layer
    (784 neurons)        (784 neurons)        (128 neurons)        (10 neurons)

    ┌───┐                ┌───┐                ┌───┐                ┌───┐
    │   │───────────────>│   │───────────────>│   │───────────────>│ 0 │  ← P(digit=0)
    │   │     weights    │   │     weights    │   │     weights    │ 1 │  ← P(digit=1)
    │784│───────────────>│784│───────────────>│128│───────────────>│ 2 │  ← P(digit=2)
    │   │                │   │                │   │                │...│
    │   │───────────────>│   │───────────────>│   │───────────────>│ 9 │  ← P(digit=9)
    └───┘                └───┘                └───┘                └───┘
    28x28 pixels         + ReLU               + ReLU               softmax
    flattened            activation           activation           (pick highest)
```

- **Input**: 28x28 = 784 pixel values, flattened into a vector
- **Hidden layers**: Transform the input through learned weights + nonlinear activations
- **Output**: 10 values (one per digit), fed through softmax to get probabilities
- **Prediction**: The digit with the highest probability

### Forward Pass

Data flows left to right through the network:

```
x = flatten(image)           # [784]
x = relu(linear1(x))         # [784] -> [784] -> relu -> [784]
x = relu(linear2(x))         # [784] -> [128] -> relu -> [128]
x = linear3(x)               # [128] -> [10]  (raw logits)
```

Each `linear` layer computes: `output = input @ weights + bias`

### Loss Function: CrossEntropyLoss

Measures how wrong the predictions are. For classification:

```
prediction = [0.1, 0.05, 0.8, 0.02, ...]   # model thinks it's a "2"
true label = 2                               # it IS a "2"
loss = -log(0.8) = 0.22                      # low loss -- correct!

prediction = [0.1, 0.05, 0.1, 0.02, ...]    # model is confused
true label = 2
loss = -log(0.1) = 2.30                      # high loss -- wrong!
```

Lower loss = better predictions. Training minimizes this.

### Backpropagation

After computing loss, we compute gradients -- how much each weight contributed to the error. Then we adjust weights to reduce the error. The chain rule lets us compute this efficiently.

This module shows backprop as a black box (`loss.backward()`). Micrograd will show you what's inside that box.

### Optimization: Adam

Adam is an optimizer that updates weights using gradients:
- **SGD (Stochastic Gradient Descent)**: `weight -= learning_rate * gradient`
- **Adam**: Like SGD but with momentum (remembers past gradients) and adaptive learning rates (different rates per parameter). Almost always better than SGD.

### Key Hyperparameters

| Parameter | Our Value | What It Controls |
|-----------|-----------|-----------------|
| `batch_size` | 64 | Images per gradient update. Larger = more stable but slower to converge. |
| `epochs` | 5 | Full passes through the training data. More = better, until overfitting. |
| `learning_rate` | 1e-3 | Step size for weight updates. Too high = diverge. Too low = slow. |

---

## Code Walkthrough

The complete code is in `hello.py`. Here's every section explained.

### 1. Configuration

```python
batch_size = 64
epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

We use GPU if available (much faster for matrix math), otherwise CPU.

### 2. Data Loading

```python
transform = transforms.Compose([
    transforms.ToTensor(),                      # PIL image -> tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # normalize to mean=0, std=1
])
```

**Why normalize?** Neural networks train better when inputs are centered around 0 with unit variance. 0.1307 and 0.3081 are the precomputed mean and std of the MNIST dataset.

### 3. Model Architecture

```python
class SimpleNN(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()        # [B, 1, 28, 28] -> [B, 784]
        self.fc1 = nn.Linear(784, 784)     # 784 -> 784
        self.relu = nn.ReLU()              # nonlinearity
        self.fc2 = nn.Linear(784, 128)     # 784 -> 128
        self.relu2 = nn.ReLU()             # nonlinearity
        self.fc3 = nn.Linear(128, 10)      # 128 -> 10 (digits)
```

**Why ReLU?** Without nonlinear activations, stacking linear layers is equivalent to a single linear layer (matrix multiplication is linear). ReLU (`max(0, x)`) breaks linearity, letting the network learn complex patterns.

### 4. Training Loop

This is the most important code block in this entire curriculum:

```python
for epoch in range(epochs):
    model.train()                            # enable training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                # 1. Clear old gradients
        outputs = model(images)              # 2. Forward pass
        loss = criterion(outputs, labels)    # 3. Compute loss
        loss.backward()                      # 4. Compute gradients
        optimizer.step()                     # 5. Update weights
```

**Why `zero_grad()` first?** PyTorch accumulates gradients by default. If you don't zero them, gradients from the previous batch add to the current batch's gradients, giving wrong updates.

**Why this exact order?**
1. Zero gradients (clean slate)
2. Forward pass (get predictions)
3. Loss (measure error)
4. Backward (compute gradients)
5. Step (update weights using gradients)

Skip any step and training breaks.

### 5. Evaluation

```python
model.eval()                                 # disable dropout, batchnorm training mode
with torch.no_grad():                        # don't compute gradients (saves memory)
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)        # highest logit = predicted digit
        correct += (preds == labels).sum()
```

**Why `model.eval()` and `torch.no_grad()`?** During evaluation, we don't need gradients (not training). `eval()` changes behavior of dropout and batch norm layers. `no_grad()` saves memory and computation.

---

## Experiments to Try

These experiments build intuition. Try each one and observe what happens.

| # | Experiment | What to Change | What to Observe |
|---|-----------|---------------|-----------------|
| 1 | **Tiny hidden layer** | `fc1`: 784->32 | Accuracy drops. Too few neurons to capture digit patterns. |
| 2 | **Huge hidden layer** | `fc1`: 784->2048 | Accuracy may improve slightly, training is slower. Diminishing returns. |
| 3 | **Deeper network** | Add fc4, fc5 layers | May or may not help. Deep nets can be harder to train. |
| 4 | **No activation** | Remove ReLU | Accuracy drops significantly. Linear model can't capture complex patterns. |
| 5 | **High learning rate** | `lr = 0.1` | Loss spikes or diverges. Steps are too big. |
| 6 | **Low learning rate** | `lr = 0.00001` | Loss barely decreases. Steps are too small. |
| 7 | **Batch size 1** | `batch_size = 1` | Very noisy training (loss jumps around). Slow per epoch. |
| 8 | **Batch size 1024** | `batch_size = 1024` | Smoother training but slower convergence (fewer updates per epoch). |
| 9 | **50 epochs** | `epochs = 50` | Watch for overfitting: train loss keeps dropping, test accuracy plateaus or drops. |
| 10 | **No normalization** | Remove `Normalize` transform | Training may be slower or less stable. |

---

## Common Mistakes

1. **Forgetting `optimizer.zero_grad()`** -- Gradients accumulate, model learns wrong direction.
2. **Using `model.train()` during eval** -- Dropout and BatchNorm behave differently. Results are inconsistent.
3. **Not moving data to the right device** -- Model on GPU, data on CPU = crash.
4. **Wrong loss function** -- CrossEntropyLoss expects raw logits, not probabilities. Don't softmax before it.
5. **Evaluating on training data** -- You'll think the model is better than it is.

---

## Key Takeaways

1. The training loop is the same everywhere: forward -> loss -> backward -> update.
2. Neural networks learn by adjusting weights to minimize loss.
3. Architecture choices (layers, sizes, activations) affect performance.
4. Hyperparameters (learning rate, batch size, epochs) require tuning.
5. Always evaluate on data the model hasn't seen.

---

## Next Steps

You've seen the training loop work. But `loss.backward()` is a black box. What actually happens when gradients flow backward through the network?

That's what you'll build from scratch in **micrograd** -->
