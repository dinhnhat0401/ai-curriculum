# Micrograd: Backpropagation from Scratch

**Time: ~5 hours | Difficulty: The hardest and most important module | Files: `engine.py`, `nn.py`, `train.py`**

---

> "If you understand this module, you understand the core of ALL neural network training."
> -- This is not an exaggeration.

Everything in deep learning -- from a 2-layer MNIST classifier to GPT-4 with 1.8 trillion parameters -- trains using the same algorithm: backpropagation via reverse-mode automatic differentiation. This module builds that algorithm from scratch in ~100 lines of Python.

---

## What is Automatic Differentiation?

Neural networks learn by adjusting weights to minimize a loss function. To know *which direction* to adjust each weight, we need the **gradient** -- the derivative of the loss with respect to that weight.

Computing gradients by hand is possible for small networks but intractable for millions of parameters. **Automatic differentiation (autograd)** computes all gradients automatically by tracking operations in a computational graph.

### The Chain Rule

The chain rule is the mathematical foundation:

```
If   y = f(g(x))
Then dy/dx = (dy/dg) * (dg/dx)

Example:
    y = (3x + 2)^2

    Let u = 3x + 2,  so y = u^2

    dy/du = 2u = 2(3x + 2)
    du/dx = 3
    dy/dx = 2(3x + 2) * 3 = 6(3x + 2)
```

The chain rule lets us decompose complex derivatives into simple local derivatives, then multiply them together. This is exactly what backpropagation does.

### Computational Graphs

Every expression can be drawn as a graph of operations:

```
    Expression: L = (a * b + c) * d

    Computational Graph:

        a ─────┐
               ├──[*]──> e ──┐
        b ─────┘              ├──[+]──> f ──┐
                              │             ├──[*]──> L
        c ────────────────────┘             │
                                            │
        d ──────────────────────────────────┘

    Forward pass (compute values):
        e = a * b
        f = e + c
        L = f * d

    Backward pass (compute gradients):
        dL/dL = 1              (seed)
        dL/df = d              (from L = f * d)
        dL/dd = f              (from L = f * d)
        dL/de = dL/df * 1 = d  (from f = e + c, chain rule)
        dL/dc = dL/df * 1 = d  (from f = e + c, chain rule)
        dL/da = dL/de * b = db (from e = a * b, chain rule)
        dL/db = dL/de * a = da (from e = a * b, chain rule)
```

Micrograd builds exactly this: a system that constructs the graph during the forward pass and traverses it backward to compute gradients.

---

## The Value Class: Line by Line

### `__init__`: Creating a Value Node

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data           # The actual numerical value
        self.grad = 0.0            # Gradient (dLoss/dThis), initially 0
        self._backward = lambda: None  # Function to compute local gradients
        self._prev = set(_children)    # Parent nodes in the graph
        self._op = _op             # Operation that produced this node
        self.label = label         # Human-readable name (for debugging)
```

Every `Value` is a node in the computational graph. It stores:
- `data`: the number itself
- `grad`: how much the final loss changes when this value changes
- `_backward`: a closure that knows how to propagate gradients to parent nodes
- `_prev`: which nodes were combined to create this one

### `__add__`: Addition and Its Gradient

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad    # d(a+b)/da = 1, so grad flows through unchanged
        other.grad += out.grad   # d(a+b)/db = 1, same thing

    out._backward = _backward
    return out
```

**Why `+=` and not `=`?** A value might be used in multiple operations. Each path contributes to the gradient, and they must be summed. For example:

```python
a = Value(3.0)
b = a + a        # a is used TWICE
# dL/da = dL/da_left + dL/da_right (multivariate chain rule)
```

### `__mul__`: Multiplication and Its Gradient

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad   # d(a*b)/da = b
        other.grad += self.data * out.grad   # d(a*b)/db = a

    out._backward = _backward
    return out
```

The gradient of multiplication swaps the values: `d(a*b)/da = b` and `d(a*b)/db = a`. Each is multiplied by the upstream gradient `out.grad` (chain rule).

### `__pow__`: Power Rule

```python
def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
        # d(x^n)/dx = n * x^(n-1)

    out._backward = _backward
    return out
```

### `__neg__`, `__sub__`, `__truediv__`: Built from Primitives

```python
def __neg__(self):      return self * -1        # negation = multiply by -1
def __sub__(self, other): return self + (-other) # subtraction = add negative
def __truediv__(self, other): return self * other**-1  # division = multiply by inverse
```

This is elegant: only `+`, `*`, and `**` need gradient implementations. Everything else is composed from them.

### `__radd__`, `__rmul__`: Reverse Operations

```python
def __radd__(self, other): return self + other
def __rmul__(self, other): return self * other
```

These handle `2 + Value(3)` -- Python calls `int.__add__(2, Value(3))` first, which fails, then tries `Value.__radd__(Value(3), 2)`.

### `tanh()`: Activation Function

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad
        # d(tanh(x))/dx = 1 - tanh(x)^2

    out._backward = _backward
    return out
```

### `backward()`: The Topological Sort

```python
def backward(self):
    topo = []
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

    build(self)

    self.grad = 1.0                    # Seed: dL/dL = 1
    for node in reversed(topo):        # Process in reverse order
        node._backward()
```

**Why topological sort?** We must process nodes so that a node's gradient is fully computed before we propagate through it. Topological order guarantees: all consumers of a value are processed before the value itself.

**Why reversed?** Topological sort gives forward order (inputs first). We need backward order (output first). So we reverse.

**Why `self.grad = 1.0`?** The seed gradient. If `L` is the loss, `dL/dL = 1`. This is where backpropagation starts.

---

## The Neural Network (`nn.py`)

### Neuron

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # random weights
        self.b = Value(0.0)                                          # bias

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)     # weighted sum
        return act.tanh()                                             # activation

    def parameters(self):
        return self.w + [self.b]                                      # all trainable params
```

A neuron computes: `output = tanh(w1*x1 + w2*x2 + ... + wn*xn + b)`

### Layer and MLP

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    # Each neuron in the layer sees the same input but has different weights

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    # MLP(3, [4, 4, 1]) = 3 inputs -> 4 hidden -> 4 hidden -> 1 output
```

The `parameters()` method on each class collects all trainable `Value` objects. This is what the optimizer uses to know what to update.

---

## The Training Loop (`train.py`)

```python
model = MLP(3, [4, 4, 1])

for k in range(100):
    # 1. Forward pass
    ypred = [model(x) for x in xs]

    # 2. Compute loss (mean squared error)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 3. Zero gradients (CRITICAL)
    for p in model.parameters():
        p.grad = 0.0

    # 4. Backward pass
    loss.backward()

    # 5. Update weights (gradient descent)
    for p in model.parameters():
        p.data -= 0.05 * p.grad     # step in the negative gradient direction
```

**Why minus?** The gradient points in the direction of steepest *increase*. We want to *decrease* loss, so we go in the opposite direction.

**Why zero gradients?** Gradients accumulate (via `+=`). Without zeroing, each iteration adds to the previous gradients, giving a completely wrong update direction.

---

## Worked Example

Let's trace a tiny network by hand. Network: 1 input, 1 hidden neuron, 1 output.

```
Input: x = 2.0
Weights: w1 = 0.5, w2 = 0.3
Biases: b1 = 0.1, b2 = 0.2
Target: y = 1.0

Forward:
    h = tanh(w1 * x + b1) = tanh(0.5 * 2.0 + 0.1) = tanh(1.1) = 0.7999
    o = tanh(w2 * h + b2) = tanh(0.3 * 0.7999 + 0.2) = tanh(0.4400) = 0.4136
    L = (o - y)^2 = (0.4136 - 1.0)^2 = 0.3438

Backward:
    dL/dL = 1.0
    dL/do = 2 * (o - y) = 2 * (0.4136 - 1.0) = -1.1728
    dL/d(w2*h+b2) = dL/do * (1 - o^2) = -1.1728 * (1 - 0.4136^2) = -0.9721
    dL/dw2 = -0.9721 * h = -0.9721 * 0.7999 = -0.7776
    dL/db2 = -0.9721 * 1 = -0.9721
    dL/dh = -0.9721 * w2 = -0.9721 * 0.3 = -0.2916
    dL/d(w1*x+b1) = dL/dh * (1 - h^2) = -0.2916 * (1 - 0.7999^2) = -0.1048
    dL/dw1 = -0.1048 * x = -0.1048 * 2.0 = -0.2096
    dL/db1 = -0.1048

Update (lr = 0.1):
    w1 = 0.5 - 0.1 * (-0.2096) = 0.5210  (increased -- good, loss will decrease)
    w2 = 0.3 - 0.1 * (-0.7776) = 0.3778  (increased -- pushing output closer to 1.0)
    b1 = 0.1 - 0.1 * (-0.1048) = 0.1105
    b2 = 0.2 - 0.1 * (-0.9721) = 0.2972
```

This is exactly what micrograd does automatically. Every `Value._backward()` computes one step of this chain.

---

## Exercises

1. **Add ReLU activation** -- Implement `relu()` on the `Value` class. ReLU(x) = max(0, x). The gradient: 1 if x > 0, else 0.

2. **Add sigmoid activation** -- sigmoid(x) = 1 / (1 + exp(-x)). Gradient: sigmoid(x) * (1 - sigmoid(x)).

3. **Implement MSE loss as a function** -- Take two lists of Values, return mean squared error as a Value.

4. **Train on XOR** -- The classic: inputs `[[0,0],[0,1],[1,0],[1,1]]`, outputs `[0,1,1,0]`. Can your MLP learn it?

5. **Visualize the computational graph** -- Using graphviz, draw the graph for a small computation. Starter:
   ```python
   from graphviz import Digraph
   def draw_dot(root):
       # Trace all nodes and edges, create a Digraph
       pass
   ```

6. **Compare with PyTorch** -- Build the exact same network in PyTorch. Verify the gradients match:
   ```python
   import torch
   x = torch.tensor([2.0, 3.0, -1.0], requires_grad=True)
   # ... build equivalent network, compare .grad values
   ```

7. **Learning rate experiment** -- Train the same network with lr=0.001, 0.01, 0.05, 0.1, 0.5. Plot loss curves. What happens at each?

8. **Implement a simple SGD class** -- Instead of `p.data -= lr * p.grad`, create an `SGD` class with `step()` and `zero_grad()` methods. Then add momentum.

9. **Gradient checking** -- For each parameter, compute the numerical gradient: `(f(x+h) - f(x-h)) / 2h` with h=1e-5. Compare with the analytical gradient from backprop. They should match to ~1e-5.

10. **Add L2 regularization** -- Add `lambda * sum(p.data**2 for p in params)` to the loss. What happens to the weights during training?

---

## How This Connects to PyTorch

| Micrograd | PyTorch | Notes |
|-----------|---------|-------|
| `Value(3.0)` | `torch.tensor(3.0, requires_grad=True)` | Same concept, PyTorch handles tensors (multi-dimensional) |
| `value.data` | `tensor.data` or `tensor.item()` | The raw number |
| `value.grad` | `tensor.grad` | The gradient |
| `value.backward()` | `tensor.backward()` | Topological sort + chain rule |
| `p.grad = 0.0` | `optimizer.zero_grad()` | Same operation, different API |
| `p.data -= lr * p.grad` | `optimizer.step()` | Same update, PyTorch wraps it |

PyTorch's autograd is micrograd scaled up to handle:
- Tensors (not just scalars)
- GPU acceleration
- Hundreds of operations (not just +, *, pow, tanh)
- Memory optimization (gradient checkpointing)
- Distributed training

But the core algorithm is identical.

---

## Common Bugs and Pitfalls

1. **Forgetting to zero gradients** -- Gradients accumulate. After 10 iterations without zeroing, your gradients are the sum of 10 iterations' gradients. The model will not converge.

2. **Not understanding `+=`** -- If you use `=` instead of `+=` in backward functions, multi-path gradients will be wrong. Any value used more than once will have incorrect gradients.

3. **Numerical instability** -- `exp(1000)` is infinity. `tanh(1000)` is fine (it saturates at 1), but intermediate `exp(2*1000)` will overflow. Production autograd engines handle this; micrograd doesn't.

4. **Wrong learning rate** -- Too high: loss oscillates or diverges. Too low: loss barely decreases after 1000 iterations. Start with 0.01-0.1 for micrograd.

---

## Key Takeaways

1. **Backpropagation = chain rule + topological sort.** That's it. No magic.
2. **Every operation records how to compute its local gradient.** The `_backward` closures capture this.
3. **Gradients flow backward from output to input.** The loss gradient is 1.0, and it propagates through the graph.
4. **Gradient accumulation (`+=`) handles multi-path graphs.** Values used multiple times get correct gradients.
5. **This is what PyTorch does.** `torch.Tensor` is a scaled-up `Value`. `loss.backward()` is this exact algorithm.

---

## Further Reading

- **Video**: Karpathy "The spelled-out intro to neural networks and backpropagation"
- **Code**: github.com/karpathy/micrograd
- **Notes**: CS231n Backpropagation notes (Stanford)
- **Paper**: "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al.)

---

Next up: **Makemore** -- where you'll use these concepts to build language models and learn about embeddings -->
