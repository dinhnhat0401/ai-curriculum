from nn import MLP
from engine import Value

# dataset (XOR-like)
xs = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]

ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)]

model = MLP(3, [4, 4, 1])

# training loop
for k in range(100):

    # forward
    ypred = [model(x) for x in xs]

    # loss
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred), Value(0.0))

    # backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in model.parameters():
        p.data -= 0.05 * p.grad

    print(k, loss.data)