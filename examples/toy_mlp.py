"""Toy MLP example (NumPy).

Simple 1-hidden-layer neural network trained on random data to illustrate
forward/backward passes and SGD. Educational only.
"""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    N = probs.shape[0]
    return -np.log(probs[np.arange(N), y] + 1e-12).mean()


def one_hot(y, K):
    oh = np.zeros((y.size, K))
    oh[np.arange(y.size), y] = 1.0
    return oh


def train():
    np.random.seed(0)
    N, D, H, K = 200, 20, 50, 3  # samples, input dim, hidden, classes

    X = np.random.randn(N, D)
    y = np.random.randint(0, K, size=N)

    W1 = 0.01 * np.random.randn(D, H)
    b1 = np.zeros(H)
    W2 = 0.01 * np.random.randn(H, K)
    b2 = np.zeros(K)

    lr = 1e-1
    epochs = 200

    for epoch in range(epochs):
        # forward
        h = X.dot(W1) + b1
        h_relu = relu(h)
        scores = h_relu.dot(W2) + b2
        probs = softmax(scores)

        loss = cross_entropy_loss(probs, y)

        if epoch % 50 == 0:
            preds = probs.argmax(axis=1)
            acc = (preds == y).mean()
            print(f"epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.3f}")

        # backward
        Nbatch = N
        dscores = probs.copy()
        dscores[np.arange(Nbatch), y] -= 1
        dscores /= Nbatch

        dW2 = h_relu.T.dot(dscores)
        db2 = dscores.sum(axis=0)

        dh = dscores.dot(W2.T)
        dh_raw = dh * relu_deriv(h)

        dW1 = X.T.dot(dh_raw)
        db1 = dh_raw.sum(axis=0)

        # update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2


if __name__ == "__main__":
    train()
