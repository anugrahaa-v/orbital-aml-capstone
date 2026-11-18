"""From-scratch MLP (NumPy): forward propagation, loss, backpropagation, gradient descent, weight updates."""
from __future__ import annotations
import numpy as np
from .nn_fundamentals import sigmoid


def _act(name):
    if name == "relu": return (lambda z: np.maximum(0, z)), (lambda z, a: (z > 0).astype(float))
    if name == "tanh": return np.tanh, (lambda z, a: 1 - a ** 2)
    if name == "sigmoid": return sigmoid, (lambda z, a: a * (1 - a))
    if name == "bipolar_sigmoid":
        return (lambda z: 2 * sigmoid(z) - 1), (lambda z, a: 0.5 * (1 + a) * (1 - a))
    raise ValueError(name)


class MLP:
    def __init__(self, layer_sizes, hidden="relu", task="regression", lr=0.01, l2=0.0, seed=0):
        assert task in ("regression", "classification")
        self.sizes, self.task, self.lr, self.l2 = list(layer_sizes), task, lr, l2
        self.f, self.df = _act(hidden)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            scale = np.sqrt(2.0 / a) if hidden == "relu" else np.sqrt(1.0 / a)
            self.W.append(rng.normal(0, scale, (a, c))); self.b.append(np.zeros(c))

    # ---- forward ----
    def forward(self, X):
        A, Z = [np.asarray(X, float)], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            Z.append(z)
            if i < len(self.W) - 1:
                A.append(self.f(z))
            else:
                if self.task == "classification":
                    A.append(sigmoid(z) if self.sizes[-1] == 1 else _softmax(z))
                else:
                    A.append(z)
        return A, Z

    def loss(self, X, y):
        A, _ = self.forward(X); p = A[-1]; y = self._y(y)
        if self.task == "regression":
            data = 0.5 * np.mean((p - y) ** 2)
        elif self.sizes[-1] == 1:
            pp = np.clip(p, 1e-12, 1 - 1e-12); data = -np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))
        else:
            data = -np.mean(np.sum(y * np.log(np.clip(p, 1e-12, 1)), axis=1))
        return float(data + 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.W))

    def _y(self, y):
        y = np.asarray(y, float)
        if self.task == "classification" and self.sizes[-1] > 1 and y.ndim == 1:
            return np.eye(self.sizes[-1])[y.astype(int)]
        return y.reshape(-1, 1) if y.ndim == 1 else y

    # ---- backprop ----
    def gradients(self, X, y):
        A, Z = self.forward(X); y = self._y(y); m = len(A[0])
        delta = (A[-1] - y) / m          # MSE(0.5*) and sigmoid/softmax+cross-entropy share this form
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = A[i].T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.df(Z[i - 1], A[i])
        return gW, gb

    def step(self, X, y):
        gW, gb = self.gradients(X, y)
        for i in range(len(self.W)):
            self.W[i] -= self.lr * gW[i]; self.b[i] -= self.lr * gb[i]      # gradient-descent weight update

    def fit(self, X, y, epochs=200, batch_size=64, Xval=None, yval=None, seed=0, shuffle=True):
        rng = np.random.default_rng(seed); n = len(X); hist = {"train_loss": [], "val_loss": []}
        for _ in range(epochs):
            idx = rng.permutation(n) if shuffle else np.arange(n)
            for s in range(0, n, batch_size):
                j = idx[s:s + batch_size]; self.step(X[j], np.asarray(y)[j])
            hist["train_loss"].append(self.loss(X, y))
            if Xval is not None:
                hist["val_loss"].append(self.loss(Xval, yval))
        return hist

    def predict(self, X):
        p = self.forward(X)[0][-1]
        if self.task == "classification":
            return (p[:, 0] >= 0.5).astype(int) if self.sizes[-1] == 1 else p.argmax(1)
        return p[:, 0] if p.shape[1] == 1 else p


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)


def gradient_check(net: MLP, X, y, eps=1e-6) -> float:
    """Max relative error between analytic and numerical gradients (should be ~1e-7 or below)."""
    gW, _ = net.gradients(X, y); worst = 0.0
    for li, W in enumerate(net.W):
        for _ in range(5):
            i, j = np.random.randint(W.shape[0]), np.random.randint(W.shape[1])
            old = W[i, j]
            W[i, j] = old + eps; lp = net.loss(X, y)
            W[i, j] = old - eps; lm = net.loss(X, y); W[i, j] = old
            num = (lp - lm) / (2 * eps); ana = gW[li][i, j]
            worst = max(worst, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
    return worst
