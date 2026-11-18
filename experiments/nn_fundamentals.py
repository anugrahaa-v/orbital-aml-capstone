"""Neuron + activation functions (NumPy reference implementation)."""
from __future__ import annotations
import numpy as np


def binary_step(x): return (np.asarray(x, float) >= 0).astype(float)
def linear(x): return np.asarray(x, float)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -500, 500)))
def bipolar_sigmoid(x): return 2.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -500, 500))) - 1.0
def tanh(x): return np.tanh(np.asarray(x, float))
def relu(x): return np.maximum(0.0, np.asarray(x, float))
def leaky_relu(x, alpha=0.01):
    x = np.asarray(x, float); return np.where(x > 0, x, alpha * x)
def softmax(x):
    x = np.asarray(x, float); e = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


ACTIVATIONS = {"binary_step": binary_step, "linear": linear, "sigmoid": sigmoid, "bipolar_sigmoid": bipolar_sigmoid,
               "tanh": tanh, "relu": relu, "leaky_relu": leaky_relu, "softmax": softmax}


def neuron(inputs, weights, bias, activation: str, alpha: float = 0.01) -> dict:
    x, w = np.asarray(inputs, float), np.asarray(weights, float)
    if x.shape != w.shape:
        raise ValueError("inputs and weights must have the same length")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}")
    terms = (x * w).tolist()
    net = float(np.dot(x, w) + bias)
    f = ACTIVATIONS[activation]
    out = f(np.array([net]), alpha) if activation == "leaky_relu" else f(np.array([net]))
    return {"terms": terms, "net_input": net, "activation": activation, "output": float(out[0]),
            "formula": f"net = sum(w_i * x_i) + b = {net:.6g}"}


def curve(activation: str, lo=-6.0, hi=6.0, n=241) -> dict:
    xs = np.linspace(lo, hi, n)
    if activation == "softmax":      # softmax over [x, 0]: the 2-class logistic view
        ys = np.array([softmax(np.array([x, 0.0]))[0] for x in xs])
    else:
        ys = ACTIVATIONS[activation](xs)
    return {"x": xs.tolist(), "y": np.asarray(ys).tolist()}
