"""Synthetic-data fidelity metrics used in Phase D.

All functions operate on raw NumPy arrays of shape ``(n_rows, n_features)``
so they can be applied to any real / synthetic pair, not just the fraud
dataset. MMD, KL, JS and correlation-Frobenius measure distributional
similarity; C2ST is a classifier two-sample test (0.5 ≈ indistinguishable).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split


def mmd_rbf(X: np.ndarray, Y: np.ndarray,
            gamma: float = 1.0, max_n: int = 5000, seed: int = 0) -> float:
    """Unbiased squared Maximum Mean Discrepancy with an RBF kernel."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    Kxx = rbf_kernel(X, X, gamma=gamma)
    np.fill_diagonal(Kxx, 0)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    np.fill_diagonal(Kyy, 0)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    n, m = len(X), len(Y)
    mmd2 = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()
    return float(mmd2)


def per_feature_kl(real: np.ndarray, synth: np.ndarray,
                   bins: int = 50, eps: float = 1e-6) -> tuple[float, float]:
    """Mean and max of per-feature KL divergence (real || synth)."""
    kls = []
    for j in range(real.shape[1]):
        lo = min(real[:, j].min(), synth[:, j].min())
        hi = max(real[:, j].max(), synth[:, j].max())
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(real[:, j],  bins=edges, density=True)
        q, _ = np.histogram(synth[:, j], bins=edges, density=True)
        p = p + eps
        p /= p.sum()
        q = q + eps
        q /= q.sum()
        kls.append(float(entropy(p, q)))
    return float(np.mean(kls)), float(np.max(kls))


def c2st_accuracy(real: np.ndarray, synth: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test accuracy (0.5 ≈ indistinguishable)."""
    X = np.vstack([real, synth])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed,
    )
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def correlation_frobenius(real: np.ndarray, synth: np.ndarray) -> float:
    """Relative Frobenius norm of the correlation-matrix difference."""
    Cr = np.corrcoef(real.T)
    Cs = np.corrcoef(synth.T)
    num = np.linalg.norm(Cr - Cs, ord="fro")
    den = np.linalg.norm(Cr, ord="fro") + 1e-12
    return float(num / den)


def js_divergence(real: np.ndarray, synth: np.ndarray,
                  bins: int = 50, eps: float = 1e-6) -> float:
    """Mean Jensen–Shannon divergence across features."""
    js = []
    for j in range(real.shape[1]):
        lo = min(real[:, j].min(), synth[:, j].min())
        hi = max(real[:, j].max(), synth[:, j].max())
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(real[:, j],  bins=edges, density=True)
        q, _ = np.histogram(synth[:, j], bins=edges, density=True)
        p += eps
        p /= p.sum()
        q += eps
        q /= q.sum()
        m = 0.5 * (p + q)
        js.append(0.5 * entropy(p, m) + 0.5 * entropy(q, m))
    return float(np.mean(js))
