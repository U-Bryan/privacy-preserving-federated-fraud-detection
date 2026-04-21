"""Privacy attacks (MIA, model inversion) and two defence baselines.

These routines are used exclusively in Phase C to reproduce the privacy
evaluation figures and tables.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def mia_attack(target_model: torch.nn.Module,
               member_X: torch.Tensor,
               nonmember_X: torch.Tensor,
               device: str,
               seed: int = 0) -> float:
    """Confidence-based black-box membership-inference attack.

    Trains a gradient-boosted classifier on the target model's prediction
    confidence to separate members from non-members. Returns attack accuracy.
    Chance level = 0.5.
    """
    target_model.eval().to(device)
    with torch.no_grad():
        in_logits = target_model(member_X.to(device)).cpu().numpy().ravel()
        out_logits = target_model(nonmember_X.to(device)).cpu().numpy().ravel()
        in_probs = 1.0 / (1.0 + np.exp(-in_logits))
        out_probs = 1.0 / (1.0 + np.exp(-out_logits))

    X = np.concatenate([in_probs, out_probs]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(in_probs)), np.zeros(len(out_probs))])
    idx = np.random.RandomState(seed).permutation(len(X))
    split = len(X) // 2
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def model_inversion_attack(model: torch.nn.Module,
                           target_feature_dim: int,
                           n_samples: int = 100,
                           steps: int = 500,
                           lr: float = 0.05,
                           device: str = "cuda") -> np.ndarray:
    """Optimisation-based model-inversion attack against the fraud classifier.

    Attempts to recover input features that maximise the positive-class score.
    Returns an array of shape ``(n_samples, target_feature_dim)``.
    """
    model.eval().to(device)
    recons: list[np.ndarray] = []
    for _ in range(n_samples):
        x = torch.randn(1, target_feature_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            logit = model(x)
            loss = -logit.mean()
            loss.backward()
            opt.step()
        recons.append(x.detach().cpu().numpy().ravel())
    return np.array(recons)


def label_smoothing_loss(logits: torch.Tensor,
                         targets: torch.Tensor,
                         alpha: float = 0.1) -> torch.Tensor:
    """Binary label-smoothed cross-entropy (MIA defence baseline)."""
    smooth = targets * (1 - alpha) + 0.5 * alpha
    return F.binary_cross_entropy_with_logits(logits, smooth)


def memguard_predict(model: torch.nn.Module,
                     X: torch.Tensor,
                     device: str,
                     eps: float = 0.1) -> np.ndarray:
    """Simple post-hoc MemGuard-style perturbation of prediction scores."""
    model.eval().to(device)
    with torch.no_grad():
        logits = model(X.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    noise = np.random.uniform(-eps, eps, size=probs.shape)
    return np.clip(probs + noise, 1e-6, 1 - 1e-6)
