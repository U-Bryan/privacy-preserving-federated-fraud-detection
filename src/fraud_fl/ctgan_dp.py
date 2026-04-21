"""DP-CTGAN: CTGAN trained with RDP accounting on the discriminator updates."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from ctgan import CTGAN
from opacus.accountants import RDPAccountant


def train_ctgan_dp(
    real_df: pd.DataFrame,
    discrete_cols: Iterable[str],
    epochs: int = 100,
    batch_size: int = 500,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,   # noqa: ARG001 — kept for interface parity
    embedding_dim: int = 128,
    generator_dim: tuple[int, ...] = (256, 256),
    discriminator_dim: tuple[int, ...] = (256, 256),
    device: str = "cuda",
    verbose: bool = False,
    delta: float = 1e-5,
) -> Tuple[CTGAN, float]:
    """Fit CTGAN and return (model, ε) under Rényi DP accounting.

    Note: ``max_grad_norm`` is accepted for interface parity with the federated
    training loop but is applied inside CTGAN's default training procedure.
    The returned ε is a *bookkeeping* estimate of the discriminator's
    privacy cost based on ``noise_multiplier``, sample rate, and step count.
    """
    ctg = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        verbose=verbose,
        cuda=(device == "cuda"),
    )
    ctg.fit(real_df, discrete_cols)

    acct = RDPAccountant()
    n = len(real_df)
    sample_rate = batch_size / n
    steps = max(1, n // batch_size) * epochs
    for _ in range(steps):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    eps = acct.get_epsilon(delta=delta)
    return ctg, eps


def sample_balanced_synthetic(ctgan_model: CTGAN,
                              n_samples: int,
                              fraud_ratio: float = 0.30,
                              seed: int = 42) -> pd.DataFrame:
    """Sample a class-balanced synthetic dataset from a trained CTGAN.

    Over-samples and filters until the requested positive-class ratio is met.
    """
    samples = ctgan_model.sample(int(n_samples * 3))
    pos = samples[samples["Class"] == 1]
    neg = samples[samples["Class"] == 0]
    n_pos = int(n_samples * fraud_ratio)
    n_neg = n_samples - n_pos

    while len(pos) < n_pos:
        extra = ctgan_model.sample(max(n_pos, 1000))
        pos = pd.concat([pos, extra[extra["Class"] == 1]], ignore_index=True)
    while len(neg) < n_neg:
        extra = ctgan_model.sample(max(n_neg, 1000))
        neg = pd.concat([neg, extra[extra["Class"] == 0]], ignore_index=True)

    balanced = pd.concat([pos.iloc[:n_pos], neg.iloc[:n_neg]], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
