"""Fast, GPU-free unit tests for the pure-function parts of the library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fraud_fl.data import (
    FEATURES,
    TARGET,
    client_train_val_test_split,
    stratified_temporal_partition,
)
from fraud_fl.metrics import (
    c2st_accuracy,
    correlation_frobenius,
    js_divergence,
    mmd_rbf,
    per_feature_kl,
)
from fraud_fl.models import FraudMLP, to_tensor
from fraud_fl.utils import set_seed


# --------------------------------------------------------------------------
# Data-pipeline invariants
# --------------------------------------------------------------------------
def _fake_df(n=2000, pos_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 29))
    y = (rng.random(n) < pos_frac).astype(int)
    cols = FEATURES + [TARGET]
    return pd.DataFrame(np.c_[X, y], columns=cols)


def test_stratified_temporal_partition_preserves_class_ratio():
    df = _fake_df(n=3000, pos_frac=0.2, seed=1)
    clients = stratified_temporal_partition(df, n_clients=3, seed=42)

    assert len(clients) == 3
    assert sum(len(c) for c in clients) == len(df)
    global_ratio = df[TARGET].mean()
    for c in clients:
        # each client's positive ratio should be within ±1 pp of global
        assert abs(c[TARGET].mean() - global_ratio) < 0.01


def test_client_train_val_test_split_sums_to_whole():
    df = _fake_df(n=1000, pos_frac=0.15, seed=2)
    train, val, test = client_train_val_test_split(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    # approximate 70 / 10 / 20 split
    assert 0.68 < len(train) / len(df) < 0.72
    assert 0.08 < len(val) / len(df)   < 0.12
    assert 0.18 < len(test) / len(df)  < 0.22


# --------------------------------------------------------------------------
# Model forward-pass shape
# --------------------------------------------------------------------------
def test_fraud_mlp_forward_shape():
    set_seed(0)
    m = FraudMLP()
    X = torch.randn(8, 29)
    out = m(X)
    assert out.shape == (8, 1)


def test_fraud_mlp_param_count_nonzero():
    m = FraudMLP()
    assert m.count_params() > 0


def test_to_tensor_round_trip():
    df = _fake_df(n=64, seed=3)
    X, y = to_tensor(df, FEATURES, TARGET)
    assert X.shape == (64, 29)
    assert y.shape == (64, 1)
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


# --------------------------------------------------------------------------
# Fidelity metrics behave sensibly on random data
# --------------------------------------------------------------------------
@pytest.fixture
def real_synth_pair():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, scale=1.0, size=(500, 10))
    synth = rng.normal(loc=0.0, scale=1.0, size=(500, 10))
    return real, synth


def test_mmd_rbf_nonnegative_and_small_for_same_distribution(real_synth_pair):
    real, synth = real_synth_pair
    val = mmd_rbf(real, synth, gamma=0.1, max_n=200)
    assert val > -1e-6            # MMD² is non-negative up to numerical noise
    assert abs(val) < 0.1         # two samples of the same distribution are close


def test_per_feature_kl_nonnegative(real_synth_pair):
    real, synth = real_synth_pair
    mean_kl, max_kl = per_feature_kl(real, synth)
    assert mean_kl >= 0.0
    assert max_kl >= mean_kl


def test_js_divergence_bounded(real_synth_pair):
    real, synth = real_synth_pair
    val = js_divergence(real, synth)
    assert 0.0 <= val <= 1.0     # JS(nats) is bounded by ln(2) ≈ 0.693


def test_c2st_near_chance_for_same_distribution(real_synth_pair):
    real, synth = real_synth_pair
    acc = c2st_accuracy(real, synth)
    # A logistic classifier cannot meaningfully separate samples from the
    # same distribution — accuracy should be close to chance.
    assert 0.4 < acc < 0.6


def test_correlation_frobenius_small_for_same_distribution(real_synth_pair):
    real, synth = real_synth_pair
    val = correlation_frobenius(real, synth)
    assert val >= 0.0
    assert val < 0.5
