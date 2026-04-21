"""Data loading, harmonisation, and stratified temporal partitioning.

The two source CSV files are the Kaggle credit-card fraud datasets:

* ULB 2013 (``creditcard.csv``)   — 284 807 rows, fraud rate ≈ 0.17 %.
* Kaggle 2023 (``creditcard_2023.csv``) — 568 630 rows, fraud rate ≈ 50 %.

Both share the same 28 PCA features (V1 … V28) plus ``Amount`` and
``Class``. We log-transform ``Amount`` and drop the time/id columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

FEATURES: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET: str = "Class"


def load_ulb(path: str | Path) -> pd.DataFrame:
    """Load ULB 2013 credit-card fraud CSV and drop the ``Time`` column."""
    df = pd.read_csv(path).drop(columns=["Time"])
    return df[FEATURES + [TARGET]]


def load_kaggle(path: str | Path) -> pd.DataFrame:
    """Load Kaggle 2023 credit-card fraud CSV and drop the ``id`` column."""
    df = pd.read_csv(path).drop(columns=["id"])
    return df[FEATURES + [TARGET]]


def harmonise_and_merge(ulb_path: str | Path,
                        kaggle_path: str | Path) -> pd.DataFrame:
    """Merge the two source datasets after log-transforming ``Amount``."""
    ulb = load_ulb(ulb_path)
    kaggle = load_kaggle(kaggle_path)
    ulb["Amount"] = np.log1p(ulb["Amount"].clip(lower=0))
    kaggle["Amount"] = np.log1p(kaggle["Amount"].clip(lower=0))
    return pd.concat([ulb, kaggle], axis=0, ignore_index=True)


def stratified_temporal_partition(df: pd.DataFrame,
                                  n_clients: int = 3,
                                  seed: int = 42) -> List[pd.DataFrame]:
    """Stratified partition across clients preserving class balance."""
    shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos = shuf[shuf[TARGET] == 1].reset_index(drop=True)
    neg = shuf[shuf[TARGET] == 0].reset_index(drop=True)
    pos_splits = np.array_split(pos, n_clients)
    neg_splits = np.array_split(neg, n_clients)
    clients: list[pd.DataFrame] = []
    for k in range(n_clients):
        c = pd.concat([pos_splits[k], neg_splits[k]], axis=0, ignore_index=True)
        c = c.sample(frac=1.0, random_state=seed + k).reset_index(drop=True)
        clients.append(c)
    return clients


def client_train_val_test_split(
    client_df: pd.DataFrame, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/10/20 stratified split per client."""
    train, tmp = train_test_split(client_df, test_size=0.30,
                                  stratify=client_df[TARGET], random_state=seed)
    val, test = train_test_split(tmp, test_size=(2 / 3),
                                 stratify=tmp[TARGET], random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def temporal_ks_validation(clients: List[pd.DataFrame]) -> pd.DataFrame:
    """Two-sample KS test on a feature aggregate between every client pair.

    Used as a sanity check that the stratified partitioning does not produce
    pathological distribution shift between clients.
    """
    results = []
    for i in range(len(clients)):
        for j in range(i + 1, len(clients)):
            a = clients[i][["V1", "V2", "V3", "V4", "V5"]].sum(axis=1).values
            b = clients[j][["V1", "V2", "V3", "V4", "V5"]].sum(axis=1).values
            D, p = ks_2samp(a, b)
            results.append({"pair": f"{i}-{j}", "D": D, "p": p})
    return pd.DataFrame(results)
