"""DP-FedAvg federated training loop with RDP accounting.

Each local round wraps the client's optimiser in Opacus's ``PrivacyEngine``
so that each SGD step is (noise_multiplier, max_grad_norm)-DP. Client weights
are aggregated by size-weighted FedAvg at the server.
"""

from __future__ import annotations

from typing import Dict, List, Type

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


def _local_train_dp(model: nn.Module,
                    train_ds: Dataset,
                    noise_multiplier: float,
                    max_grad_norm: float,
                    epochs: int,
                    batch_size: int,
                    lr: float,
                    device: str,
                    max_physical: int = 2048) -> Dict[str, torch.Tensor]:
    """Train one client locally under DP-SGD and return its state dict."""
    model = model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    pe = PrivacyEngine()
    model_p, optim_p, loader_p = pe.make_private(
        module=model, optimizer=optim, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
    )

    for _ in range(epochs):
        with BatchMemoryManager(
            data_loader=loader_p,
            max_physical_batch_size=max_physical,
            optimizer=optim_p,
        ) as mem_loader:
            for X, y in mem_loader:
                X, y = X.to(device), y.to(device)
                optim_p.zero_grad()
                loss = criterion(model_p(X), y)
                loss.backward()
                optim_p.step()

    return {k.replace("_module.", ""): v.detach().cpu()
            for k, v in model_p.state_dict().items()}


def fedavg_round(global_state: Dict[str, torch.Tensor],
                 client_datasets: List[Dataset],
                 model_class: Type[nn.Module],
                 device: str,
                 noise_multiplier: float,
                 max_grad_norm: float,
                 local_epochs: int,
                 batch_size: int,
                 lr: float) -> Dict[str, torch.Tensor]:
    """Run one DP-FedAvg communication round across all clients."""
    client_states: list[Dict[str, torch.Tensor]] = []
    sizes: list[int] = []
    for ds in client_datasets:
        m = model_class()
        m.load_state_dict(global_state)
        state = _local_train_dp(
            m, ds, noise_multiplier, max_grad_norm,
            local_epochs, batch_size, lr, device,
        )
        client_states.append(state)
        sizes.append(len(ds))

    total = sum(sizes)
    new_state: Dict[str, torch.Tensor] = {}
    for key in client_states[0].keys():
        stacked = torch.stack([
            client_states[k][key].float() * (sizes[k] / total)
            for k in range(len(client_states))
        ], dim=0)
        new_state[key] = stacked.sum(dim=0)
    return new_state


def evaluate(model: nn.Module,
             X: torch.Tensor,
             y: torch.Tensor,
             device: str,
             threshold: float = 0.5) -> Dict[str, float]:
    """Compute the full metric suite used throughout the paper."""
    model.eval().to(device)
    with torch.no_grad():
        logits = model(X.to(device)).cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_np = y.cpu().numpy().ravel()
        pred = (probs >= threshold).astype(int)

    try:
        prob_bins, acc_bins = calibration_curve(
            y_np, probs, n_bins=10, strategy="uniform",
        )
        ece = float(np.mean(np.abs(prob_bins - acc_bins)))
    except Exception:
        ece = float("nan")

    return {
        "accuracy":  accuracy_score(y_np, pred),
        "f1":        f1_score(y_np, pred, zero_division=0),
        "precision": precision_score(y_np, pred, zero_division=0),
        "recall":    recall_score(y_np, pred, zero_division=0),
        "auc":       roc_auc_score(y_np, probs),
        "auprc":     average_precision_score(y_np, probs),
        "ece":       ece,
    }


def rdp_epsilon(noise_multiplier: float,
                sample_rate: float,
                steps: int,
                delta: float = 1e-5) -> float:
    """Bookkeeping: compute ε under RDP accounting for given hyper-parameters."""
    acct = RDPAccountant()
    for _ in range(steps):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acct.get_epsilon(delta=delta)
