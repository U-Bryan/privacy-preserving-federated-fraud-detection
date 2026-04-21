#!/usr/bin/env python
"""Phase C — privacy attacks (MIA defences + model inversion).

Compares MIA attack accuracy across five defence conditions and measures
model-inversion reconstruction error at four privacy budgets. Produces the
data behind the MIA-defence table and Figure 7 in the paper.

    python scripts/run_phase_c.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader, TensorDataset

from fraud_fl.attacks import (
    label_smoothing_loss,
    memguard_predict,
    mia_attack,
    model_inversion_attack,
)
from fraud_fl.data import FEATURES, TARGET, client_train_val_test_split
from fraud_fl.federated import evaluate, fedavg_round
from fraud_fl.models import FraudMLP, to_tensor
from fraud_fl.utils import (
    ensure_dirs,
    gpu_info,
    load_config,
    resolve_device,
    set_seed,
    setup_logging,
)


def _train_plain_real(splits, device, epochs=5, lr=1e-3,
                      smoothing_alpha=0.0, adv_reg=False):
    """Train a non-DP baseline on real data (with optional defences)."""
    set_seed(0)
    m = FraudMLP().to(device)
    X, y = to_tensor(pd.concat([tr for (tr, _, _) in splits]), FEATURES, TARGET)
    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = m(xb)
            if adv_reg:
                ce = F.binary_cross_entropy_with_logits(logits, yb)
                conf_penalty = (torch.sigmoid(logits) ** 2).mean()
                loss = ce - 0.1 * conf_penalty
            elif smoothing_alpha > 0:
                loss = label_smoothing_loss(logits, yb, alpha=smoothing_alpha)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
    return m


def _train_dp_fedavg_real(splits, cfg, device):
    """DP-FedAvg trained directly on real data (no synthetic substitution)."""
    set_seed(0)
    client_ds = [TensorDataset(*to_tensor(tr, FEATURES, TARGET))
                 for (tr, _, _) in splits]
    m = FraudMLP().to(device)
    state = m.state_dict()
    for _ in range(cfg["federated"]["rounds"]):
        state = fedavg_round(
            state, client_ds, FraudMLP, device,
            cfg["federated"]["noise_multiplier"],
            cfg["federated"]["max_grad_norm"],
            cfg["federated"]["local_epochs"],
            cfg["federated"]["batch_size"],
            cfg["federated"]["lr"],
        )
    m.load_state_dict(state)
    return m


def _our_framework(cfg, device, ckpt):
    """Full proposed framework: DP-CTGAN synthetic data + DP-FedAvg."""
    set_seed(7)
    p_seed = cfg["phase_c"]["partition_seed"]
    synth_ds = [
        TensorDataset(*to_tensor(
            pd.read_parquet(ckpt / "synthetic" / f"p{p_seed}_c{k}.parquet"),
            FEATURES, TARGET,
        ))
        for k in range(cfg["data"]["n_clients"])
    ]
    m = FraudMLP().to(device)
    state = m.state_dict()
    for _ in range(cfg["federated"]["rounds"]):
        state = fedavg_round(
            state, synth_ds, FraudMLP, device,
            cfg["federated"]["noise_multiplier"],
            cfg["federated"]["max_grad_norm"],
            cfg["federated"]["local_epochs"],
            cfg["federated"]["batch_size"],
            cfg["federated"]["lr"],
        )
    m.load_state_dict(state)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = Path(cfg["paths"]["checkpoints"])
    result = Path(cfg["paths"]["results"]) / "phase_C_attacks"
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "phase_c.log")
    logger.info("Device: %s", gpu_info())

    p_seed = cfg["phase_c"]["partition_seed"]
    clients = pickle.load(
        open(ckpt / "partitions" / f"partition_{p_seed}.pkl", "rb")
    )
    splits = [client_train_val_test_split(c, seed=p_seed) for c in clients]

    member_X, _ = to_tensor(
        pd.concat([tr for (tr, _, _) in splits]), FEATURES, TARGET,
    )
    nonmember_X, nonmember_y = to_tensor(
        pd.concat([te for (_, _, te) in splits]), FEATURES, TARGET,
    )

    # -------- MIA defence comparison --------
    conditions = {
        "no_defence":      lambda: _train_plain_real(splits, device),
        "label_smoothing": lambda: _train_plain_real(splits, device,
                                                     smoothing_alpha=0.1),
        "adv_reg":         lambda: _train_plain_real(splits, device,
                                                     adv_reg=True),
        "dp_fedavg_only":  lambda: _train_dp_fedavg_real(splits, cfg, device),
        "proposed":        lambda: _our_framework(cfg, device, ckpt),
    }
    mia_n = cfg["phase_c"]["mia_sample_size"]
    mia_results: list[dict] = []
    for name, builder in conditions.items():
        out = result / f"{name}.json"
        if out.exists():
            mia_results.append(json.load(open(out)))
            continue
        logger.info("[%s] training...", name)
        t0 = time.time()
        m = builder()
        idx_m = np.random.RandomState(0).choice(len(member_X),    mia_n, replace=False)
        idx_n = np.random.RandomState(1).choice(len(nonmember_X), mia_n, replace=False)
        mia_acc = mia_attack(m, member_X[idx_m], nonmember_X[idx_n], device)
        metrics = evaluate(m, nonmember_X, nonmember_y, device)
        rec = dict(
            condition=name, mia_accuracy=mia_acc,
            wall_clock_min=(time.time() - t0) / 60, **metrics,
        )
        json.dump(rec, open(out, "w"), indent=2)
        mia_results.append(rec)
        logger.info("  MIA acc = %.3f, F1 = %.4f  (%.1f min)",
                    mia_acc, metrics["f1"], rec["wall_clock_min"])

    # MemGuard post-hoc defence (applied to no-defence baseline)
    m_nd = _train_plain_real(splits, device)
    probs = memguard_predict(m_nd, member_X[:mia_n], device)
    probs_n = memguard_predict(m_nd, nonmember_X[:mia_n], device)
    X = np.concatenate([probs.ravel(), probs_n.ravel()]).reshape(-1, 1)
    y = np.concatenate([np.ones(mia_n), np.zeros(mia_n)])
    idx = np.random.RandomState(0).permutation(len(X))
    split = len(X) // 2
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
    clf.fit(X[idx[:split]], y[idx[:split]])
    mg_acc = clf.score(X[idx[split:]], y[idx[split:]])
    metrics_nd = evaluate(m_nd, nonmember_X, nonmember_y, device)
    mg_rec = dict(condition="memguard_posthoc",
                  mia_accuracy=float(mg_acc), **metrics_nd)
    json.dump(mg_rec, open(result / "memguard_posthoc.json", "w"), indent=2)
    mia_results.append(mg_rec)
    pd.DataFrame(mia_results).to_csv(result / "mia_defences.csv", index=False)

    # -------- Model inversion at multiple ε --------
    inv_results: dict[str, dict] = {}
    for eps_name, nm in cfg["phase_c"]["inversion_eps_noise"].items():
        if nm is None:  # "inf" → no DP
            m = _train_plain_real(splits, device)
        else:
            set_seed(0)
            synth_ds = [
                TensorDataset(*to_tensor(
                    pd.read_parquet(ckpt / "synthetic"
                                    / f"p{p_seed}_c{k}.parquet"),
                    FEATURES, TARGET,
                ))
                for k in range(cfg["data"]["n_clients"])
            ]
            m = FraudMLP().to(device)
            state = m.state_dict()
            for _ in range(cfg["federated"]["rounds"]):
                state = fedavg_round(
                    state, synth_ds, FraudMLP, device,
                    nm, cfg["federated"]["max_grad_norm"],
                    cfg["federated"]["local_epochs"],
                    cfg["federated"]["batch_size"],
                    cfg["federated"]["lr"],
                )
            m.load_state_dict(state)

        recons = model_inversion_attack(
            m, target_feature_dim=len(FEATURES),
            n_samples=cfg["phase_c"]["inversion_n_samples"],
            steps=cfg["phase_c"]["inversion_steps"],
            lr=cfg["phase_c"]["inversion_lr"],
            device=device,
        )
        _, _member_y = to_tensor(
            pd.concat([tr for (tr, _, _) in splits]), FEATURES, TARGET,
        )
        pos_ref = member_X[_member_y.ravel().long().bool()].numpy().mean(axis=0)
        errs = np.linalg.norm(recons - pos_ref, axis=1)
        inv_results[eps_name] = {
            "mean_recon_error": float(errs.mean()),
            "std_recon_error":  float(errs.std()),
            "reconstructions":  recons[:5].tolist(),
        }
        logger.info("inversion ε=%s: mean err = %.3f", eps_name, errs.mean())

    json.dump(inv_results, open(result / "inversion_results.json", "w"), indent=2)
    logger.info("Phase C complete — results in %s", result)


if __name__ == "__main__":
    main()
