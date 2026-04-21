#!/usr/bin/env python
"""Smoke test: run the full pipeline at a small scale.

    python scripts/smoke_test.py --config configs/smoke.yaml

If this completes without error, every component is wired correctly and the
full campaign will run. Results are not numerically meaningful.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from torch.utils.data import TensorDataset

from fraud_fl.ctgan_dp import sample_balanced_synthetic, train_ctgan_dp
from fraud_fl.data import (
    FEATURES,
    TARGET,
    client_train_val_test_split,
    harmonise_and_merge,
    stratified_temporal_partition,
    temporal_ks_validation,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml",
                        help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = resolve_device(cfg.get("device", "auto"))
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "smoke.log")
    logger.info("Device: %s", gpu_info())
    set_seed(cfg["phase_a"]["training_seeds"][0])

    t0 = time.time()

    # 1. Load and merge data
    ulb = Path(cfg["paths"]["datasets"]) / cfg["data"]["ulb_csv"]
    kag = Path(cfg["paths"]["datasets"]) / cfg["data"]["kaggle_csv"]
    df = harmonise_and_merge(ulb, kag)
    logger.info("Merged data: %s, fraud ratio %.4f%%",
                df.shape, 100 * df[TARGET].mean())

    # 2. Partition across clients
    clients = stratified_temporal_partition(
        df, n_clients=cfg["data"]["n_clients"],
        seed=cfg["phase_a"]["partition_seeds"][0],
    )
    for i, c in enumerate(clients):
        logger.info("  Client %d: %d rows, fraud %.4f%%",
                    i, len(c), 100 * c[TARGET].mean())

    ks = temporal_ks_validation(clients)
    logger.info("KS validation (max D, min p): %.4f, %.4f",
                ks["D"].max(), ks["p"].min())

    # 3. Split each client into train / val / test
    splits = [
        client_train_val_test_split(c, seed=cfg["phase_a"]["partition_seeds"][0])
        for c in clients
    ]

    # 4. CTGAN on the first client (subset of what the full campaign does)
    logger.info("Training DP-CTGAN on client 0 ...")
    ctg, eps_c = train_ctgan_dp(
        splits[0][0][FEATURES + [TARGET]],
        discrete_cols=[TARGET],
        epochs=cfg["ctgan"]["epochs"],
        batch_size=cfg["ctgan"]["batch_size"],
        noise_multiplier=cfg["ctgan"]["noise_multiplier"],
        max_grad_norm=cfg["ctgan"]["max_grad_norm"],
        embedding_dim=cfg["ctgan"]["embedding_dim"],
        generator_dim=tuple(cfg["ctgan"]["generator_dim"]),
        discriminator_dim=tuple(cfg["ctgan"]["discriminator_dim"]),
        device=device,
    )
    logger.info("  ε_ctgan ≈ %.3f", eps_c)

    synth0 = sample_balanced_synthetic(
        ctg,
        n_samples=cfg["synthetic"]["per_client"],
        fraud_ratio=cfg["synthetic"]["fraud_ratio"],
    )
    logger.info("  Synthetic sample: %s, fraud %.4f%%",
                synth0.shape, 100 * synth0[TARGET].mean())

    # 5. One federated round end-to-end
    client_ds = []
    for tr, _, _ in splits:
        X, y = to_tensor(tr, FEATURES, TARGET)
        client_ds.append(TensorDataset(X, y))

    global_m = FraudMLP()
    logger.info("Model params: %d", global_m.count_params())
    new_state = fedavg_round(
        global_m.state_dict(),
        client_ds,
        FraudMLP,
        device,
        cfg["federated"]["noise_multiplier"],
        cfg["federated"]["max_grad_norm"],
        cfg["federated"]["local_epochs"],
        cfg["federated"]["batch_size"],
        cfg["federated"]["lr"],
    )
    global_m.load_state_dict(new_state)

    test_all = pd.concat([te for (_, _, te) in splits], ignore_index=True)
    Xt, yt = to_tensor(test_all, FEATURES, TARGET)
    metrics = evaluate(global_m, Xt, yt, device)
    logger.info("Smoke metrics: %s", json.dumps(metrics, indent=2))

    dur = (time.time() - t0) / 60
    logger.info("Smoke test PASSED in %.1f min", dur)


if __name__ == "__main__":
    main()
