#!/usr/bin/env python
"""Phase B1 — synthetic-sample-size ablation.

Justifies the 150 000 sample choice used in the main campaign by showing
how convergence stability (CoV of round-to-round ΔF1) and final F1 vary
with per-client synthetic size. Outputs summary CSV plus paired Wilcoxon
and one-way ANOVA results.

    python scripts/run_phase_b1.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, wilcoxon
from torch.utils.data import TensorDataset

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = Path(cfg["paths"]["checkpoints"])
    result = Path(cfg["paths"]["results"]) / "phase_B1_size"
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "phase_b1.log")
    logger.info("Device: %s", gpu_info())

    p_seed = cfg["phase_b1"]["partition_seed"]
    clients = pickle.load(
        open(ckpt / "partitions" / f"partition_{p_seed}.pkl", "rb")
    )
    splits = [client_train_val_test_split(c, seed=p_seed) for c in clients]
    synth_full = [
        pd.read_parquet(ckpt / "synthetic" / f"p{p_seed}_c{k}.parquet")
        for k in range(cfg["data"]["n_clients"])
    ]

    test_all = pd.concat([te for (_, _, te) in splits], ignore_index=True)
    Xtest, ytest = to_tensor(test_all, FEATURES, TARGET)

    rows: list[dict] = []
    for S in cfg["phase_b1"]["sizes"]:
        for seed in cfg["phase_b1"]["seeds"]:
            run_id = f"s{S}_seed{seed}"
            out = result / f"{run_id}.json"
            if out.exists():
                rows.append(json.load(open(out)))
                logger.info("[%s] cached", run_id)
                continue

            set_seed(seed)
            rng = np.random.default_rng(seed)

            client_ds = []
            for synth in synth_full:
                idx = rng.choice(len(synth), size=S, replace=(S > len(synth)))
                sub = synth.iloc[idx].reset_index(drop=True)
                X, y = to_tensor(sub, FEATURES, TARGET)
                client_ds.append(TensorDataset(X, y))

            global_m = FraudMLP().to(device)
            state = global_m.state_dict()
            round_f1: list[float] = []
            t0 = time.time()

            for r in range(cfg["federated"]["rounds"]):
                state = fedavg_round(
                    state, client_ds, FraudMLP, device,
                    cfg["federated"]["noise_multiplier"],
                    cfg["federated"]["max_grad_norm"],
                    cfg["federated"]["local_epochs"],
                    cfg["federated"]["batch_size"],
                    cfg["federated"]["lr"],
                )
                if r >= 1:
                    global_m.load_state_dict(state)
                    m = evaluate(global_m, Xtest, ytest, device)
                    round_f1.append(m["f1"])

            deltas = np.diff(round_f1)
            cov = float(np.std(deltas) / (np.abs(np.mean(deltas)) + 1e-8))
            final_metrics = evaluate(global_m, Xtest, ytest, device)
            rec = dict(
                size=S, seed=seed,
                final_f1=round_f1[-1],
                cov_delta=cov,
                wall_clock_min=(time.time() - t0) / 60,
                **final_metrics,
            )
            json.dump(rec, open(out, "w"), indent=2)
            rows.append(rec)
            logger.info("[%s] F1=%.4f CoV(ΔF1)=%.3f  (%.1f min)",
                        run_id, rec["final_f1"], cov, rec["wall_clock_min"])

    df_b1 = pd.DataFrame(rows)
    df_b1.to_csv(result / "phase_B1_all.csv", index=False)

    # Paired Wilcoxon across sizes (50k vs 150k; 150k vs 250k)
    def _wilcoxon(a, b):
        try:
            w, p = wilcoxon(a, b)
            return {"W": float(w), "p": float(p)}
        except ValueError:
            return {"W": None, "p": None}

    sizes = cfg["phase_b1"]["sizes"]
    stats = {
        "wilcoxon_small_vs_mid":
            _wilcoxon(df_b1[df_b1["size"] == sizes[0]]["cov_delta"].values,
                      df_b1[df_b1["size"] == 150000]["cov_delta"].values)
            if 150000 in sizes else None,
        "wilcoxon_mid_vs_large":
            _wilcoxon(df_b1[df_b1["size"] == 150000]["cov_delta"].values,
                      df_b1[df_b1["size"] == sizes[-1]]["cov_delta"].values)
            if 150000 in sizes else None,
    }
    anova_groups = [df_b1[df_b1["size"] == s]["final_f1"].values for s in sizes]
    if all(len(g) > 1 for g in anova_groups):
        F_anova, p_anova = f_oneway(*anova_groups)
        stats["anova_final_f1"] = {"F": float(F_anova), "p": float(p_anova)}

    json.dump(stats, open(result / "phase_B1_stats.json", "w"), indent=2)

    summary = df_b1.groupby("size").agg(
        mean_cov=("cov_delta", "mean"), std_cov=("cov_delta", "std"),
        mean_f1=("final_f1", "mean"), std_f1=("final_f1", "std"),
    ).reset_index()
    summary.to_csv(result / "phase_B1_summary.csv", index=False)
    logger.info("Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
