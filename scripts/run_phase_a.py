#!/usr/bin/env python
"""Phase A — main 10-run campaign.

Produces Table VII (Accuracy, F1, Precision, Recall, AUC, AUPRC, ECE with
95% Student-t confidence intervals across 5 partition seeds x 2 training
seeds = 10 independent runs).

Results cache per-run to disk so the script is safely interruptible and
resumable: delete individual JSON files under ``results/phase_A/`` to rerun.

    python scripts/run_phase_a.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scst
from torch.utils.data import TensorDataset

from fraud_fl.ctgan_dp import sample_balanced_synthetic, train_ctgan_dp
from fraud_fl.data import (
    FEATURES,
    TARGET,
    client_train_val_test_split,
    harmonise_and_merge,
    stratified_temporal_partition,
)
from fraud_fl.federated import evaluate, fedavg_round, rdp_epsilon
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
    result = Path(cfg["paths"]["results"]) / "phase_A"
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "phase_a.log")
    logger.info("Device: %s", gpu_info())

    # 1. Load (or cache) the merged real dataset
    merged_cache = ckpt / "merged.parquet"
    if merged_cache.exists():
        df = pd.read_parquet(merged_cache)
        logger.info("Loaded cached merged data: %s", df.shape)
    else:
        df = harmonise_and_merge(
            Path(cfg["paths"]["datasets"]) / cfg["data"]["ulb_csv"],
            Path(cfg["paths"]["datasets"]) / cfg["data"]["kaggle_csv"],
        )
        df.to_parquet(merged_cache)
        logger.info("Merged and cached: %s", df.shape)

    all_results: list[dict] = []

    for p_seed in cfg["phase_a"]["partition_seeds"]:
        # Partition (cached)
        partition_file = ckpt / "partitions" / f"partition_{p_seed}.pkl"
        if partition_file.exists():
            clients = pickle.load(open(partition_file, "rb"))
        else:
            clients = stratified_temporal_partition(
                df, n_clients=cfg["data"]["n_clients"], seed=p_seed,
            )
            pickle.dump(clients, open(partition_file, "wb"))
        splits = [client_train_val_test_split(c, seed=p_seed) for c in clients]

        # Synthetic data per client (cached across training seeds)
        synth_datasets: list[pd.DataFrame] = []
        for k, (tr, _, _) in enumerate(splits):
            synth_path = ckpt / "synthetic" / f"p{p_seed}_c{k}.parquet"
            if synth_path.exists():
                synth_datasets.append(pd.read_parquet(synth_path))
                logger.info("[P%d/C%d] synthetic cached (%d rows)",
                            p_seed, k, len(synth_datasets[-1]))
                continue

            t0 = time.time()
            ctg, eps_c = train_ctgan_dp(
                tr[FEATURES + [TARGET]], discrete_cols=[TARGET],
                epochs=cfg["ctgan"]["epochs"],
                batch_size=cfg["ctgan"]["batch_size"],
                noise_multiplier=cfg["ctgan"]["noise_multiplier"],
                max_grad_norm=cfg["ctgan"]["max_grad_norm"],
                embedding_dim=cfg["ctgan"]["embedding_dim"],
                generator_dim=tuple(cfg["ctgan"]["generator_dim"]),
                discriminator_dim=tuple(cfg["ctgan"]["discriminator_dim"]),
                device=device,
            )
            synth = sample_balanced_synthetic(
                ctg,
                n_samples=cfg["synthetic"]["per_client"],
                fraud_ratio=cfg["synthetic"]["fraud_ratio"],
            )
            synth.to_parquet(synth_path)
            synth_datasets.append(synth)
            logger.info("[P%d/C%d] CTGAN %.1f min, ε_ctgan≈%.3f",
                        p_seed, k, (time.time() - t0) / 60, eps_c)

        # Held-out test set assembled across clients
        test_all = pd.concat([te for (_, _, te) in splits], ignore_index=True)
        Xtest, ytest = to_tensor(test_all, FEATURES, TARGET)

        # Federated training for each training seed
        for t_seed in cfg["phase_a"]["training_seeds"]:
            run_id = f"p{p_seed}_t{t_seed}"
            out_file = result / f"{run_id}.json"
            if out_file.exists():
                logger.info("[%s] already done — loading", run_id)
                all_results.append(json.load(open(out_file)))
                continue

            set_seed(t_seed)

            client_ds = []
            for synth in synth_datasets:
                X, y = to_tensor(synth, FEATURES, TARGET)
                client_ds.append(TensorDataset(X, y))

            global_m = FraudMLP().to(device)
            state = global_m.state_dict()
            round_log: list[dict] = []
            t_start = time.time()

            for r in range(cfg["federated"]["rounds"]):
                state = fedavg_round(
                    state, client_ds, FraudMLP, device,
                    cfg["federated"]["noise_multiplier"],
                    cfg["federated"]["max_grad_norm"],
                    cfg["federated"]["local_epochs"],
                    cfg["federated"]["batch_size"],
                    cfg["federated"]["lr"],
                )
                if (r + 1) % 5 == 0 or r == cfg["federated"]["rounds"] - 1:
                    global_m.load_state_dict(state)
                    m = evaluate(global_m, Xtest, ytest, device)
                    m["round"] = r + 1
                    round_log.append(m)

            final = round_log[-1].copy()
            steps_per_round = (
                cfg["synthetic"]["per_client"] // cfg["federated"]["batch_size"]
            )
            eps_fed = rdp_epsilon(
                cfg["federated"]["noise_multiplier"],
                cfg["federated"]["batch_size"] / cfg["synthetic"]["per_client"],
                cfg["federated"]["local_epochs"] * steps_per_round
                * cfg["federated"]["rounds"],
            )
            final.update({
                "partition_seed":     p_seed,
                "training_seed":      t_seed,
                "run_id":             run_id,
                "eps_fed":            eps_fed,
                "wall_clock_sec":     time.time() - t_start,
                "round_log":          round_log,
            })
            json.dump(final, open(out_file, "w"), indent=2)
            all_results.append(final)
            logger.info("[%s] F1=%.4f AUC=%.4f ECE=%.4f  (%.1f min)",
                        run_id, final["f1"], final["auc"], final["ece"],
                        final["wall_clock_sec"] / 60)

    # -------- Aggregate across all runs --------
    df_res = pd.DataFrame([
        {k: v for k, v in r.items() if k != "round_log"} for r in all_results
    ])
    df_res.to_csv(result / "phase_A_all_runs.csv", index=False)

    logger.info("=== PHASE A SUMMARY (%d runs) ===", len(df_res))
    agg_rows = []
    for metric in ["accuracy", "f1", "precision", "recall", "auc", "auprc", "ece"]:
        vals = df_res[metric].values
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals))
        tcrit = scst.t.ppf(0.975, df=len(vals) - 1)
        ci = tcrit * sem
        agg_rows.append({
            "metric": metric, "mean": mean, "std": vals.std(ddof=1),
            "ci95_halfwidth": ci,
            "lo": mean - ci, "hi": mean + ci, "n": len(vals),
        })
        logger.info("  %-10s %.4f ± %.4f  (std=%.4f, n=%d)",
                    metric, mean, ci, vals.std(ddof=1), len(vals))

    pd.DataFrame(agg_rows).to_csv(result / "phase_A_aggregate.csv", index=False)
    logger.info("Phase A complete — results in %s", result)


if __name__ == "__main__":
    main()
