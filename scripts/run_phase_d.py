#!/usr/bin/env python
"""Phase D — synthetic-data fidelity metrics.

Computes MMD, KL, JS, C2ST, and correlation-Frobenius ratio between real
and synthetic per-client data, plus the TSTR/TRTR utility ratio.

    python scripts/run_phase_d.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_fl.data import FEATURES, client_train_val_test_split
from fraud_fl.metrics import (
    c2st_accuracy,
    correlation_frobenius,
    js_divergence,
    mmd_rbf,
    per_feature_kl,
)
from fraud_fl.utils import ensure_dirs, load_config, setup_logging

# TSTR = Train-on-Synthetic, Test-on-Real; TRTR is the fully-real upper
# bound reported in the paper. Hard-coded because re-deriving TRTR requires
# a full Phase-A-equivalent run on real data.
TRTR_F1_REPORTED = 0.998


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    ckpt = Path(cfg["paths"]["checkpoints"])
    result = Path(cfg["paths"]["results"]) / "phase_D_fidelity"
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "phase_d.log")

    p_seed = cfg["phase_d"]["partition_seed"]
    clients = pickle.load(
        open(ckpt / "partitions" / f"partition_{p_seed}.pkl", "rb")
    )
    splits = [client_train_val_test_split(c, seed=p_seed) for c in clients]

    all_metrics: list[dict] = []
    sub_n = cfg["phase_d"]["subsample_n"]
    for k in range(cfg["data"]["n_clients"]):
        real = splits[k][0][FEATURES].values
        synth = pd.read_parquet(
            ckpt / "synthetic" / f"p{p_seed}_c{k}.parquet"
        )[FEATURES].values

        n = min(len(real), len(synth), sub_n)
        rng = np.random.default_rng(0)
        r = real[rng.choice(len(real), n, replace=False)]
        s = synth[rng.choice(len(synth), n, replace=False)]

        mean_kl, max_kl = per_feature_kl(r, s)
        m = dict(
            client=k,
            mmd2=mmd_rbf(r, s, gamma=1.0),
            kl_mean=mean_kl,
            kl_max=max_kl,
            js_mean=js_divergence(r, s),
            c2st_acc=c2st_accuracy(r, s),
            corr_frob_rel=correlation_frobenius(r, s),
        )
        all_metrics.append(m)
        logger.info("Client %d: MMD²=%.4f KL_mean=%.4f C2ST=%.3f CorrFrob=%.3f",
                    k, m["mmd2"], m["kl_mean"], m["c2st_acc"], m["corr_frob_rel"])

    df = pd.DataFrame(all_metrics)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["mean", "client"] = "mean"
    df.to_csv(result / "fidelity_metrics.csv", index=False)

    # TSTR/TRTR utility ratio
    phase_a_agg_path = (Path(cfg["paths"]["results"])
                        / "phase_A" / "phase_A_aggregate.csv")
    if phase_a_agg_path.exists():
        phase_a = pd.read_csv(phase_a_agg_path)
        tstr_f1 = float(phase_a[phase_a["metric"] == "f1"]["mean"].values[0])
        ratio = {
            "tstr_f1": tstr_f1,
            "trtr_f1": TRTR_F1_REPORTED,
            "tstr_trtr_ratio": tstr_f1 / TRTR_F1_REPORTED,
        }
        json.dump(ratio, open(result / "tstr_trtr.json", "w"), indent=2)
        logger.info("TSTR/TRTR ratio: %s", ratio)
    else:
        logger.warning("Phase A aggregate not found — skipping TSTR/TRTR.")

    logger.info("Phase D complete — results in %s", result)


if __name__ == "__main__":
    main()
