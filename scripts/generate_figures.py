#!/usr/bin/env python
"""Generate every figure in the paper from the per-phase result artefacts.

Regenerates Figures 5, 7a, 7b, 9, and 10. Safe to run after the
corresponding phases have completed. Requires only CPU.

    python scripts/generate_figures.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fraud_fl.plotting import IEEE_2COL, IEEE_COL, PALETTE, save_fig, set_pub_style
from fraud_fl.utils import ensure_dirs, load_config, setup_logging


# --------------------------------------------------------------------------
# Figure 5 — privacy–utility trade-off
# --------------------------------------------------------------------------
def fig5_privacy_utility(cfg, logger) -> None:
    src = Path(cfg["paths"]["results"]) / "phase_B2_epsilon" / "phase_B2_all.csv"
    dst = Path(cfg["paths"]["figures"]) / "fig5_privacy_utility"
    df_b2 = pd.read_csv(src)

    summary = df_b2.groupby("eps_target").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        rec_mean=("recall", "mean"), rec_std=("recall", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(IEEE_COL, IEEE_COL * 0.72))
    metric_map = [
        ("acc", PALETTE["navy"],   "Accuracy"),
        ("f1",  PALETTE["orange"], "F1"),
        ("auc", PALETTE["green"],  "AUC"),
        ("rec", PALETTE["red"],    "Recall"),
    ]
    for key, colour, label in metric_map:
        m = summary[f"{key}_mean"]
        s = summary[f"{key}_std"]
        ax.plot(summary["eps_target"], m, marker="o", color=colour, label=label)
        ax.fill_between(summary["eps_target"], m - s, m + s,
                        color=colour, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Metric value")
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("Privacy–utility trade-off (mean ± 1 SD)")
    save_fig(fig, dst)
    plt.close(fig)
    logger.info("  wrote %s{.png,.pdf}", dst)


# --------------------------------------------------------------------------
# Figure 7a/b — model-inversion resistance
# --------------------------------------------------------------------------
def fig7_inversion(cfg, logger) -> None:
    src = Path(cfg["paths"]["results"]) / "phase_C_attacks" / "inversion_results.json"
    inv = json.load(open(src))

    eps_order = ["1.0", "3.0", "10.0", "inf"]
    xs = [1.0, 3.0, 10.0, 30.0]     # "inf" plotted as x=30 for the baseline
    mus = [inv[k]["mean_recon_error"] for k in eps_order]
    sds = [inv[k]["std_recon_error"] for k in eps_order]

    # 7a
    fig, ax = plt.subplots(figsize=(IEEE_COL, IEEE_COL * 0.72))
    ax.errorbar(xs[:-1], mus[:-1], yerr=sds[:-1], fmt="o-", capsize=3,
                color=PALETTE["navy"], label="With DP")
    ax.axhline(mus[-1], color=PALETTE["red"], linestyle="--",
               label="No DP (baseline)")
    ax.fill_between(
        xs[:-1],
        [mu - sd for mu, sd in zip(mus[:-1], sds[:-1])],
        [mu + sd for mu, sd in zip(mus[:-1], sds[:-1])],
        color=PALETTE["navy"], alpha=0.18,
    )
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Reconstruction L2 error")
    ax.set_title("Model-inversion resistance")
    ax.legend()
    save_fig(fig, Path(cfg["paths"]["figures"]) / "fig7a_inversion_error")
    plt.close(fig)

    # 7b — parallel coordinates
    fig, ax = plt.subplots(figsize=(IEEE_2COL, IEEE_COL * 0.7))
    n_feat = len(inv[eps_order[0]]["reconstructions"][0])
    features_idx = np.arange(n_feat)
    colours = {"inf": PALETTE["red"], "10.0": PALETTE["orange"],
               "3.0": PALETTE["navy"], "1.0": PALETTE["green"]}
    labels = {"inf": "No DP", "10.0": "ε=10",
              "3.0": "ε=3 (proposed)", "1.0": "ε=1"}
    for eps_name in eps_order:
        rec = np.array(inv[eps_name]["reconstructions"])
        for r in rec:
            ax.plot(features_idx, r, color=colours[eps_name],
                    alpha=0.25, linewidth=0.6)
        ax.plot(features_idx, rec.mean(axis=0),
                color=colours[eps_name], linewidth=1.8,
                label=labels[eps_name])
    ax.set_xticks([0, 7, 14, 21, 28])
    ax.set_xticklabels(["V1", "V8", "V15", "V22", "V29(Amt)"])
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Reconstructed value (standardised)")
    ax.set_title("Model-inversion reconstructions across privacy budgets")
    ax.legend(loc="upper right", ncol=4)
    save_fig(fig, Path(cfg["paths"]["figures"]) / "fig7b_inversion_parallel")
    plt.close(fig)
    logger.info("  wrote fig7a / fig7b")


# --------------------------------------------------------------------------
# Figure 9 — federated convergence
# --------------------------------------------------------------------------
def fig9_convergence(cfg, logger) -> None:
    phase_a_dir = Path(cfg["paths"]["results"]) / "phase_A"
    all_rounds = []
    for f in sorted(phase_a_dir.glob("p*_t*.json")):
        r = json.load(open(f))
        for rec in r["round_log"]:
            all_rounds.append({
                "round": rec["round"], "f1": rec["f1"],
                "auc": rec["auc"], "accuracy": rec["accuracy"],
                "run": r["run_id"],
            })
    if not all_rounds:
        logger.warning("  Phase A per-run JSON not found — skipping fig9")
        return

    dfr = pd.DataFrame(all_rounds)
    agg = dfr.groupby("round").agg(
        f1_mean=("f1", "mean"),   f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(IEEE_COL * 1.2, IEEE_COL * 0.75))
    for key, colour, label in [
        ("f1",  PALETTE["orange"], "F1"),
        ("auc", PALETTE["green"],  "AUC"),
        ("acc", PALETTE["navy"],   "Accuracy"),
    ]:
        m = agg[f"{key}_mean"]
        s = agg[f"{key}_std"]
        ax.plot(agg["round"], m, marker="o", markersize=3,
                color=colour, label=label)
        ax.fill_between(agg["round"], m - s, m + s, color=colour, alpha=0.18)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Metric")
    ax.legend(loc="lower right")
    ax.set_title("Federated convergence (mean ± 1 SD across 10 runs)")
    save_fig(fig, Path(cfg["paths"]["figures"]) / "fig9_convergence")
    plt.close(fig)
    logger.info("  wrote fig9_convergence")


# --------------------------------------------------------------------------
# Figure 10 — synthetic data quality (PCA + correlation diff)
# --------------------------------------------------------------------------
def fig10_synthetic_quality(cfg, logger) -> None:
    ckpt = Path(cfg["paths"]["checkpoints"])
    merged_path = ckpt / "merged.parquet"
    synth_path = ckpt / "synthetic" / f"p{cfg['phase_d']['partition_seed']}_c0.parquet"
    if not (merged_path.exists() and synth_path.exists()):
        logger.warning("  merged or synthetic parquet missing — skipping fig10")
        return

    cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    real = pd.read_parquet(merged_path).sample(5000, random_state=0)[cols].values
    synth = pd.read_parquet(synth_path).sample(5000, random_state=0)[cols].values

    pca = PCA(n_components=2).fit(real)
    pr = pca.transform(real)
    ps = pca.transform(synth)

    fig, axs = plt.subplots(1, 2, figsize=(IEEE_2COL, IEEE_COL * 0.8))
    axs[0].scatter(pr[:, 0], pr[:, 1], s=3, alpha=0.4,
                   color=PALETTE["navy"], label="Real")
    axs[0].scatter(ps[:, 0], ps[:, 1], s=3, alpha=0.4,
                   color=PALETTE["orange"], label="Synthetic")
    axs[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axs[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axs[0].set_title("(a) PCA projection")
    axs[0].legend()

    Cr = np.corrcoef(real.T)
    Cs = np.corrcoef(synth.T)
    im = axs[1].imshow(np.abs(Cr - Cs), cmap="viridis", vmin=0, vmax=0.3)
    axs[1].set_title("(b) |Corr diff|")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    save_fig(fig, Path(cfg["paths"]["figures"]) / "fig10_synthetic_quality")
    plt.close(fig)
    logger.info("  wrote fig10_synthetic_quality")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_pub_style()
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "figures.log")
    logger.info("Generating figures...")

    fig5_privacy_utility(cfg, logger)
    fig7_inversion(cfg, logger)
    fig9_convergence(cfg, logger)
    fig10_synthetic_quality(cfg, logger)

    logger.info("Figures written to %s", cfg["paths"]["figures"])


if __name__ == "__main__":
    main()
