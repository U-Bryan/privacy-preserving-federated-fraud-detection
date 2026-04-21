#!/usr/bin/env python
"""Phase B2 — privacy-budget (ε) sensitivity with error bands.

Sweeps target ε across a logarithmic range, repeats each setting across
``phase_b2.seeds``, and records utility metrics plus realised ε from RDP
accounting. Produces the data behind Figure 5 in the paper.

    python scripts/run_phase_b2.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
from torch.utils.data import TensorDataset

from fraud_fl.data import FEATURES, TARGET, client_train_val_test_split
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
    result = Path(cfg["paths"]["results"]) / "phase_B2_epsilon"
    logger = setup_logging(Path(cfg["paths"]["logs"]) / "phase_b2.log")
    logger.info("Device: %s", gpu_info())

    p_seed = cfg["phase_b2"]["partition_seed"]
    clients = pickle.load(
        open(ckpt / "partitions" / f"partition_{p_seed}.pkl", "rb")
    )
    splits = [client_train_val_test_split(c, seed=p_seed) for c in clients]
    synth = [
        pd.read_parquet(ckpt / "synthetic" / f"p{p_seed}_c{k}.parquet")
        for k in range(cfg["data"]["n_clients"])
    ]
    test_all = pd.concat([te for (_, _, te) in splits], ignore_index=True)
    Xtest, ytest = to_tensor(test_all, FEATURES, TARGET)

    rows: list[dict] = []
    for eps_t in cfg["phase_b2"]["eps_targets"]:
        nm = cfg["phase_b2"]["noise_by_eps"][str(eps_t)]
        for seed in cfg["phase_b2"]["seeds"]:
            run_id = f"eps{eps_t}_seed{seed}"
            out = result / f"{run_id}.json"
            if out.exists():
                rows.append(json.load(open(out)))
                continue

            set_seed(seed)
            client_ds = [TensorDataset(*to_tensor(s, FEATURES, TARGET))
                         for s in synth]

            global_m = FraudMLP().to(device)
            state = global_m.state_dict()
            t0 = time.time()

            for _ in range(cfg["federated"]["rounds"]):
                state = fedavg_round(
                    state, client_ds, FraudMLP, device,
                    nm, cfg["federated"]["max_grad_norm"],
                    cfg["federated"]["local_epochs"],
                    cfg["federated"]["batch_size"],
                    cfg["federated"]["lr"],
                )

            global_m.load_state_dict(state)
            m = evaluate(global_m, Xtest, ytest, device)
            steps = (cfg["synthetic"]["per_client"]
                     // cfg["federated"]["batch_size"]) \
                * cfg["federated"]["local_epochs"] \
                * cfg["federated"]["rounds"]
            eps_realised = rdp_epsilon(
                nm,
                cfg["federated"]["batch_size"] / cfg["synthetic"]["per_client"],
                steps,
            )
            rec = dict(
                eps_target=eps_t, noise_multiplier=nm,
                eps_realised=eps_realised, seed=seed,
                wall_clock_min=(time.time() - t0) / 60,
                **m,
            )
            json.dump(rec, open(out, "w"), indent=2)
            rows.append(rec)
            logger.info("[%s] F1=%.4f AUC=%.4f ε_real=%.2f",
                        run_id, m["f1"], m["auc"], eps_realised)

    pd.DataFrame(rows).to_csv(result / "phase_B2_all.csv", index=False)
    logger.info("Phase B2 complete — results in %s", result)


if __name__ == "__main__":
    main()
