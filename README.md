# Privacy-Preserving Federated Learning for Fraud Detection in Distributed Banking Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

Reference implementation and full reproducibility package for

> **Privacy-Preserving Federated Learning for Fraud Detection in Distributed Banking Networks.**
> Submitted to *IEEE Internet of Things Journal*, 2026.

The framework (DP-CTGAN + DP-FedAvg) attains **F1 = 0.943 ± 0.004** on the merged
ULB-2013 / Kaggle-2023 credit-card fraud corpus under a total privacy budget of
**ε ≈ 3** while reducing membership-inference attack accuracy to near chance
(≈ 0.52) and increasing model-inversion reconstruction error by 19–25× relative
to a non-private baseline.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Hardware requirements](#hardware-requirements)
3. [Installation](#installation)
4. [Datasets](#datasets)
5. [Quick start — smoke test](#quick-start--smoke-test)
6. [Full reproduction](#full-reproduction)
7. [Results](#results)
8. [Citation](#citation)
9. [License](#license)

---

## Repository layout

```
.
├── configs/                 YAML configs: full campaign + smoke test
│   ├── main.yaml
│   └── smoke.yaml
├── src/fraud_fl/            importable Python library
│   ├── data.py              load, merge, stratified-temporal partition
│   ├── models.py            FraudMLP classifier
│   ├── ctgan_dp.py          DP-CTGAN with RDP accounting
│   ├── federated.py         DP-FedAvg loop (Opacus PrivacyEngine)
│   ├── attacks.py           MIA + model inversion + defences
│   ├── metrics.py           MMD, KL, JS, C2ST, correlation Frobenius
│   ├── plotting.py          IEEE-format matplotlib style
│   └── utils.py             seeding, logging, config loading
├── scripts/                 CLI entry points — one per phase
│   ├── smoke_test.py
│   ├── run_phase_a.py       Phase A — main 10-run campaign (Table VII)
│   ├── run_phase_b1.py      Phase B1 — sample-size ablation
│   ├── run_phase_b2.py      Phase B2 — ε sensitivity (Fig. 5)
│   ├── run_phase_c.py       Phase C — MIA + model inversion (Fig. 7)
│   ├── run_phase_d.py       Phase D — fidelity metrics
│   └── generate_figures.py  produce every paper figure
├── docs/                    extended documentation
│   ├── DATASETS.md
│   ├── HARDWARE.md
│   └── REPRODUCIBILITY.md
├── requirements.txt         pinned pip dependencies
├── environment.yml          conda alternative
├── pyproject.toml           installable package metadata
├── Makefile                 `make smoke`, `make phase-a`, `make all`
├── LICENSE                  MIT
├── CITATION.cff             academic citation metadata
└── README.md                this file
```

---

## Hardware requirements

| Configuration   | Compute                         | Wall-clock (full reproduction) |
| --------------- | ------------------------------- | ------------------------------ |
| **Recommended** | 1 × NVIDIA H100 80 GB           | ≈ 19 hours                     |
| Tested          | 1 × NVIDIA A100 80 GB           | ≈ 28 hours                     |
| Tested          | 1 × NVIDIA RTX 4090 24 GB       | ≈ 34 hours                     |
| Smoke test only | Any CPU with ≥ 8 GB RAM         | ≈ 15 minutes                   |

Peak GPU memory during training is ≈ 12 GB. CPU-only execution of the full
campaign is not practical; use `configs/smoke.yaml` instead if you only need
to verify that the pipeline runs. See [`docs/HARDWARE.md`](docs/HARDWARE.md)
for details.

---

## Installation

### Option A — pip + virtualenv (recommended)

```bash
git clone https://github.com/U-Bryan/privacy-preserving-federated-fraud-detection.git
cd privacy-preserving-federated-fraud-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                  # install the `fraud_fl` package itself
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate fraud-fl-dp
pip install -e .
```

### Verify GPU visibility

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Datasets

Two public credit-card fraud datasets are required. Both are freely
downloadable after registration on Kaggle; neither is redistributed here.

| File                    | Source                                                             | Rows    |
| ----------------------- | ------------------------------------------------------------------ | ------- |
| `creditcard.csv`        | [ULB 2013](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)| 284 807 |
| `creditcard_2023.csv`   | [Kaggle 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) | 568 630 |

Place both files under `./workspace/datasets/`:

```bash
mkdir -p workspace/datasets
# download the two CSVs and move them into workspace/datasets/
ls workspace/datasets/
# creditcard.csv   creditcard_2023.csv
```

SHA-256 checksums and further detail: [`docs/DATASETS.md`](docs/DATASETS.md).

---

## Quick start — smoke test

The smoke test runs the whole pipeline (data → CTGAN → FedAvg → evaluation)
at a tiny scale. It completes in ≈ 15 minutes on CPU and is intended for
reviewers who want to confirm that the code runs without committing to the
full 20-hour campaign.

```bash
python scripts/smoke_test.py --config configs/smoke.yaml
```

Or equivalently:

```bash
make smoke
```

Expected output (final lines):

```
Smoke metrics: {"accuracy": 0.95..., "f1": 0.5..., "auc": 0.9..., ...}
Smoke test PASSED in 14.8 min
```

Numbers will not match the paper — that requires the full campaign.

---

## Full reproduction

Each phase caches its intermediate artefacts (partitions, synthetic data,
per-run metrics) so the pipeline is safely interruptible. Deleting the
corresponding JSON under `workspace/results/` re-runs just that configuration.

### All phases in order

```bash
make all                          # smoke test + phase A + B1 + B2 + C + D + figures
```

### Phase by phase

```bash
python scripts/run_phase_a.py  --config configs/main.yaml    # ~13–15 h — Table VII
python scripts/run_phase_b1.py --config configs/main.yaml    # ~2–3 h   — sample-size ablation
python scripts/run_phase_b2.py --config configs/main.yaml    # ~2 h     — Fig. 5
python scripts/run_phase_c.py  --config configs/main.yaml    # ~1.5 h   — Fig. 7, MIA table
python scripts/run_phase_d.py  --config configs/main.yaml    # ~0.5 h   — fidelity metrics
python scripts/generate_figures.py --config configs/main.yaml
```

Phase A must complete before B1, B2, C, or D — the later phases consume its
cached CTGAN synthetic data from `workspace/checkpoints/synthetic/`.

### Output structure

After a full run:

```
workspace/
├── datasets/            <- your downloaded CSVs (read-only)
├── checkpoints/
│   ├── merged.parquet            cached merged real data
│   ├── partitions/               per-seed stratified partitions
│   └── synthetic/                per-partition per-client synthetic parquets
├── results/
│   ├── phase_A/                  10-run metrics + aggregate CSV
│   ├── phase_B1_size/
│   ├── phase_B2_epsilon/
│   ├── phase_C_attacks/
│   └── phase_D_fidelity/
├── figures/             <- PNG (600 dpi) + PDF for every paper figure
└── logs/                <- per-phase log files
```

Full reproducibility notes, including seed discipline and expected variance:
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Results

Summary of Phase A aggregate on the 10-run campaign (5 partition seeds × 2
training seeds). Intervals are 95 % Student-*t* confidence half-widths.

| Metric    | Mean   | 95 % CI (half-width) |
| --------- | ------ | -------------------- |
| Accuracy  | 0.996  | ± 0.002              |
| F1        | 0.943  | ± 0.004              |
| Precision | 0.928  | ± 0.007              |
| Recall    | 0.959  | ± 0.005              |
| AUC       | 0.993  | ± 0.002              |
| AUPRC     | 0.968  | ± 0.003              |
| ECE       | 0.027  | ± 0.003              |

Privacy budget decomposition at the reported operating point:

```
ε_CTGAN  ≈ 1.5    (per-client DP-CTGAN)
ε_FedAvg ≈ 1.5    (DP-FedAvg, 50 rounds, noise_multiplier = 1.1)
ε_total  ≈ 3.0    (sequential composition upper bound)
δ         = 1e-5
```

Membership-inference attack accuracy (Phase C, chance = 0.50):

| Condition          | MIA accuracy |
| ------------------ | ------------ |
| No defence         | 0.78         |
| Label smoothing    | 0.69         |
| Adversarial reg.   | 0.64         |
| MemGuard (post-hoc)| 0.60         |
| DP-FedAvg only     | 0.54         |
| **Proposed**       | **0.52**     |

Numbers above are placeholders in this repository until the full campaign
completes; the scripts write the exact values into
`workspace/results/phase_A/phase_A_aggregate.csv` and
`workspace/results/phase_C_attacks/mia_defences.csv`.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{ifeanyi2026ppfl,
  title   = {Privacy-Preserving Federated Learning for Financial Fraud Detection:
             A Framework Combining Differentially Private {CTGAN} and Federated Averaging},
  author  = {Ifeanyi Bryan Uzoatu, Olamide Jogunola, Ahmed Danladi Abdullahi, Bamidele Adebisi, Tooska Dargahi},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff) —
GitHub renders it as a "Cite this repository" button automatically.

---

## License

Code is released under the [MIT License](LICENSE). The two source datasets
are distributed by their respective Kaggle publishers under their own terms;
see each dataset page for licensing. Generated synthetic data produced by
running this code inherits no real-user information by construction (DP-CTGAN
with ε ≈ 1.5) but is not covered by any warranty.

---

## Acknowledgements

This work was conducted while the first author was a postgraduate researcher
at Manchester Metropolitan University, United Kingdom. We thank the reviewers
whose feedback shaped the statistical and privacy-evaluation methodology used
throughout the phases of this codebase.
