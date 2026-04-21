# Reproducibility notes

This document explains how experiment seeding works, what variance to
expect, and how to diagnose results that diverge from those reported in
the paper.

## Seeding discipline

Two seed families are separated throughout the code:

* **Partition seed** — controls stratified-temporal partitioning across
  clients and the train/validation/test split within each client.
* **Training seed** — controls model initialisation, DP-SGD noise, Opacus
  sample order, CTGAN initialisation, and synthetic sampling.

For Phase A, the paper reports 10 runs from the Cartesian product of
5 partition seeds × 2 training seeds. You can find them in
`configs/main.yaml` under `phase_a`:

```yaml
phase_a:
  partition_seeds: [42, 101, 202, 303, 404]
  training_seeds:  [7, 77]
```

Every script calls [`fraud_fl.utils.set_seed`](../src/fraud_fl/utils.py) at
run start, which seeds Python's `random`, NumPy, and PyTorch (both CPU
and CUDA).

## Non-determinism under CUDA

Three sources of non-determinism remain even with matching seeds:

1. **CUDA kernel non-determinism.** Some PyTorch ops (e.g. atomic adds in
   backward passes) are not deterministic by default. Setting
   `torch.use_deterministic_algorithms(True)` would enforce determinism but
   at a large speed cost, so it is left off. Differences on the order of
   10⁻⁴ between runs with the same seed on the same hardware are expected.
2. **Opacus noise realisation.** DP-SGD adds Gaussian noise calibrated to
   the seed; different PyTorch versions can produce different byte-level
   realisations of that noise. We pin Opacus to 1.5.4 for this reason.
3. **Hardware.** Different GPUs (e.g. H100 vs A100) produce slightly
   different numerical results because of different reduction orderings.
   Reproduction on a different GPU generation will still land within the
   reported 95 % CI but may not bit-exactly match.

## Expected per-metric variance

From the 10-run Phase A campaign (5 × 2 seeds):

| Metric    | Mean      | 95 % CI (half-width) |
| --------- | --------- | -------------------- |
| Accuracy  | 0.9961    | ± 0.0018             |
| F1        | 0.9431    | ± 0.0042             |
| Precision | 0.9283    | ± 0.0067             |
| Recall    | 0.9590    | ± 0.0051             |
| AUC       | 0.9931    | ± 0.0024             |
| AUPRC     | 0.9680    | ± 0.0031             |
| ECE       | 0.0271    | ± 0.0029             |

Any reproduction whose aggregate means fall within 2× these half-widths is
consistent with the paper. Individual runs can differ by up to
± 0.01 on F1.

## Caching and resumability

Every phase writes its intermediate artefacts to `workspace/checkpoints/`
and its per-configuration result JSONs to `workspace/results/phase_X/`.
Scripts check for these files before doing any work:

* Merged real data:        `workspace/checkpoints/merged.parquet`
* Client partitions:       `workspace/checkpoints/partitions/partition_{seed}.pkl`
* Synthetic parquets:      `workspace/checkpoints/synthetic/p{seed}_c{k}.parquet`
* Per-run JSONs:           `workspace/results/phase_A/p{seed}_t{seed}.json`

To force re-computation of a single configuration:

```bash
rm workspace/results/phase_A/p42_t7.json
python scripts/run_phase_a.py --config configs/main.yaml
```

## Troubleshooting

### "CUDA out of memory" during CTGAN or DP-FedAvg

See [HARDWARE.md](HARDWARE.md#memory-discipline) for mitigation. The
`max_physical_batch_size` inside `federated._local_train_dp` is the first
knob to turn.

### Reported F1 differs by more than 0.01 from the paper

Confirm in order:

1. Both CSV datasets have the SHA-256 checksums recorded in
   [DATASETS.md](DATASETS.md#checksums).
2. Python ≥ 3.10 is in use, and `opacus==1.5.4` and `ctgan==0.12.0` exactly.
3. The 10 runs completed — check that
   `workspace/results/phase_A/phase_A_all_runs.csv` has 10 rows.
4. No runs silently failed — search `workspace/logs/phase_a.log` for
   `Traceback`.

### "ε_total" disagrees with the paper

The realised ε depends on the exact step count, which is
`local_epochs × (per_client // batch_size) × rounds` and can differ by
±1 % depending on how Opacus counts partial batches. The paper reports
a ceiling ε (sequential composition upper bound), not a tight value.

### Figures look different

Matplotlib's default fonts have changed across versions. All figure code
pins `font.family = "DejaVu Sans"` (included with Matplotlib), so results
should be stable; if you see missing glyphs, confirm DejaVu Sans is
installed in your environment.
