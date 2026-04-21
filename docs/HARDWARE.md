# Hardware requirements

## Summary

| Target                         | Minimum | Recommended     |
| ------------------------------ | ------- | --------------- |
| GPU memory                     | 16 GB   | 40 GB or more   |
| System RAM                     | 16 GB   | 32 GB           |
| Disk space                     | 20 GB   | 40 GB           |
| Python                         | 3.10    | 3.11            |
| CUDA (for GPU)                 | 11.8    | 12.1            |

## GPU benchmarks

Full reproduction wall-clock on a single GPU:

| GPU                          | Memory used | Wall-clock | Notes                                      |
| ---------------------------- | ----------- | ---------- | ------------------------------------------ |
| NVIDIA H100 80 GB (SXM5)     | ≈ 12 GB     | 19 h       | Recommended target for reproduction.       |
| NVIDIA A100 80 GB (SXM4)     | ≈ 12 GB     | 28 h       |                                            |
| NVIDIA A100 40 GB (PCIe)     | ≈ 12 GB     | 30 h       |                                            |
| NVIDIA RTX 4090 24 GB        | ≈ 14 GB     | 34 h       | Consumer card, works fine.                 |
| NVIDIA RTX 3090 24 GB        | ≈ 14 GB     | 48 h       |                                            |
| NVIDIA V100 32 GB            | ≈ 12 GB     | 55 h       |                                            |
| NVIDIA T4 16 GB              | ≈ 14 GB     | 80 h +     | Works but close to the memory ceiling.     |

## Phase-level wall-clock (H100)

| Phase                                          | Wall-clock  |
| ---------------------------------------------- | ----------- |
| A — main 10-run campaign (CTGAN × 15 + FedAvg × 10) | 13 – 15 h   |
| B1 — synthetic sample-size ablation            |  2 – 3 h    |
| B2 — ε sensitivity sweep                       |  ≈ 2 h      |
| C — MIA defences + model inversion             |  ≈ 1.5 h    |
| D — fidelity metrics (CPU-bound)               |  ≈ 0.5 h    |
| Figure regeneration                            |  ≈ 5 min    |
| **Total**                                      | **≈ 19 h**  |

CTGAN training accounts for roughly 60 % of Phase A's wall-clock. If you
already have cached synthetic parquets (from a prior run), each training
seed costs only ≈ 45 minutes.

## Memory discipline

DP-SGD with Opacus is memory-hungry because per-sample gradients are
materialised before clipping and noising. The scripts use Opacus's
`BatchMemoryManager` with `max_physical_batch_size = 2048` to work around
this on 12–16 GB cards. If you hit OOM:

1. Lower `max_physical_batch_size` inside
   [`src/fraud_fl/federated.py`](../src/fraud_fl/federated.py) — halving it
   halves peak memory at a slight throughput cost.
2. Lower `federated.batch_size` in your config (keep it a power of two).
3. Lower `ctgan.embedding_dim` and the generator/discriminator widths in
   the `ctgan:` block — these are the biggest memory consumers during
   CTGAN training.

## CPU-only execution

Running the full campaign on CPU is **not** practical — expect ≥ 30× slower
than H100 for CTGAN and ≥ 100× slower for DP-SGD. The repository ships a
`configs/smoke.yaml` that reduces dataset sizes, epochs, and round counts so
the entire pipeline completes in ≈ 15 min on a modern laptop CPU. Use the
smoke config for correctness validation only; its numerical results are
not comparable to the paper.

```bash
python scripts/smoke_test.py --config configs/smoke.yaml
```

## Cloud reproduction

Rough estimates at on-demand spot-market prices (2026):

| Provider             | Instance                | Approx. cost for full run |
| -------------------- | ----------------------- | ------------------------- |
| Google Colab Pro+    | A100 40 GB              | ≈ $15 (compute units)     |
| Lambda Labs          | 1×H100 PCIe             | ≈ $55                     |
| RunPod               | 1×A100 SXM              | ≈ $35                     |
| AWS                  | p5.48xlarge (1×H100)    | ≈ $100                    |

Attach a persistent disk for `workspace/` so that interrupted runs can
resume from the parquet checkpoints.
