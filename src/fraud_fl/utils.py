"""Shared utilities: logging, seeding, config loading, device resolution."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def setup_logging(log_file: str | Path | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with console and optional file output."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to 'cuda' when available, otherwise 'cpu'."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable — falling back to CPU.")
        return "cpu"
    return requested


def ensure_dirs(cfg: Mapping[str, Any]) -> None:
    """Create every path declared under `cfg['paths']` (and some sub-dirs)."""
    paths = cfg["paths"]
    for key, value in paths.items():
        Path(value).mkdir(parents=True, exist_ok=True)
    # Sub-directories for checkpoints and phase results
    ckpt = Path(paths["checkpoints"])
    (ckpt / "partitions").mkdir(parents=True, exist_ok=True)
    (ckpt / "synthetic").mkdir(parents=True, exist_ok=True)
    res = Path(paths["results"])
    for sub in ("phase_A", "phase_B1_size", "phase_B2_epsilon",
                "phase_C_attacks", "phase_D_fidelity"):
        (res / sub).mkdir(parents=True, exist_ok=True)


def gpu_info() -> str:
    """Return a short string describing the active compute device."""
    if not torch.cuda.is_available():
        return "CPU-only"
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"{name} ({mem:.1f} GB, CUDA {torch.version.cuda})"
