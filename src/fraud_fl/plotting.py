"""Publication-quality plotting configuration (IEEE column widths)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401 — side-effect import for backend
import seaborn as sns

# IEEE figure widths (inches)
IEEE_COL = 3.5      # single column
IEEE_2COL = 7.16    # double column

# Colour-blind safe palette (Okabe & Ito)
PALETTE = {
    "orange": "#E69F00",
    "blue":   "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "navy":   "#0072B2",
    "red":    "#D55E00",
    "pink":   "#CC79A7",
    "grey":   "#999999",
}


def set_pub_style() -> None:
    """Apply IEEE-ready matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi":         120,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "font.family":        "DejaVu Sans",
        "font.size":          9,
        "axes.labelsize":     9,
        "axes.titlesize":     10,
        "legend.fontsize":    8,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "axes.linewidth":     0.8,
        "grid.linewidth":     0.4,
        "lines.linewidth":    1.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "legend.frameon":     False,
    })
    sns.set_palette(list(PALETTE.values()))


def save_fig(fig, path_no_ext: str | Path) -> None:
    """Save a figure as both PNG (600 dpi) and PDF (vector)."""
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_no_ext}.png", dpi=600)
    fig.savefig(f"{path_no_ext}.pdf")
