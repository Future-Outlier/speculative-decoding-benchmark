"""Two explicitly hypothetical renewal models for sensitivity analysis.

Model (lockstep batch b, chain drafting):
  q = p_hit^b (batch skips JIT drafting only when every sequence hits)
  hit round:  commit b*tau_s in t_v      miss round: commit b*tau_f in t_v+t_d
  R_ssd/R_sync = [q*r + (1-q)] * (1+x) / (1 + (1-q)*x),  r=tau_s/tau_f, x=t_d/t_v

Break-even r* = 1/(1+x): independent of p_hit and b (miss rounds cancel).
Batch size only attenuates the magnitude via q = p^b.

The mixed-miss variant assumes hitting sequences keep stale proposals on miss
rounds and missing sequences run a fresh neural backup.  Neither model is a
general SSD theorem or a measurement of the official implementation.

Usage: python breakeven_plot.py --out figs/breakeven.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

# Legacy A10 component timings and unclipped accepted-length proxy ratios.  The
# renewal model is steady-state, so finite-request boundary clipping is excluded.
MEASURED = [
    ("DFlash t=1", 7.2 / 36.3, 0.595, ORANGE, "o"),
    ("DFlash t=0", 7.0 / 35.7, 0.600, ORANGE, "^"),
    ("DSpark t=1", 9.1 / 36.6, 0.618, AQUA, "o"),
    ("DSpark t=0", 8.2 / 35.8, 0.620, AQUA, "^"),
]


def r_all_or_nothing(r: float, x: float, p: float, b: np.ndarray) -> np.ndarray:
    # miss round refreshes everyone: commit uses q = p^b on both sides.
    q = p**b
    return (q * r + (1 - q)) * (1 + x) / (1 + (1 - q) * x)


def r_mixed(r: float, x: float, p: float, b: np.ndarray) -> np.ndarray:
    # Renewal-reward: every hitting sequence commits tau_s regardless of the
    # round outcome (P(seq hits)=p), but drafting latency is only skipped
    # when the whole batch hits (P=p^b).
    q = p**b
    return (p * r + (1 - p)) * (1 + x) / (1 + (1 - q) * x)


def mixed_break_even(x: float, p: float, b: float) -> float:
    # Solve r_mixed == 1 analytically.
    q = p**b
    return ((1 + (1 - q) * x) / (1 + x) - (1 - p)) / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/breakeven.png")
    args = ap.parse_args()

    x_ds = 0.25  # DSpark t_d/t_v on A10
    print("all-or-nothing break-even r* (=1/(1+x)):", round(1 / (1 + x_ds), 3))
    for b in (1, 2, 4, 8, 16, 64):
        print(f"mixed break-even at p=0.9, b={b}: {mixed_break_even(x_ds, 0.9, b):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Panel A: required ratio vs time ratio ------------------------------
    xs = np.linspace(0.0, 1.2, 200)
    thr = 1 / (1 + xs)
    ax1.plot(xs, thr, color=BLUE, linewidth=2, label="break-even  r* = 1/(1+x)")
    ax1.fill_between(xs, thr, 1.35, color=BLUE, alpha=0.08)
    ax1.text(0.58, 1.02, "toy model above parity", color=BLUE, fontsize=9)
    ax1.text(0.60, 0.47, "toy model below parity", color=MUTED, fontsize=9)

    for name, x, r, c, m in MEASURED:
        ax1.scatter([x], [r], color=c, marker=m, s=55, zorder=3)
    ax1.annotate(
        "retained-mask-KV surrogate\n(two tested temperatures)",
        xy=(0.225, 0.605), xytext=(0.33, 0.66), fontsize=9, color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    ax1.set_xlabel("x = t_draft / t_verify")
    ax1.set_ylabel("required  τ_stale / τ_fresh")
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0.4, 1.35)
    ax1.set_title("p_hit=1 toy-model threshold", fontsize=11)
    ax1.legend(loc="lower left", fontsize=8.5, frameon=False)

    # Panel B: attenuation with batch size --------------------------------
    bs = np.arange(1, 65)
    p = 0.9
    for r, c, lbl in ((1.00, BLUE, "r = 1.00 (no degradation)"),
                      (0.95, AQUA, "r = 0.95"),
                      (0.62, ORANGE, "r = 0.62 (surrogate)")):
        ys = r_mixed(r, x_ds, p, bs)
        ax2.plot(bs, ys, color=c, linewidth=2)
        ax2.text(bs[-1] * 1.03, ys[-1], lbl, color=c, fontsize=8.5, va="center")
    ax2.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax2.text(1.05, 1.007, "parity vs sync SD", color=MUTED, fontsize=8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax2.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax2.set_xlabel(f"batch size b   (per-seq p_hit = {p}, x = {x_ds})")
    ax2.set_ylabel("modeled R_async / R_sync")
    ax2.set_xlim(1, 130)
    ax2.set_ylim(0.55, 1.32)
    ax2.text(
        2.1, 0.585,
        "mixed model: break-even r*(b) rises 0.80 → 1.0\n"
        "(b = 1, 8, 16: 0.80, 0.90, 0.96)",
        fontsize=8, color=MUTED,
    )
    ax2.set_title("Hypothetical mixed neural-backup model", fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
